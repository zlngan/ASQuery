import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import InstanceNorm1d as InsNorm1d

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from .heads import ActionHead
from .losses import Criterion


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = [],
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )
        self.norm.append(LayerNorm(num_classes))

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)): 
            cur_out = cur_feat
            for idx in range(len(self.head)): 
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask) 
            cur_logits = self.norm[-1](cur_logits)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), ) 

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
        conv_dila=[[1,1],[1,1],[1,1,1,1,1]],
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(  #[1,2,4,8,16,32]
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range #[(0,4),(4,8),(8,16),(16,32),(32,64),(64,100000)]
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1]) #[19,19,19,19,19,19]
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            # assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor # 576=32*18

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius'] # 1.5
        self.train_loss_weight = train_cfg['loss_weight'] # 1.0
        self.train_cls_prior_prob = train_cfg['cls_prior_prob'] # 0.01
        self.train_dropout = train_cfg['dropout'] # 0.0
        self.train_droppath = train_cfg['droppath'] # 0.1
        self.train_label_smoothing = train_cfg['label_smoothing'] # 0.0

        self.decoder_dropout = train_cfg['decoder_dropout'] # 0.0
        self.decoder_droppath = train_cfg['decoder_droppath'] # 0.1

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh'] # 0.001
        self.test_pre_nms_topk = test_cfg['pre_nms_topk'] # 2000
        self.test_iou_threshold = test_cfg['iou_threshold'] # 0.1
        self.test_min_score = test_cfg['min_score'] # 0.001
        self.test_max_seg_num = test_cfg['max_seg_num'] # 200
        self.test_nms_method = test_cfg['nms_method'] # soft
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh'] # 0.05
        self.test_multiclass_nms = test_cfg['multiclass_nms'] # True
        self.test_nms_sigma = test_cfg['nms_sigma'] # 0.5
        self.test_voting_thresh = test_cfg['voting_thresh'] # 0.7

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv', 'asformer']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'conv_dila': conv_dila,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                }
            )
        elif backbone_type == 'conv':
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln,
                    'conv_dila': conv_dila,
                }
            )
        else:
            self.backbone = make_backbone(
                'asformer',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'arch': backbone_arch,
                    'conv_dila': conv_dila,
                    'scale_factor': scale_factor,
                    'win_size': self.mha_win_size[0],
                    'dropout': self.train_dropout,
                })
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim) # 512

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck( # identity
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln,
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks) 
        fpn_feats, fpn_masks = self.neck(feats, masks) 

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats) 

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len 
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None] 

        # push to device
        batched_inputs = batched_inputs.to(self.device) # [2, 2048, 2304]
        batched_masks = batched_masks.unsqueeze(1).to(self.device) # [2,1,2304]

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0) #[4536,4]
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1) #[4536, 4]

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2) #[4536, 4, 2]
        left = concat_points[:, 0, None] - gt_segs[:, :, 0] #[4536,4]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None] # [4536, 4]
        reg_targets = torch.stack((left, right), dim=-1) #

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius # [4536,4]
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius # [4536,4]
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack( # [4536,4,2]
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0 
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and( 
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets #[4536,20] [4536, 2]

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask) # [2,4536]

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = None
                # segs, scores, labels = batched_nms(
                #     segs, scores, labels,
                #     self.test_iou_threshold,
                #     self.test_min_score,
                #     self.test_max_seg_num,
                #     use_soft_nms = (self.test_nms_method == 'soft'),
                #     multiclass = self.test_multiclass_nms,
                #     sigma = self.test_nms_sigma,
                #     voting_thresh = self.test_voting_thresh
                # )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results


@register_meta_arch("MaskPointTransformer")
class MaskPtTransformer(PtTransformer):
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,               # other cfg for testing
        n_dec_layers=2,         # number of decoder layers
        n_blocks=6,             # number of blocks
        conv_dila=[[1,1],[1,1],[1,1,1,1,1]],
        ignore_index=-1,
        with_aux_outputs=False,
        frame_cls_loss= False,
        frame_mask_loss= False,
        action_cls_loss= False,
        action_mask_loss= False,
        frame_action_cls_loss= False,
        tmse_loss= False,
        boundary_loss= False,
        frame_cls_weight=1,
        frame_mask_weight=1,
        action_cls_weight=1,
        tmse_weight=1,
        boundary_weight=1,
        focal_loss_gamma=2,
        focal_loss_alpha=0.25,
        with_frame=False,
        with_action=False,
        with_frame_action=False,
        with_query_pos=False, 
        with_feat_pos=True,
        with_aux_smooth_outputs=False,
        boundary_len=5,
        boundary_sigma=6,
        boundary_thr=0.1,
        with_aux_boundary=False,
        action_len=1,
        action_sigma=0.1,
        normalize_before=False,
        downsample_rate=1,
        decoder_type = 'transformer',
        action_cls_cross_loss=False,
        action_cls_smooth_loss=False
    ):
        super().__init__(
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,
        conv_dila=conv_dila,
        )

        self.with_frame=with_frame
        self.with_action=with_action
        self.with_frame_action=with_frame_action
        self.boundary_loss = boundary_loss
        self.with_aux_outputs = with_aux_outputs
        self.with_aux_smooth_outputs = with_aux_smooth_outputs
        self.with_aux_boundary = with_aux_boundary
        self.boundary_thr = boundary_thr
        self.downsample_rate = downsample_rate
        self.ignore_index = ignore_index
        self.num_level = backbone_arch[-1]+1
        self.decoder_type = decoder_type

        self.frame_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls'],
            )

        self.action_head = ActionHead( # MaskHead
                        n_embd = head_dim,
                        max_seq_len = max_seq_len,
                        num_classes = num_classes,
                        n_head = n_head,  
                        n_dec_layers = n_dec_layers,
                        n_blocks = n_blocks,
                        attn_pdrop = self.decoder_dropout,
                        proj_pdrop = self.decoder_dropout,
                        path_pdrop = self.decoder_droppath,
                        with_query_pos = with_query_pos,
                        with_feat_pos = with_feat_pos,
                        with_aux_outputs=with_aux_outputs,
                        with_aux_boundary=with_aux_boundary,
                        normalize_before=normalize_before,
                        decoder_type=decoder_type,
                        ) 
        

        self.criterion = Criterion(
            num_classes,
            frame_cls_loss=frame_cls_loss,
            frame_mask_loss=frame_mask_loss,
            action_cls_loss=action_cls_loss,
            action_mask_loss=action_mask_loss, 
            tmse_loss=tmse_loss,
            boundary_loss=boundary_loss,
            frame_cls_weight=frame_cls_weight,
            frame_mask_weight=frame_mask_weight,
            action_cls_weight=action_cls_weight,
            tmse_weight=tmse_weight,
            boundary_weight=boundary_weight,
            boundary_len=boundary_len,
            boundary_sigma=boundary_sigma,
            action_len=action_len,
            action_sigma=action_sigma,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            action_cls_cross_loss=action_cls_cross_loss,
            action_cls_smooth_loss=action_cls_smooth_loss,
        )
        

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks, multi_level_labels = self.preprocessing(video_list, self.num_level)

        # forward the network (backbone -> neck -> heads)
        # backbone
        feats, masks = self.backbone(batched_inputs, batched_masks) 
        # neck
        mask_feat, fpn_feats, fpn_masks = self.neck(feats, masks) 
        # cls head
        out_cls_logits = self.frame_head(fpn_feats, fpn_masks)
        # action head
        fpn_masks = [x.squeeze(1) for x in fpn_masks] #[B,1,L] -> [B,L]

        # return loss during training
        if self.training:
            output_mask, output_boundary = self.action_head(mask_feat, fpn_feats, fpn_masks)
            losses = self.criterion(multi_level_labels, output_mask, fpn_masks, self.with_aux_outputs,\
                            with_aux_boundary=self.with_aux_boundary, mask_feat=mask_feat, boundary_pred=output_boundary,
                            frame_output_class=out_cls_logits)
            return losses
        else:
            output_mask, output_boundary = self.action_head(mask_feat, fpn_feats, fpn_masks)
            results = self.inference(multi_level_labels, output_mask, output_boundary, frame_cls_logits=out_cls_logits,)
            return results

    def inference(self, multi_level_labels, action_mask_output, output_boundary, frame_cls_logits=False):
        results = []
        for i in range(len(multi_level_labels)):
            # gt
            gt_vid = multi_level_labels[i]
            duration = gt_vid.shape[-1]
            if self.with_frame:
                frame_cls_logits = frame_cls_logits[0] if isinstance(frame_cls_logits, tuple) else frame_cls_logits
                pred = torch.argmax(frame_cls_logits[i].transpose(0,1), -1)
                pred = pred.repeat_interleave(self.downsample_rate)[:duration]
                results.append((gt_vid, pred))
            if self.with_action:
                action_mask_output = action_mask_output[-1] if self.with_aux_outputs else action_mask_output
                pred = torch.argmax(action_mask_output[i].transpose(0,1)[:duration], -1)
                if self.boundary_loss:
                    output_boundary = output_boundary[-1] if self.with_aux_boundary else output_boundary
                    pred_segs_1 = self.get_boundary(output_boundary, duration//self.downsample_rate, i)
                    pred_segs = pred_segs_1[1]
                    for seg in pred_segs:
                        seg_s, seg_e = int(seg[0]), int(seg[1])
                        pred[seg_s:seg_e] = torch.argmax(torch.bincount(pred[seg_s:seg_e]))

                pred = pred.repeat_interleave(self.downsample_rate)[:duration] 
                # results.append((gt_vid, pred, pred_segs)) 
                results.append((gt_vid, pred)) 

            if self.with_frame_action:
                pred = torch.argmax(torch.softmax(frame_cls_logits[i],0).transpose(0,1)[:duration]*action_mask_output[i].sigmoid().transpose(0,1)[:duration], -1)
                results.append((gt_vid, pred)) 

        return results
    

    @torch.no_grad()
    def preprocessing(self, video_list, num_level, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        gt_segments = [x['segments'].to(self.device) for x in video_list]
        gt_labels = [x['labels'].to(self.device) for x in video_list]
        duration = [x['duration'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        device = feats[0].device
        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len 
            multi_level_labels=tuple()
            batched_labels = torch.ones(len(video_list), max_len) * self.ignore_index
            for i in range(len(video_list)):
                for seg, label in zip(gt_segments[i], gt_labels[i]):
                    batched_labels[i, int(seg[0]):int(seg[1]+1)] = label
            multi_level_labels += (batched_labels.long().to(device),)
            for i in range(1, num_level):
                lvl_batched_labels = F.interpolate(batched_labels.unsqueeze(1), size=int(max_len*0.5**i), mode='nearest')
                multi_level_labels += (lvl_batched_labels.squeeze(1).long().to(device),)

            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)
            multi_level_labels=tuple()
            batched_labels = torch.ones(len(video_list), max_len, dtype=torch.long, device=device) * self.ignore_index
            for i in range(len(video_list)):
                for seg, label in zip(gt_segments[i], gt_labels[i]):
                    batched_labels[i, int(seg[0]):int(seg[1]+1)] = label
                    batched_labels_duration = batched_labels[i, :duration[i]]
                multi_level_labels += (batched_labels_duration,)
            
        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None] 
        # push to device
        batched_inputs = batched_inputs.to(self.device) # [2, 2048, 2304]
        batched_masks = batched_masks.unsqueeze(1).to(self.device) # [2,1,2304]

        return batched_inputs, batched_masks, multi_level_labels


    @torch.no_grad()
    def get_boundary(self, boundary_output, duration, idx):
        boundary_pred = boundary_output[idx].sigmoid()[:duration]
        local_max = F.max_pool1d(boundary_pred.unsqueeze(0), 15, stride=1, padding=7)
        local_max[local_max<self.boundary_thr] = 0

        keep_max = local_max.squeeze(0) == boundary_pred
        max_idx = torch.nonzero(keep_max).squeeze(1)
        max_idx = torch.masked_select(max_idx, max_idx != 0)
        max_idx = torch.masked_select(max_idx, max_idx != duration-1)
        max_idx = torch.masked_select(max_idx, max_idx != duration-2) 
        pred_seg_s = torch.cat((max_idx.new_tensor([0]), max_idx+1))
        pred_seg_e = torch.cat((max_idx+1, max_idx.new_tensor([duration])))
        seg_list = []
        for seg in zip(pred_seg_s, pred_seg_e):
            seg_s, seg_e = int(seg[0]), int(seg[1])
            seg_list.append([seg_s, seg_e])
        return boundary_pred, seg_list
