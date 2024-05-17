import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class Criterion(nn.Module):
    def __init__(self,
                    num_classes,
                    frame_cls_loss= False,
                    frame_mask_loss= False,
                    action_cls_loss= False,
                    action_mask_loss= False,
                    action_metric_loss= False,
                    frame_action_cls_loss=False,
                    smooth_cls_loss=False,
                    smooth_mask_loss=False,
                    tmse_loss=False,
                    boundary_loss=False,
                    cluster_loss=False,
                    exist_loss=False,
                    dn_cls_loss=False,
                    dn_mask_loss=False,
                    frame_cls_weight=1,
                    frame_mask_weight=1,
                    action_cls_weight=1,
                    action_mask_weight=1,
                    action_metric_weight=1,
                    frame_action_cls_weight=1,
                    smooth_cls_weight=1,
                    smooth_mask_weight=1,
                    tmse_weight=1,
                    boundary_weight=1,
                    cluster_weight=1,
                    exist_weight=1,
                    dn_cls_weight=1,
                    dn_mask_weight=1,
                    boundary_len=5,
                    boundary_sigma=6,
                    action_len=1,
                    action_sigma=0.1,
                    focal_loss_gamma=2,
                    focal_loss_alpha=0.25,
                    eps=1e-6,
                    num_patterns=1,
                ): 
        super().__init__() 
        self.eps = eps
        self.num_classes = num_classes

        self.frame_cls_loss = frame_cls_loss
        self.frame_mask_loss = frame_mask_loss
        self.action_cls_loss = action_cls_loss
        self.action_mask_loss = action_mask_loss
        self.action_metric_loss = action_metric_loss
        self.frame_action_cls_loss = frame_action_cls_loss
        self.smooth_cls_loss = smooth_cls_loss
        self.smooth_mask_loss = smooth_mask_loss
        self.tmse_loss = tmse_loss
        self.boundary_loss = boundary_loss
        self.cluster_loss = cluster_loss
        self.exist_loss = exist_loss
        self.dn_cls_loss = dn_cls_loss
        self.dn_mask_loss = dn_mask_loss

        self.frame_cls_weight = frame_cls_weight
        self.frame_mask_weight = frame_mask_weight
        self.action_cls_weight = action_cls_weight
        self.action_mask_weight = action_mask_weight
        self.action_metric_weight = action_metric_weight
        self.frame_action_cls_weight = frame_action_cls_weight
        self.smooth_cls_weight = smooth_cls_weight
        self.smooth_mask_weight = smooth_mask_weight
        self.tmse_weight = tmse_weight
        self.boundary_weight = boundary_weight
        self.cluster_weight = cluster_weight
        self.exist_weight = exist_weight
        self.dn_cls_weight = dn_cls_weight
        self.dn_mask_weight = dn_mask_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        # self.loss_action_cls = self.loss_frame_cls
        self.loss_action_mask = self.loss_frame_mask

        self.boundary_len = boundary_len
        self.boundary_sigma = boundary_sigma
        self.action_len = action_len
        self.action_sigma = action_sigma
        self.num_patterns = num_patterns

    def focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none"):

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction=reduction)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        loss_mask = loss*mask.unsqueeze(-1).repeat(1,loss.shape[1])
        focal_loss_value = loss_mask.sum()/mask.sum()

        return focal_loss_value


    def focal_loss_without_logits(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none"):

        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction=reduction)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        loss_mask = loss*mask.unsqueeze(-1).repeat(1,loss.shape[1])
        focal_loss_value = loss_mask.sum()/mask.sum()

        return focal_loss_value


    def dice_loss(self, out_cls, gt_cls, b_mask):
        out_cls = out_cls.sigmoid()
        out_cls = out_cls * b_mask.detach()
        gt_cls = gt_cls.transpose(0,1) * b_mask.detach()
        input = out_cls.flatten(1) 
        target = gt_cls.flatten(1).float() 
        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + self.eps
        c = torch.sum(target * target, 1) + self.eps 
        d = (2 * a) / (b + c)
        
        dice_loss_value = 1 - d
        return dice_loss_value.mean()


    def dice_loss_without_logits(self, out_cls, gt_cls, b_mask):
        out_cls = out_cls * b_mask.detach()
        gt_cls = gt_cls.transpose(0,1) * b_mask.detach()
        input = out_cls.flatten(1) 
        target = gt_cls.flatten(1).float() 
        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + self.eps
        c = torch.sum(target * target, 1) + self.eps 
        d = (2 * a) / (b + c)
        
        dice_loss_value = 1 - d
        return dice_loss_value.mean()


    def loss_frame_cls(self, out_cls, gt_cls):
        out_cls = out_cls.transpose(-2, -1).contiguous()
        frame_cls_loss = F.cross_entropy(out_cls.view(-1, self.num_classes), gt_cls.flatten(), ignore_index=-1)
        return frame_cls_loss

    
    def loss_frame_mask(self, out_cls, gt_cls, b_mask):
        # boundary_points = torch.arange(1, self.action_len+1, device=out_cls.device)
        # gaussian_boundary = torch.exp(-0.5 * (boundary_points **2)/ (self.action_sigma **2))

        gt_cls_onehot = F.one_hot(gt_cls+1, num_classes=self.num_classes+1)
        gt_cls_onehot = gt_cls_onehot[:,1:]

        # get actions
        action_unique = torch.unique(gt_cls)
        action_mask = action_unique != -1
        action_unique = torch.masked_select(action_unique, action_mask)
        action_gt_cls_onehot = gt_cls_onehot[:,action_unique].float()
        # for i in range(len(action_unique)):
        #     action_gt = action_gt_cls_onehot[:,i]
        #     start_idx = ((action_gt[1:] - action_gt[:-1])==1).nonzero()
        #     end_idx = ((action_gt[1:] - action_gt[:-1])==-1).nonzero()
        #     for start in start_idx:
        #         action_gt[max(0,start-self.action_len+1):start+1] = gaussian_boundary.flip(0)[:min(self.action_len, start+1)]
        #     for end in end_idx:
        #         action_gt[end+1:min((end+self.action_len+1), (b_mask.nonzero()[-1]+1))] = gaussian_boundary[:min(self.action_len, (b_mask.nonzero()[-1]-end))]
        #     action_gt_cls_onehot[:,i] = action_gt

        action_out_cls = out_cls[action_unique]
        ### default dice loss
        frame_mask_loss = self.dice_loss(action_out_cls, action_gt_cls_onehot, b_mask)   
        # frame_mask_loss = self.focal_loss(action_out_cls, action_gt_cls_onehot, b_mask)

        return frame_mask_loss


    def loss_action_cls(self, out_cls, gt_cls, mask):
        gt_cls_onehot = F.one_hot(gt_cls+1, num_classes=self.num_classes+1)
        gt_cls_onehot = gt_cls_onehot[:, :,1:]
        gt_cls_onehot = gt_cls_onehot.flatten(0,1)
        out_cls = out_cls.transpose(-2, -1).flatten(0,1)
        mask = mask.flatten()
        action_cls_loss = self.focal_loss(out_cls, gt_cls_onehot, mask, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        return action_cls_loss


    def loss_action_metric(self, out_cls):
        b = out_cls.size(0)
        # out_cls = F.normalize(out_cls, dim=1)
        pred = out_cls.transpose(-2,-1) @ out_cls
        pred = pred.reshape(-1)
        gt_cls = torch.eye(self.num_classes, dtype=torch.float32, device=out_cls.device)
        target = gt_cls.expand((b, self.num_classes, self.num_classes)) 
        target = target.reshape(-1)
        # action_metric_loss = F.l1_loss(pred, target) 
        action_metric_loss = F.binary_cross_entropy_with_logits(pred, target) 
        return action_metric_loss
    

    def loss_frame_action_cls(self, f_out_cls, a_out_mask, gt_cls):
        # pred = f_out_cls * a_out_mask
        # pred = pred.transpose(-2, -1).contiguous()
        # frame_action_cls_loss = F.cross_entropy(pred.view(-1, self.num_classes), gt_cls.flatten(), ignore_index=-1)
        pred = F.softmax(f_out_cls, dim=-2) * a_out_mask.sigmoid()
        pred = (pred+self.eps).transpose(-2, -1).contiguous().log()
        frame_action_cls_loss = F.nll_loss(pred.view(-1, self.num_classes), gt_cls.flatten(), ignore_index=-1)
        return frame_action_cls_loss
    

    def loss_tmse(self, out_cls):
        tmse_loss = torch.clamp(F.mse_loss(F.log_softmax(out_cls[:,:,1:], dim=1), \
                                                F.log_softmax(out_cls.detach()[:,:,:-1], dim=1)), min=0, max=16)
        return tmse_loss

    def loss_boundary(self, bb_out, bgt_f, bmask):
        boundary_points = torch.arange(-self.boundary_len, self.boundary_len+1, device=bb_out.device)
        gaussian_boundary = torch.exp(-0.5 * (boundary_points **2)/ (self.boundary_sigma **2))

        diff_gt = torch.nonzero(bgt_f[1:] - bgt_f[:-1]).squeeze(1)[:-1] 
        # assert min(diff_gt) >= self.boundary_len 
        boundary_start = (diff_gt - self.boundary_len).clamp(min=0) #
        boundary_end =  diff_gt + self.boundary_len + 1
        gt_boundary = torch.zeros(len(bgt_f)).to(bb_out.device)
        for start, end in zip(boundary_start, boundary_end):
            if end-start < len(gaussian_boundary): 
                gt_boundary[start:end] = gaussian_boundary[start-end:]
            else:
                gt_boundary[start:end] = gaussian_boundary

        boundary_loss = F.binary_cross_entropy_with_logits(bb_out, gt_boundary, reduction='none')
        boundary_loss = boundary_loss * bmask 
        boundary_loss = boundary_loss.sum()/bmask.sum() 
        return boundary_loss
    

    def loss_cluster(self, out_cls, gt_cls):
        action_unique = torch.unique(gt_cls)
        action_mask = action_unique != -1 
        action_unique = torch.masked_select(action_unique, action_mask)
        out_cls = out_cls.transpose(0,1) # C,L -> L,C
        num_action = len(action_unique)
        cluster_loss = 0
        for action in action_unique:
            action_mask = gt_cls == action 
            action_out_cls = out_cls[action_mask]
            gt_action_out_cls = action_out_cls.new_ones(action_out_cls.shape[0]) 
            # action_out_cls = F.normalize(action_out_cls, dim=1) 
            mean_action_out_cls = action_out_cls.mean(dim=0) 
            # mean_action_out_cls = F.normalize(mean_action_out_cls, dim=0) 
            # pred_action_cls = torch.einsum('c, lc->l', mean_action_out_cls, action_out_cls) 
            pred_action_cls = action_out_cls - mean_action_out_cls
            pred_action_cls = pred_action_cls.pow(2).sum(dim=1).sqrt().mean() 
            cluster_loss += pred_action_cls/num_action 
            # cluster_loss += F.l1_loss((pred_action_cls/512).sigmoid(), gt_action_out_cls)/num_action 

        return cluster_loss
    

    def loss_exist(self, out_cls, gt_cls, b_mask):
        action_unique = torch.unique(gt_cls)
        action_mask = action_unique != -1
        action_unique = torch.masked_select(action_unique, action_mask)
        gt_exist = out_cls.new_zeros(self.num_classes)
        gt_exist[action_unique] = 1
        
        out_cls_max = torch.max(out_cls.sigmoid()*b_mask, dim=1)[0]
        exist_loss = F.binary_cross_entropy_with_logits(out_cls_max, gt_exist)
        return exist_loss

    def loss_dn_cls(self, b_cls_preds, b_cls_labels):
        pred = b_cls_preds.transpose(0,1)
        loss = F.cross_entropy(pred, b_cls_labels)
        return loss
            

    def loss_dn_mask(self, b_mask_preds, b_mask_labels, b_mask):
        b_mask_preds = b_mask_preds.sigmoid()
        b_mask_preds = b_mask_preds * b_mask.detach()
        input = b_mask_preds.flatten(1) 
        target = b_mask_labels.flatten(1).float() 
        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + self.eps
        c = torch.sum(target * target, 1) + self.eps 
        d = (2 * a) / (b + c)
        
        dice_loss_value = 1 - d
        return dice_loss_value.mean()

    def forward(self, gt_frame_class, action_output_class, action_output_mask, fpn_masks, with_aux_outputs=False, smooth_output=False,\
                with_aux_smooth_outputs=False, frame_output_class=False, boundary_pred=False, with_aux_boundary=False, mask_feat=False, \
                exist_pred=False, dn_cls=False, dn_mask=False):
        losses = {}
        losses['frame_cls_loss'] = losses['frame_mask_loss'] = losses['action_cls_loss'] \
            = losses["action_mask_loss"] = losses['action_metric_loss'] = losses['frame_action_cls_loss'] \
            = losses['smooth_cls_loss'] = losses['smooth_mask_loss'] = losses['tmse_loss'] = losses['boundary_loss'] \
            = losses['cluster_loss'] = losses['exist_loss'] = losses['dn_cls_loss'] = losses['dn_mask_loss'] =0
        bs, _ = fpn_masks[0].shape

        f_out_cls, gt_f_cls, masks, a_out_cls, a_out_mask, s_out, b_out = \
            frame_output_class, gt_frame_class, fpn_masks, action_output_class, action_output_mask, smooth_output, boundary_pred
        # frame loss
        # frame class loss
        if self.frame_cls_loss:
            f_out_cls_cat4cls = torch.cat(f_out_cls, -1) if type(f_out_cls) is tuple else f_out_cls 
            gt_f_cls_cat4cls = torch.cat(gt_f_cls, -1) if type(gt_f_cls) is tuple else gt_f_cls
            losses['frame_cls_loss'] = self.loss_frame_cls(f_out_cls_cat4cls, gt_f_cls_cat4cls) # cross-entropy loss
        # frame mask loss
        if self.frame_mask_loss:
            for b in range(bs):
                f_out_cls_cat4mask = torch.cat(f_out_cls, -1) if type(f_out_cls) is tuple else f_out_cls
                gt_f_cls_cat4mask = torch.cat(gt_f_cls, -1) if type(gt_f_cls) is tuple else gt_f_cls
                mask_cat4mask = torch.cat(masks, -1) if type(masks) is list else masks
                bf_out_cls_cat4mask = f_out_cls_cat4mask[b]
                bgt_f_cls_cat4mask = gt_f_cls_cat4mask[b]
                bmask_cat4mask = mask_cat4mask[b]
                losses['frame_mask_loss'] += self.loss_frame_mask(bf_out_cls_cat4mask, bgt_f_cls_cat4mask, \
                                            bmask_cat4mask) /bs

        # action loss
        if with_aux_outputs:
            num_action_lvl = len(a_out_cls)
            for la_out_cls, la_out_mask in zip(a_out_cls, a_out_mask):
                # action metric loss
                if self.action_metric_loss:
                    losses['action_metric_loss'] += self.loss_action_metric(la_out_cls)/num_action_lvl # l1 loss
                # action class loss
                if self.action_cls_loss:
                    gt_f_cls_select4cls = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                    mask_select4cls = masks[0] if type(masks) is list else masks
                    losses['action_cls_loss'] += self.loss_action_cls(la_out_mask, gt_f_cls_select4cls, mask_select4cls)/num_action_lvl # focal loss
                # action mask loss
                if self.action_mask_loss:
                    for b in range(bs):
                        a_out_mask_select4mask = la_out_mask
                        gt_f_cls_select4mask = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                        mask_select4mask = masks[0] if type(masks) is list else masks
                        ba_out_mask_select4mask = a_out_mask_select4mask[b]
                        bgt_f_cls_select4mask = gt_f_cls_select4mask[b]
                        bmask_select4mask = mask_select4mask[b]
                        losses['action_mask_loss'] += self.loss_action_mask(ba_out_mask_select4mask, bgt_f_cls_select4mask, bmask_select4mask)/(bs*num_action_lvl) # dice loss
        else:
            la_out_cls, la_out_mask = a_out_cls, a_out_mask
            # action metric loss
            if self.action_metric_loss:
                losses['action_metric_loss'] += self.loss_action_metric(la_out_cls) # l1 loss
            # action class loss
            if self.action_cls_loss:
                gt_f_cls_select4cls = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                mask_select4cls = masks[0] if type(masks) is list else masks
                losses['action_cls_loss'] += self.loss_action_cls(la_out_mask, gt_f_cls_select4cls, mask_select4cls) # focal loss
            # action mask loss
            if self.action_mask_loss:
                for b in range(bs):
                    a_out_mask_select4mask = la_out_mask
                    gt_f_cls_select4mask = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                    mask_select4mask = masks[0] if type(masks) is list else masks
                    ba_out_mask_select4mask = a_out_mask_select4mask[b]
                    bgt_f_cls_select4mask = gt_f_cls_select4mask[b]
                    bmask_select4mask = mask_select4mask[b]
                    losses['action_mask_loss'] += self.loss_action_mask(ba_out_mask_select4mask, bgt_f_cls_select4mask, bmask_select4mask)/bs # dice loss


        if self.frame_action_cls_loss:
            f_out_cls_select4cls = f_out_cls[0] if type(f_out_cls) is tuple else f_out_cls
            a_out_mask_select4cls = a_out_mask[0] if type(a_out_mask) is tuple else a_out_mask
            gt_f_cls_select4cls = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
            losses['frame_action_cls_loss'] = self.loss_frame_action_cls(f_out_cls_select4cls, a_out_mask_select4cls, gt_f_cls_select4cls)

        # smooth loss
        if with_aux_smooth_outputs:
            num_smooth_lvl = len(s_out)
            for ls_out in s_out:
                # smooth class loss
                if self.smooth_cls_loss:
                    gt_f_cls_select4cls = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                    mask_select4cls = masks[0] if type(masks) is list else masks
                    losses['smooth_cls_loss'] += self.loss_action_cls(ls_out, gt_f_cls_select4cls, mask_select4cls)/num_smooth_lvl # focal loss
                # action mask loss
                if self.smooth_mask_loss:
                    for b in range(bs):
                        ls_out4mask = ls_out
                        gt_f_cls_select4mask = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                        mask_select4mask = masks[0] if type(masks) is list else masks
                        bls_out4mask = ls_out4mask[b]
                        bgt_f_cls_select4mask = gt_f_cls_select4mask[b]
                        bmask_select4mask = mask_select4mask[b]
                        losses['smooth_mask_loss'] += self.loss_action_mask(bls_out4mask, bgt_f_cls_select4mask, bmask_select4mask)/(bs*num_smooth_lvl) # dice loss
        else:
            ls_out = s_out
            # action class loss
            if self.smooth_cls_loss:
                gt_f_cls_select4cls = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                mask_select4cls = masks[0] if type(masks) is list else masks
                losses['smooth_cls_loss'] += self.loss_action_cls(ls_out, gt_f_cls_select4cls, mask_select4cls) # focal loss
            # action mask loss
            if self.smooth_mask_loss:
                for b in range(bs):
                    ls_out4mask = ls_out
                    gt_f_cls_select4mask = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                    mask_select4mask = masks[0] if type(masks) is list else masks
                    bls_out4mask = ls_out4mask[b]
                    bgt_f_cls_select4mask = gt_f_cls_select4mask[b]
                    bmask_select4mask = mask_select4mask[b]
                    losses['smooth_mask_loss'] += self.loss_action_mask(bls_out4mask, bgt_f_cls_select4mask, bmask_select4mask)/bs # dice loss

        if self.tmse_loss:
            a_out_mask4tmse = torch.cat(a_out_mask, -1) if type(a_out_mask) is tuple else a_out_mask 
            # mask = fpn_masks[0].unsqueeze(1).repeat(1,self.num_classes,len(s_out))
            losses['tmse_loss'] = self.loss_tmse(a_out_mask4tmse) # cross-entropy loss

        # boundary loss
        if self.boundary_loss:
            if with_aux_boundary:
                num_boundary = len(b_out)
                for lb_out in b_out:
                    for b in range(bs):
                        blb_out = lb_out[b]
                        mask_select4boundary = masks[0] if type(masks) is list else masks
                        bmask_select4boundary = mask_select4boundary[b]
                        gt_f_cls_select4boundary = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                        bgt_f_cls_select4boundary = gt_f_cls_select4boundary[b]
                        losses['boundary_loss'] += self.loss_boundary(blb_out, bgt_f_cls_select4boundary, bmask_select4boundary)/(bs*num_boundary)
            else:
                for b in range(bs):
                    bb_out = b_out[b]
                    mask_select4boundary = masks[0] if type(masks) is list else masks
                    bmask_select4boundary = mask_select4boundary[b]
                    gt_f_cls_select4boundary = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                    bgt_f_cls_select4boundary = gt_f_cls_select4boundary[b]
                    losses['boundary_loss'] += self.loss_boundary(bb_out, bgt_f_cls_select4boundary, bmask_select4boundary)/bs


        if self.cluster_loss:
            f_out_cls_select4cluster = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
            for b in range(bs):
                bmask_feat = mask_feat[b]
                bf_out_cls_select4cluster = f_out_cls_select4cluster[b]
                losses['cluster_loss'] += self.loss_cluster(bmask_feat, bf_out_cls_select4cluster)/bs

        
        if self.exist_loss:
            gt_f_cls_select4exist = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
            mask_select4exist = masks[0] if type(masks) is list else masks
            if with_aux_outputs:
                # num_action_lvl = len(exist_pred)
                num_action_lvl = len(a_out_mask)
                # for exist in exist_pred:
                for exist in a_out_mask:
                    for b in range(bs):
                        b_exist = exist[b]
                        b_mask = mask_select4exist[b]
                        bgt_f_cls_select4exist = gt_f_cls_select4exist[b] 
                        losses['exist_loss'] += self.loss_exist(b_exist, bgt_f_cls_select4exist, b_mask)/(bs*num_action_lvl)
            else:
                exist = a_out_mask
                for b in range(bs):
                    b_exist = exist[b]
                    b_mask = mask_select4exist[b]
                    bgt_f_cls_select4exist = gt_f_cls_select4exist[b] 
                    losses['exist_loss'] += self.loss_exist(b_exist, bgt_f_cls_select4exist, b_mask)/bs

        # dn loss
        # dn cls loss
        if with_aux_outputs and (self.dn_cls_loss or self.dn_mask_loss):
            num_action_lvl = len(dn_cls)
            for l_dn_cls, l_dn_mask in zip(dn_cls, dn_mask):
                gt_labels4dn = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                # dn cls loss
                if self.dn_cls_loss:
                    for b in range(bs):
                        bl_dn_cls = l_dn_cls[b]
                        bgt_labels4dn = gt_labels4dn[b]
                        actions = torch.unique(bgt_labels4dn)
                        actions = actions[1:] if -1 in actions else actions
                        num_actions = actions.size()[0]
                        b_cls_labels = actions.repeat(self.num_patterns)
                        b_cls_preds = bl_dn_cls[:, :num_actions*self.num_patterns]
                        losses['dn_cls_loss'] += self.loss_dn_cls(b_cls_preds, b_cls_labels)/(bs*num_action_lvl)

                # dn mask loss
                if self.dn_mask_loss:
                    mask_dn = masks[0] if type(masks) is list else masks
                    for b in range(bs):
                        bl_dn_mask = l_dn_mask[b]
                        bgt_labels4dn = gt_labels4dn[b]
                        bmask_dn = mask_dn[b]
                        actions = torch.unique(bgt_labels4dn)
                        actions = actions[1:] if -1 in actions else actions
                        num_actions = actions.size()[0]
                        b_mask_labels = torch.zeros((num_actions, bl_dn_mask.size(-1)), device=bl_dn_mask.device).type(torch.bool)
                        for i in range(num_actions):
                            b_mask_labels[i] = (bgt_labels4dn == actions[i])
                        b_mask_labels = b_mask_labels.repeat(self.num_patterns, 1)
                        b_mask_preds = bl_dn_mask[:num_actions*self.num_patterns]
                        losses['dn_mask_loss'] += self.loss_dn_mask(b_mask_preds, b_mask_labels, bmask_dn)/(bs*num_action_lvl) # dice loss
        else:
            l_dn_cls, l_dn_mask = dn_cls, dn_mask
            gt_labels4dn = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
            # dn cls loss
            if self.dn_cls_loss:
                for b in range(bs):
                    bl_dn_cls = l_dn_cls[b]
                    bgt_labels4dn = gt_labels4dn[b]
                    actions = torch.unique(bgt_labels4dn)
                    actions = actions[1:] if -1 in actions else actions
                    num_actions = actions.size()[0]
                    b_cls_labels = actions.repeat(self.num_patterns)
                    b_cls_preds = bl_dn_cls[:, :, num_actions*self.num_patterns]
                    losses['dn_cls_loss'] += self.loss_dn_cls(b_cls_preds, b_cls_labels)/bs

                # dn mask loss
                if self.dn_mask_loss:
                    mask_dn = masks[0] if type(masks) is list else masks
                    for b in range(bs):
                        bl_dn_mask = l_dn_mask[b]
                        bgt_labels4dn = gt_labels4dn[b]
                        bmask_dn = mask_dn[b]
                        actions = torch.unique(bgt_labels4dn)
                        actions = actions[1:] if -1 in actions else actions
                        num_actions = actions.size()[0]
                        b_mask_labels = torch.zeros((num_actions, bl_dn_mask.size(-1)), device=bl_dn_mask.device).type(torch.bool)
                        for i in range(num_actions):
                            b_mask_labels[i] = (bgt_labels4dn == actions[i])
                        b_mask_labels = b_mask_labels.repeat(self.num_patterns, 1)
                        b_mask_preds = bl_dn_mask[:, :, :num_actions*self.num_patterns]
                        losses['dn_mask_loss'] += self.loss_dn_mask(b_mask_preds, b_mask_labels, bmask_dn)/bs


        losses['final_loss'] = self.frame_cls_weight*losses['frame_cls_loss']+\
                                self.frame_mask_weight*losses['frame_mask_loss']+\
                                self.action_cls_weight*losses['action_cls_loss']+\
                                self.action_mask_weight*losses['action_mask_loss']+\
                                self.action_metric_weight*losses['action_metric_loss']+\
                                self.frame_action_cls_weight*losses['frame_action_cls_loss']+\
                                self.smooth_cls_weight*losses['smooth_cls_loss']+\
                                self.smooth_mask_weight*losses['smooth_mask_loss']+\
                                self.tmse_weight*losses['tmse_loss']+\
                                self.boundary_weight*losses['boundary_loss']+\
                                self.cluster_weight*losses['cluster_loss']+\
                                self.exist_weight*losses['exist_loss']+\
                                self.dn_cls_weight*losses['dn_cls_loss']+\
                                self.dn_mask_weight*losses['dn_mask_loss']
        

        losses.pop('frame_cls_loss') if not self.frame_cls_loss else None
        losses.pop('frame_mask_loss') if not self.frame_mask_loss else None
        losses.pop('action_cls_loss') if not self.action_cls_loss else None
        losses.pop('action_mask_loss') if not self.action_mask_loss else None
        losses.pop('action_metric_loss') if not self.action_metric_loss else None
        losses.pop('frame_action_cls_loss') if not self.frame_action_cls_loss else None
        losses.pop('smooth_cls_loss') if not self.smooth_cls_loss else None
        losses.pop('smooth_mask_loss') if not self.smooth_mask_loss else None
        losses.pop('tmse_loss') if not self.tmse_loss else None
        losses.pop('boundary_loss') if not self.boundary_loss else None
        losses.pop('cluster_loss') if not self.cluster_loss else None
        losses.pop('exist_loss') if not self.exist_loss else None
        losses.pop('dn_cls_loss') if not self.dn_cls_loss else None
        losses.pop('dn_mask_loss') if not self.dn_mask_loss else None
        
        return losses