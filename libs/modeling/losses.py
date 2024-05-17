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
                    frame_action_cls_loss= False,
                    tmse_loss=False,
                    boundary_loss=False,

                    action_cls_cross_loss= False,
                    action_cls_smooth_loss=False,

                    frame_cls_weight=1,
                    frame_mask_weight=1,
                    action_cls_weight=1,
                    action_mask_weight=1,
                    frame_action_cls_weight=1,

                    tmse_weight=1,
                    boundary_weight=1,
                    cluster_weight=1,
                    boundary_len=5,
                    boundary_sigma=6,
                    action_len=1,
                    action_sigma=0.1,
                    focal_loss_gamma=2,
                    focal_loss_alpha=0.25,
                    eps=1e-6,
                ): 
        super().__init__() 
        self.eps = eps
        self.num_classes = num_classes

        self.frame_cls_loss = frame_cls_loss
        self.frame_mask_loss = frame_mask_loss
        self.action_cls_loss = action_cls_loss
        self.action_mask_loss = action_mask_loss
        self.tmse_loss = tmse_loss
        self.boundary_loss = boundary_loss
        self.frame_action_cls_loss = frame_action_cls_loss
        self.action_cls_cross_loss = action_cls_cross_loss
        self.action_cls_smooth_loss = action_cls_smooth_loss

        self.frame_cls_weight = frame_cls_weight
        self.frame_mask_weight = frame_mask_weight
        self.action_cls_weight = action_cls_weight
        self.action_mask_weight = action_mask_weight
        self.frame_action_cls_weight = frame_action_cls_weight

        self.tmse_weight = tmse_weight
        self.boundary_weight = boundary_weight
        self.cluster_weight = cluster_weight

        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        self.loss_action_mask = self.loss_frame_mask

        self.boundary_len = boundary_len
        self.boundary_sigma = boundary_sigma
        self.action_len = action_len
        self.action_sigma = action_sigma

    def focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none"):

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy(p, targets.float(), reduction=reduction)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        mask_action = mask.unsqueeze(1)
        loss_mask = loss*mask_action # loss [num_action, len]
        focal_loss_value = loss_mask.sum()/mask_action.sum()

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
        out_cls = out_cls.sigmoid() # [num_actions, len]
        out_cls = out_cls * b_mask.detach() # 
        gt_cls = gt_cls * b_mask.detach() # [num_actions, len]
        input = out_cls 
        target = gt_cls.float() 
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
        action_unique = action_unique[action_unique!= -1]
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
        frame_mask_loss = self.dice_loss(action_out_cls, action_gt_cls_onehot.transpose(0,1), b_mask) 

        return frame_mask_loss


    def loss_action_cls(self, out_cls, gt_cls, mask):
        gt_cls_onehot = F.one_hot(gt_cls+1, num_classes=self.num_classes+1)
        gt_cls_onehot = gt_cls_onehot[:, :,1:]
        gt_cls_onehot = gt_cls_onehot.flatten(0,1)
        out_cls = out_cls.transpose(-2, -1).flatten(0,1)
        mask = mask.flatten()
        action_cls_loss = self.focal_loss(out_cls, gt_cls_onehot, mask, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        return action_cls_loss
    

    def loss_frame_action_cls(self, f_out_cls, a_out_mask, gt_cls):
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
        boundary_start = (diff_gt - self.boundary_len).clamp(min=0) 
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

    
    def loss_action_cls_cross(self, out_cls, gt_cls, mask):
        # out_cls [2,48,15360] gt_cls [2,15360] mask [2,15360]
        out_cls = out_cls.transpose(-2, -1).flatten(0,1)
        gt_cls = gt_cls.flatten()
        cls_cross_loss = F.cross_entropy(out_cls, gt_cls, ignore_index=-1)
        return cls_cross_loss
    
    def loss_action_cls_smooth(self, out_cls, mask):
        bs = mask.size(0)
        cls_smooth_loss = 0
        for i in range(bs):
            bout_cls = out_cls[i]
            bmask = mask[i]
            len_bmask = bmask.sum()
            cls_smooth_loss += torch.clamp(F.mse_loss(F.log_softmax(bout_cls[:,1:len_bmask+1], dim=0), \
                                                F.log_softmax(bout_cls.detach()[:,:len_bmask], dim=0)), min=0, max=16)
        return cls_smooth_loss/bs
    

    def forward(self, gt_cls, action_output_mask, fpn_masks, with_aux_outputs=False, 
                 frame_output_class=False, with_aux_boundary=False, mask_feat=False, boundary_pred=False):
        losses = {}
        losses['frame_cls_loss'] = losses['frame_mask_loss'] = losses['action_cls_loss'] = losses['frame_action_cls_loss']\
            = losses["action_mask_loss"] = losses['tmse_loss'] = losses['boundary_loss']  = losses['action_cls_cross_loss'] \
            = losses['action_cls_smooth_loss'] =0
        bs, _ = fpn_masks[0].shape

        f_out_cls, gt_f_cls, masks, a_out_mask, b_out = \
            frame_output_class, gt_cls, fpn_masks, action_output_mask, boundary_pred
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
            num_action_lvl = len(a_out_mask)
            for la_out_mask in  a_out_mask:
                # action class loss
                gt_f_cls_select4cls = gt_f_cls[0] if type(gt_f_cls) is tuple else gt_f_cls
                mask_select4cls = masks[0] if type(masks) is list else masks
                if self.action_cls_loss:
                    losses['action_cls_loss'] += self.loss_action_cls(la_out_mask, gt_f_cls_select4cls, mask_select4cls)/num_action_lvl # focal loss
                if self.action_cls_cross_loss:
                    losses['action_cls_cross_loss'] += self.loss_action_cls_cross(la_out_mask, gt_f_cls_select4cls, mask_select4cls)/num_action_lvl # cross loss
                if self.action_cls_smooth_loss:
                    losses['action_cls_smooth_loss'] += self.loss_action_cls_smooth(la_out_mask, mask_select4cls)/num_action_lvl # smooth loss
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
            la_out_mask = a_out_mask
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


        losses['final_loss'] = self.frame_cls_weight*losses['frame_cls_loss']+\
                                self.frame_mask_weight*losses['frame_mask_loss']+\
                                self.action_cls_weight*losses['action_cls_loss']+\
                                self.frame_action_cls_weight*losses['frame_action_cls_loss']+\
                                self.tmse_weight*losses['tmse_loss']+\
                                self.boundary_weight*losses['boundary_loss']+\
                                self.action_mask_weight*losses['action_mask_loss']+\
                                self.action_cls_weight*losses['action_cls_cross_loss']+\
                                self.action_cls_weight*losses['action_cls_smooth_loss']
        

        losses.pop('frame_cls_loss') if not self.frame_cls_loss else None
        losses.pop('frame_mask_loss') if not self.frame_mask_loss else None
        losses.pop('action_cls_loss') if not self.action_cls_loss else None
        losses.pop('action_mask_loss') if not self.action_mask_loss else None
        losses.pop('frame_action_cls_loss') if not self.frame_action_cls_loss else None
        losses.pop('tmse_loss') if not self.tmse_loss else None
        losses.pop('boundary_loss') if not self.boundary_loss else None
        losses.pop('action_cls_cross_loss') if not self.action_cls_cross_loss else None
        losses.pop('action_cls_smooth_loss') if not self.action_cls_smooth_loss else None
        
        return losses