import os
import copy
import random
import numpy as np
import random
import torch
import torch.nn.functional as F

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r] 
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len) 
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32) 

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1) #[13, 2]
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs 

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0: 
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict 


def mask_feats(data_dict, mask_ratio=0):
    feats = data_dict['feats']
    mask = torch.rand(feats.shape[1]) > mask_ratio
    masked_feats = feats*mask.unsqueeze(0).repeat(feats.shape[0], 1)
    data_dict['feats'] = masked_feats

    return data_dict


def scale_feats(data_dict, scale_ratio=[0.5, 1.5]):
    feats = data_dict['feats']
    segments = data_dict['segments']
    labels = data_dict['labels']
    action_list = segment2list(segments, labels)
    scale = (torch.rand(1)*(scale_ratio[1] - scale_ratio[0]) + scale_ratio[0]).item()

    scale_feats = F.interpolate(feats.unsqueeze(0), scale_factor=scale, mode='linear', align_corners=False).squeeze(0)
    scale_action_list = F.interpolate(action_list.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode='nearest').squeeze(0).squeeze(0)
    scale_segments = list2segment(scale_action_list)

    data_dict['labels'] = action_list2label(scale_action_list)
    data_dict['feats'] = scale_feats
    data_dict['segments'] = scale_segments
    data_dict['duration'] = scale_feats.shape[1]

    return data_dict

def segment2list(segments, labels):
    action_list = torch.zeros(int(segments[-1,-1]+1))
    for i, label in enumerate(labels):
        s, e = int(segments[i][0]), int(segments[i][1])
        action_list[s:e+1] = label
    return action_list

def list2segment(action_list):
    diff_idx = (action_list[1:] != action_list[:-1]).nonzero().squeeze(1)
    start_idx = torch.cat((torch.tensor([0]), diff_idx+1))
    end_idx = torch.cat((diff_idx, torch.tensor([len(action_list)-1])))
    segments = torch.stack((start_idx, end_idx), dim=1)

    return segments

def action_list2label(action_list):
    diff_idx = (action_list[1:] != action_list[:-1]).nonzero().squeeze(1)
    label = torch.cat((action_list[diff_idx], torch.tensor([action_list[-1]]))) 

    return label
