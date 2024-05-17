import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats, mask_feats, scale_feats

@register_dataset("breakfast")
@register_dataset("50salads")
@register_dataset("gtea")
class BreakfastDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        anno_dir,       # dir of annotations
        gt_dir,
        mapping_file,
        fps,
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        mask_ratio=0,
        scale_ratio=None,
        scale_prob=0,
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(anno_dir)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext


        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.fps = fps
        self.mask_ratio = mask_ratio
        self.scale_ratio = scale_ratio
        self.scale_prob = scale_prob

        # load database and select the subset
        self.mapping_file = mapping_file
        label_dict = self._get_label_dict(self.mapping_file)
        self.anno_dir = anno_dir
        self.anno_file = f'{self.anno_dir}/{self.split[0]}.split{self.split[1]}.bundle'
        self.gt_dir = gt_dir

        dict_db = self._load_annotaion(self.anno_file, gt_dir, label_dict)
        
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'breakfast',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _get_label_dict(self, mapping_file):
        mapping_content = open(mapping_file, 'r').readlines()
        label_dict = {}
        for content in mapping_content:
            label_id, label_name = content.split()
            label_dict[label_name] = int(label_id)
        return label_dict

    def _load_annotaion(self, anno_file, gt_dir, label_dict):
        dict_db = tuple()
        anno_file_names = open(anno_file).readlines()
        duration_list = []
        for anno_file_name in anno_file_names:
            anno_file_path = os.path.join(gt_dir, anno_file_name.strip())
            anno_file_content = open(anno_file_path).readlines()
            if self.is_training:
                anno_file_content = [x for i, x in enumerate(anno_file_content) if i % self.downsample_rate == 0] 
            duration = len(anno_file_content)
            segments, labels = self._get_seg_label(anno_file_content, label_dict)
            dict_db += ({'id': os.path.splitext(anno_file_name)[0],
                         'fps' : self.fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )
            duration_list.append(duration)
        return dict_db

    def _get_seg_label(self, anno_cont, label_dict):
        segments, labels = [], []
        f_s = []
        for idx in range(1, len(anno_cont)):
            if anno_cont[idx] != anno_cont[idx-1]:
                f_s.append(idx)
        segments.append([0, f_s[0]-1])
        [segments.append([f_s[i], len(anno_cont)-1]) if i == len(f_s)-1 else segments.append([f_s[i], f_s[i+1]-1]) for i in range(len(f_s))]
        labels.append(label_dict[anno_cont[f_s[0]-1].strip()])
        [labels.append(label_dict[anno_cont[f_s[i]].strip()]) for i in range(len(f_s))]
        assert len(segments) == len(labels)
        segments = np.asarray(segments, dtype=np.float32) 
        labels = np.asarray(labels, dtype=np.int64)

        return segments, labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32) 

        # feats /= np.linalg.norm(feats, axis=0, keepdims=True) # normalize

        # deal with downsampling (= increased feat stride)
        feats = feats[:, ::self.downsample_rate]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride #0.5*16/4=2
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats)) 

        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] 
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (self.scale_ratio is not None):
            # data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)
            scale_prob = torch.rand(1)<self.scale_prob
            data_dict = scale_feats(data_dict, scale_ratio=self.scale_ratio) if scale_prob else data_dict
            # data_dict = mask_feats(data_dict, self.mask_ratio)

        return data_dict
