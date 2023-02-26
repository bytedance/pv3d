# Copyright 2022 ByteDance and/or its affiliates.
#
# Copyright (2022) PV3D Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.


import os, json
import torch
import numpy as np
from lib.datasets.base_dataset import BaseDataset
from lib.utils.dataset import fetch_video_clip

_CONSTANTS = {
    'dataset_name': 'voxceleb', 
    'video_ext': 'mp4', 
    'video_folder': 'recrop_faces', 
    'camera_folder': "smoothed_camera2world", 
    'video_list': 'voxceleb.list', 
}

class VoxCelebDataset(BaseDataset):

    def __init__(self, config, dataset_type, *args, **kwargs):
        super().__init__(config, _CONSTANTS['dataset_name'], dataset_type)

        self._data_dir = config.data_dir
        self._num_frames = config.num_frames
        self._bi_frame = config.bi_frame
        
        if not os.path.exists(self._data_dir):
            raise RuntimeError(
                 f"Data folder {self._data_dir} for {_CONSTANTS['dataset_name']} not found."
            )

        self._load()

    def __getitem__(self, index):
        sample = fetch_video_clip(self.idx_db[index],  
                                  self._num_frames, 
                                  self.dataset_type,
                                  bi_frame=self.bi_frame)
        try:
            sample["real_cam"] = self.get_label(index, sample)
        except:
            print(self.idx_db[index])
        if self.skip_processor:
            sample["frames"] = torch.from_numpy(sample["frames"])
            if not self.bi_frame and sample["frames"].shape[0] == 1:
                sample["frames"] = sample["frames"].squeeze(0).permute(2, 0, 1)
            return sample

        for processor_name in self.processor_map:
            processor = getattr(self, processor_name)
            sample = processor(sample)

        if not self.bi_frame and sample["frames"].shape[0] == 1:
            sample["frames"] = sample["frames"].squeeze(0)

        return sample
    
    def __len__(self):
        return len(self.idx_db)

    def _load(self):
        if self.dataset_type in ['train', 'val']:
            self._load_dataset()
        else:
            raise RuntimeError("Invalid {} {} set".format(
                                _CONSTANTS['dataset_name'], 
                                self.dataset_type))

    def _load_dataset(self):
        video_path = os.path.join(self._data_dir, 
                                  _CONSTANTS['video_folder'])
        video_list = os.path.join(self._data_dir, 
                                  _CONSTANTS['video_list'])

        with open(video_list, 'r') as f:
            lines = f.readlines()
        self._idx_db = [os.path.join(video_path, video.strip()) for video in lines]
        self._labels = None

        if self.dataset_type == "val":
            np.random.shuffle(self._idx_db)
            self._idx_db = self._idx_db[:self.config.num_eval_videos]
    
    def get_label(self, index, sample=None):
        video = self.idx_db[index]
        camera_path = os.path.join(self._data_dir, 
                                   _CONSTANTS['camera_folder'], 
                                   os.path.basename(video).replace("mp4", "json"))
        with open(camera_path, "r") as f:
            cameras = json.load(f)
        total_num_frames = len(cameras)
        
        if sample and "frame_idx" in sample:
            frame_idx = sample.pop("frame_idx")
        elif not self._bi_frame:
            offset = torch.randint(max(1, total_num_frames-self._num_frames-1), [1]).item()
            frame_idx = list(range(offset, offset+self._num_frames))
        else:
            offset = torch.randint(max(1, total_num_frames-self._num_frames-1), [1]).item()
            frames = [np.random.beta(2, 1, size=1), np.random.beta(1, 2, size=1)]
            frames = sorted([int(frames[0] * self._num_frames), int(frames[1] * self._num_frames)])
            frame_idx = [offset+min(frames), offset+max(frames)]

        cams = []
        for idx in frame_idx:
            orig_idx = min(idx, len(cameras)-1)
            cams.append(torch.FloatTensor(list(map(float, cameras[orig_idx][1]))).unsqueeze(0))
        cams = torch.cat(cams, dim=0)

        if not self.bi_frame and cams.shape[0] == 1:
            cams = cams.squeeze(0)

        return cams

    @property
    def idx_db(self):
        return self._idx_db
    
    @property
    def bi_frame(self):
        return self._bi_frame
    
    @bi_frame.setter
    def bi_frame(self, v):
        self._bi_frame = v
    
    @property
    def labels(self):
        if self._labels is None:
            self._labels = []
            for index in range(len(self)):
                camera_path = os.path.join(self._data_dir, 
                              _CONSTANTS['camera_folder'], 
                              os.path.basename(self.idx_db[index]).replace("mp4", "json"))
                with open(camera_path, "r") as f:
                    cameras = json.load(f)
                self._labels.append(cameras[torch.randint(len(cameras), [1]).item()][1])
            self._labels = np.array(self._labels)
        return self._labels

    
