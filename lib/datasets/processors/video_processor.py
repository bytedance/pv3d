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


import warnings
import torch
import numpy as np
import torchvision.transforms as T
from omegaconf import OmegaConf
from lib.common.registry import registry
from lib.datasets.processors.functional import *
from lib.datasets.processors.processors import BaseProcessor


@registry.register_processor('face_to_tensor')
class FaceToTensor(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'

    def __call__(self, item):
        assert isinstance(item["frames"], np.ndarray), 'invalid frame data type'
        frames = [torch.tensor(f).unsqueeze(0) for f in item["frames"]]
        frames = torch.cat(frames, dim=0)
        item["frames"] = video_to_normalized_float_tensor(frames)
        return item


@registry.register_processor('face_augment')
class FaceAugment(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.blur = config.get('blur', None)
        self.hflip = config.get('hflip', None)
        self.jitter = config.get('jitter', None)
        self.rand_rotate = config.get('rand_rotate', None)

        if self.blur:
            self.gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        
        if self.jitter:
            self.color_jitter = T.ColorJitter(brightness=.5, hue=.3)
    
    def __call__(self, item):
        assert isinstance(item["frames"], torch.Tensor), 'invalid frame data type'

        if self.hflip and torch.rand([1]).item() < self.hflip.prob:
            item["frames"] = video_hflip(item["frames"])
            item["real_cam"] = cam_hflip(item["real_cam"])

        if self.rand_rotate and torch.rand([1]).item() < self.rand_rotate.prob:
            angle = (torch.rand([1]).item() * 2 - 1) * self.rand_rotate.max_angle
            angle = int(360 + angle) if angle < 0 else int(angle)
            item["frames"] = video_rotate(item["frames"], angle)
        
        if self.blur and  torch.rand([1]).item() < self.blur.prob:
            item["frames"] = self.gaussian_blur(item["frames"])
        
        if self.jitter and  torch.rand([1]).item() < self.jitter.prob:
            item["frames"] = self.color_jitter(item["frames"])

        return item


@registry.register_processor('face_resize')
class FaceResize(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.dest_size = list(config.dest_size)
    
    def __call__(self, item):
        assert isinstance(item["frames"], torch.Tensor), 'invalid frame data type'
        item["frames"] = video_resize(item["frames"], self.dest_size)
        return item


@registry.register_processor('face_normalize')
class FaceNormalize(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.mean = config.mean
        self.std = config.std

    def __call__(self, item):
        assert isinstance(item["frames"], torch.Tensor), 'invalid frame data type'
        item["frames"] = video_normalize(item["frames"], mean=self.mean, std=self.std)
        return item


@registry.register_processor('face_pad')
class FacePad(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        assert OmegaConf.is_dict(config),'invalid processor param'
        self.length = config.length
    
    def __call__(self, item):
        assert isinstance(item["frames"], np.ndarray), 'invalid frame data type'
        if not item["frames"].shape[0]:
            warnings.warn("empty image, padding with zero")
            item["frames"] = np.zeros(shape=(1, 224, 224, 3))
        if item["frames"].shape[0] < self.length:
            pad_width = ((0, self.length-item["frames"].shape[0]), (0, 0), (0, 0), (0, 0))
            item["frames"] = np.pad(item["frames"], pad_width=pad_width, mode='edge')
        return item