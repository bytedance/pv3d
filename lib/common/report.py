# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import collections
import copy, os
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import imageio
import numpy as np
from torchvision.utils import save_image
from lib.utils.general import infer_batch_size, write_gif_videos, save_image_grid
from lib.utils.distributed import get_world_size, get_rank, is_master, synchronize
from lib.utils.render_utils import *


class Report(OrderedDict):
    def __init__(
        self, batch: Dict = None, model_output: Dict[str, Any] = None, *args
    ):
        super().__init__(self)
        if batch is None:
            return
        if model_output is None:
            model_output = {}
        if self._check_and_load_tuple(batch):
            return

        all_args = [batch, model_output] + [*args]
        for idx, arg in enumerate(all_args):
            if not isinstance(arg, collections.abc.Mapping):
                raise TypeError(
                    "Argument {:d}, {} must be of instance of "
                    "collections.abc.Mapping".format(idx, arg)
                )

        self.batch_size = infer_batch_size(batch)
        self.warning_string = (
            "Updating forward report with key {}"
            "{}, but it already exists in {}. "
            "Please consider using a different key, "
            "as this can cause issues during loss and "
            "metric calculations."
        )

        for idx, arg in enumerate(all_args):
            for key, item in arg.items():
                if key in self and idx >= 2:
                    log = self.warning_string.format(
                        key, "", "in previous arguments to report"
                    )
                    warnings.warn(log)
                self[key] = item
        
        self.world_size = get_world_size()
        self.rank = get_rank()

    def get_batch_size(self) -> int:
        return self.batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def _check_and_load_tuple(self, batch):
        if isinstance(batch, collections.abc.Mapping):
            return False

        if isinstance(batch[0], (tuple, list)) and isinstance(batch[0][0], str):
            for kv_pair in batch:
                self[kv_pair[0]] = kv_pair[1]
            return True
        else:
            return False

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self) -> List[str]:
        return list(self.keys())

    def apply_fn(self, fn: Callable, fields: Optional[List[str]] = None):
        """Applies a function `fn` on all items in a report. Can apply to specific
        fields if `fields` parameter is passed

        Args:
            fn (Callable): A callable to called on each item in report
            fields (List[str], optional): Use to apply on specific fields.
                Defaults to None.

        Returns:
            Report: Update report after apply fn
        """
        for key in self.keys():
            if fields is not None and isinstance(fields, (list, tuple)):
                if key not in fields:
                    continue
            self[key] = fn(self[key])
            if isinstance(self[key], collections.MutableSequence):
                for idx, item in enumerate(self[key]):
                    self[key][idx] = fn(item)
            elif isinstance(self[key], dict):
                for subkey in self[key].keys():
                    self[key][subkey] = fn(self[key][subkey])
        return self

    def detach(self) -> "Report":
        """Similar to tensor.detach, detach all items in a report from their graphs.
        This is useful in clearing up memory sometimes.

        Returns:
            Report: Detached report is returned back.
        """
        return self.apply_fn(self._detach_tensor)

    def to(
        self,
        device: Union[torch.device, str],
        non_blocking: bool = True,
        fields: Optional[List[str]] = None,
    ):
        """Move report to a specific device defined 'device' parameter.
        This is similar to how one moves a tensor or dcit to a device

        Args:
            device (torch.device): Device can be str defining device or torch.device
            non_blocking (bool, optional): Whether transfer should be non_blocking.
                Defaults to True.
            fields (List[str], optional): Use this is you only want to move some
                specific fields to the device instead of full report. Defaults to None.

        Raises:
            TypeError: If device type is not correct

        Returns:
            Report: Updated report is returned back
        """
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or "
                    "'torch.device' type, {} found".format(type(device))
                )
            device = torch.device(device)

        def fn(x):
            if hasattr(x, "to"):
                x = x.to(device, non_blocking=non_blocking)
            return x

        return self.apply_fn(fn, fields)

    def accumulate_tensor_fields_and_loss(
        self, report: "Report", field_list: List[str]
    ):
        self._accumulate_tensor_fields(report, field_list)
        self._accumulate_loss(report)
    
    def update_tensor_fields(
        self, report: "Report", field_list: List[str]
    ):
        for key in field_list: 
            self[key] = report[key]
    
    def _accumulate_tensor_fields(
        self, report: "Report", field_list: List[str]
    ):
        for key in field_list:
            if key == "__prediction_report__":
                continue
            if key not in self.keys():
                warnings.warn(
                    f"{key} not found in report. Metrics calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.cat((self[key], report[key]), dim=0)
            
            if isinstance(self[key], list):
                self[key].extend(report[key])
    
    def _accumulate_loss(self, report: "Report", divisor=1):
        for key, value in report.losses.items():
            if key not in self.losses.keys():
                # warnings.warn(
                #     f"{key} not found in report. Loss calculation "
                #     + "might not work as expected."
                # )
                self.losses[key] = value / divisor
            elif isinstance(self.losses[key], torch.Tensor):
                self.losses[key] += value / divisor

    def copy(self) -> "Report":
        """Get a copy of the current Report

        Returns:
            Report: Copy of current Report.

        """
        report = Report()

        fields = self.fields()

        for field in fields:
            report[field] = copy.deepcopy(self[field])

        return report
    
    def _detach_tensor(self, tensor: Any) -> Any:
        """Detaches any element passed which has a `.detach` function defined.
        Currently, can be Report or a tensor.

        Args:
            tensor (Any): Item to be detached

        Returns:
            Any: Detached element
        """
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        return tensor
    
    def save_generated_images(self):
        if is_master():
            save_image(self["image"], os.path.join(self.fake_dir, f'fakes{self["num_updates"]:06d}.png'), 
                    normalize=True, value_range=(-1, 1), nrow=self["nrow"], padding=0)
            save_image(self["image_raw"], os.path.join(self.fake_dir, f'fakes{self["num_updates"]:06d}_raw.png'), 
                    normalize=True, value_range=(-1, 1), nrow=self["nrow"], padding=0)
            depth = -1*self["image_depth"].cpu()
            save_image(depth, os.path.join(self.fake_dir, f'fakes{self["num_updates"]:06d}_depth.png'), 
                    normalize=True, value_range=(depth.min(), depth.max()), nrow=self["nrow"], padding=0)
        synchronize()
    
    def save_generated_videos(self, loaded_batches, save_gif=False, fps=25):
        key = "generated_imgs"
        # with open('test.obj', 'w') as f: depth_mesh = xyz2mesh(self['xyz'].permute(0,3,1,2)[0].unsqueeze(0)); depth_mesh.export(f, file_type='obj')
        if key in self:
            batch_size = self[key].shape[0]
            for idx, video in enumerate(self[key]):
                video_id = ((loaded_batches - 1) * self.world_size + self.rank) * batch_size + idx
                if video.shape[0] > 1:
                    video = video.clamp(-1, 1).cpu()
                    video = ((video + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
                    imageio.mimwrite(os.path.join(self.fake_dir, f'{video_id:0>5}.mp4'), video, fps=fps)
                    if save_gif:
                        gif_path = f"{self.fake_dir}_gif"
                        if not os.path.exists(gif_path):
                            os.makedirs(gif_path, exist_ok=True)
                        write_gif_videos(os.path.join(gif_path, f'{video_id:0>5}.gif'), video, fps=fps)
                else:
                    save_image(video[0], os.path.join(self.fake_dir, f'{video_id:0>5}.jpg'), 
                               normalize=True, value_range=(-1, 1))