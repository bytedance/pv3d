# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import collections
from torchvision import transforms
from omegaconf import OmegaConf
from lib.common.registry import registry
from lib.datasets.processors.processors import BaseProcessor


@registry.register_processor("transforms")
class TorchvisionTransforms(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert OmegaConf.is_dict(transform_params) or OmegaConf.is_list(
            transform_params
        )
        if OmegaConf.is_dict(transform_params):
            transform_params = [transform_params]

        transforms_list = []

        for param in transform_params:
            if OmegaConf.is_dict(param):
                # This will throw config error if missing
                transform_type = param.type
                transform_param = param.get("params", OmegaConf.create({}))
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    + "type and params"
                )
                transform_type = param
                transform_param = OmegaConf.create([])

            transform = getattr(transforms, transform_type, None)
    
            if transform is None:
                transform = registry.get_processor_class(transform_type)
            assert transform is not None, (
                f"transform {transform_type} is not present in torchvision, "
                + "torchaudio or processor registry"
            )

            # https://github.com/omry/omegaconf/issues/248
            transform_param = OmegaConf.to_container(transform_param)
            if "interpolation" in transform_param:
                transform_param["interpolation"] = eval(transform_param["interpolation"])
            # If a dict, it will be passed as **kwargs, else a list is *args
            if isinstance(transform_param, collections.abc.Mapping):
                transform_object = transform(**transform_param)
            else:
                transform_object = transform(*transform_param)

            transforms_list.append(transform_object)

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, sample):
        # Support both dict and normal mode
        if isinstance(sample, collections.abc.Mapping):
            if "frames" in sample:
                sample["frames"] = self.transform(sample["frames"])
            else:
                raise FileNotFoundError("Key: images not found")
            return sample
        else:
            return self.transform(sample)