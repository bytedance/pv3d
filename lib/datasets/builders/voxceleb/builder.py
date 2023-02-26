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

from lib.common.registry import registry
from lib.datasets.base_dataset_builder import BaseDatasetBuilder
from .dataset import VoxCelebDataset

@registry.register_dataset_builder("voxceleb")
class VoxCelebBuilder(BaseDatasetBuilder):
    def __init__(self,
                 dataset_name='voxceleb', 
                 dataset_class=VoxCelebDataset):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = dataset_class
    
    def load(self, config, dataset_type, *args, **kwargs):
        if dataset_type in ['train']:
            self.dataset = self.dataset_class(config, dataset_type)
            return self.dataset
        else:
            return None

    @classmethod
    def config_path(cls):
        return "configs/datasets/voxceleb.yaml"
