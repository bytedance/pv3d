# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


from collections import OrderedDict
from lib.common.registry import registry


class BaseDataset():
    '''
    All of the datasets need to inherit BaseDataset
    '''
    def __init__(self, config, dataset_name, dataset_type, *args, **kwds) -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.skip_processor = False

        self._processors_map = []
        processor_config = config.get('processors', None)
        if processor_config is not None:
            self.init_processor_map(processor_config)
    
    def build(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """
        Basically, __getitem__ of a torch dataset.

        Args:
            idx (int): Index of the sample to be loaded.
        """

        raise NotImplementedError
    
    def __len__(self):
        """
        Basically, __len__ of a torch dataset.
        """
        
        raise NotImplementedError

    def init_processors(self):
        if "processors" not in self.config:
            return

        from lib.common.build import build_processors

        extra_params = {"data_dir": self.config.data_dir}
        reg_key = f"{self.dataset_name}_{{}}"
        processor_dict = build_processors(
            self.config.processors, reg_key, **extra_params
        )
        for processor_key, processor_instance in processor_dict.items():
            setattr(self, processor_key, processor_instance)
            full_key = reg_key.format(processor_key)
            registry.register(full_key, processor_instance)
    

    def init_processor_map(self, config):
        '''
        init processor map from config
        '''
        for p in OrderedDict(config).keys():
            if self.dataset_type != 'train' and 'augment' in p:
                continue
            self._processors_map.append(p)
    
    @property
    def processor_map(self):
        return self._processors_map