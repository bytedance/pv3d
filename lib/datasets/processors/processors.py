# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import logging
from typing import Dict, Any
from dataclasses import dataclass
from lib.common.registry import registry
from lib.utils.logger import log_class_usage


logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfigType:
    type: str
    params: Dict[str, Any]


class BaseProcessor:
    """Every processor needs to inherit this class for compatibility. 
    End user mainly needs to implement ``__call__`` function.

    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.

    """

    def __init__(self, config: ProcessorConfigType = None, *args, **kwargs):

        log_class_usage("Processor", self.__class__)

    def __call__(self, item, *args, **kwargs):
        """Main function of the processor. Takes in a dict and returns back
        a dict

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.

        """
        return item


class Processor:
    """Wrapper class used to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (DictConfig): DictConfig containing ``type`` of the processor to
                             be initialized and ``params`` of that processor.

    """

    def __init__(self, config: ProcessorConfigType, *args, **kwargs):
        if "type" not in config:
            raise AttributeError(
                "Config must have 'type' attribute to specify type of processor"
            )

        processor_class = registry.get_processor_class(config.type)

        params = {}
        if "params" not in config:
            logger.warning(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                f"of type {config.type}. Setting to default {{}}"
            )
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if "_dir_representation" in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif "processor" in self.__dict__ and hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(f"The processor {name} doesn't exist in the registry.")