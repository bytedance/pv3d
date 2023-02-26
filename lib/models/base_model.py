# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import logging
import warnings
import collections
import torch.nn as nn
from copy import deepcopy
from omegaconf import MISSING
from dataclasses import dataclass
from typing import List, Optional, Dict
from lib.common.registry import registry
from lib.common.losses import Losses, LossConfigType
from lib.utils.logger import log_class_usage
from lib.utils.checkpoint import load_pretrained_model


logger = logging.getLogger(__name__)


@dataclass
class ModelConfigType:
    # Name of the model that is used in registry
    model: str = MISSING
    losses: Optional[List[LossConfigType]] = MISSING


class BaseModel(nn.Module):
    """For integration with trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a dict as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.
    """
    def __init__(self, config: ModelConfigType, *args, **kwds) -> None:
        super().__init__()
        self.config = config

        log_class_usage("Model", self.__class__)
    
    def build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError(
            "Build method not implemented in the child model class."
        )

    def __call__(self, sample, *args, **kwargs):

        model_output = super().__call__(sample, *args, **kwargs)

        # Make sure that the output from the model is a Mapping
        assert isinstance(
            model_output, collections.abc.Mapping
        ), "A dict must be returned from the forward of the model."

        if "losses" in model_output:
            if not self._logged_warning["losses_present"]:
                warnings.warn(
                    "'losses' already present in model output. "
                    "No calculation will be done in base model."
                )
                self._logged_warning["losses_present"] = True

            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."
        elif hasattr(self, "losses"):
            model_output["losses"] = self.losses(sample, model_output)
        else:
            model_output["losses"] = {}

        return model_output
    
    def forward(self, sample: Dict) -> Dict:
        """To be implemented by child class. Takes in a dict and
        returns back a dict.

        Args:
            x (dict): Dict returned by the DataLoader for
            current iteration

        Returns:
            Dict: Dict containing scores object.

        """
        raise NotImplementedError(
            "Forward of the child model class needs to be implemented."
        )

    def get_optimizer_parameters(self, config):
        
        optimizer_params = config.optimizer.params
        parameters = {}

        for group in optimizer_params:
            params = optimizer_params.get(group)
            if not isinstance(params, collections.abc.Mapping):
                continue
            else:
                parameter_group = {"params": getattr(self, group).parameters()}
                parameter_group.update(params)
                parameters[group] = [parameter_group]

        return parameters

    def init_losses(self):
        """Initializes loss for the model based ``losses`` key. Automatically called
         internally after building the model.
        """
        losses = self.config.get("losses", [])
        if len(losses) == 0:
            warnings.warn(
                "No losses are defined in model configuration. You are expected "
                "to return loss in your return dict from forward."
            )

        self.losses = Losses(losses)
    
    @classmethod
    def config_path(cls):
        return None
    
    def download_checkpoint(self):
        pass
    
    def _load_base_checkpoint(self):
        ckpt_path = self.config.base_ckpt_path
        if ckpt_path != "":
            logger.info(f"initializing model from {ckpt_path}")
            base_checkpoint = load_pretrained_model(ckpt_path, init = True)["model"]
            self.load_state_dict(base_checkpoint["model"], strict=True)
    
    @classmethod
    def format_state_key(cls, key):
        """Can be implemented if something special needs to be done to the
        key when pretrained model is being loaded. This will adapt and return
        keys according to that. Useful for backwards compatibility. See
        updated load_state_dict below. For an example, see VisualBERT model's
        code.
        Args:
            key (string): key to be formatted
        Returns:
            string: formatted key
        """
        return key

    def load_state_dict(self, state_dict, *args, **kwargs):
        copied_state_dict = deepcopy(state_dict)
        for key in list(copied_state_dict.keys()):
            formatted_key = self.format_state_key(key)
            copied_state_dict[formatted_key] = copied_state_dict.pop(key)

        return super().load_state_dict(copied_state_dict, *args, **kwargs)