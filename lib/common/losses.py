# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
from torch import Tensor
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Any, Dict, Union, List
from lib.common.registry import registry
from lib.utils.distributed import is_dist_initialized
from lib.utils.general import get_batch_size, get_world_size
from lib.utils.logger import log_class_usage


@dataclass
class LossConfigType:
    type: str = MISSING
    params: Dict[str, Any] = MISSING


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_config`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (ListConfig): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instantiations of each loss
                                   passed in config
    """

    # TODO: Union types are not supported in OmegaConf.
    # Later investigate for a workaround.for
    def __init__(self, loss_list: List[Union[str, LossConfigType]]):
        super().__init__()
        self.losses = nn.ModuleList()
        config = registry.get("config")
        self._evaluation_predict = False
        if config:
            self._evaluation_predict = config.get("evaluation", {}).get(
                "predict", False
            )

        for loss in loss_list:
            self.losses.append(Loss(loss))

    def forward(self, sample: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        """Takes in the original ``Sample`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample (Dict): Sample given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}

        for loss in self.losses:
            eval_applicable = getattr(loss, 'eval_applicable', True)
            if not eval_applicable and not self.training:
                continue
            output.update(loss(sample, model_output))

        should_register = registry.get("losses") is None

        if should_register and not torch.jit.is_scripting():
            if "train_discriminator" in sample:
                registry_loss_key = "{}.{}.{}.{}".format(
                    "losses", sample["dataset_name"], sample["dataset_type"],
                    "discriminator" if sample["train_discriminator"] else "generator"
                )
            elif "phase" in sample:
                registry_loss_key = "{}.{}.{}.{}".format(
                    "losses", sample["dataset_name"], 
                    sample["dataset_type"], sample["phase"]
                )
            else:
                registry_loss_key = "{}.{}.{}".format(
                    "losses", sample["dataset_name"], sample["dataset_type"]
                )
            # Register the losses to registry
            registry.register(registry_loss_key, output)

        return output


class Loss(nn.Module):
    """Internal helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/ffhq/logit_bce": 27.4}``, in case
    `logit_bce` is used and sample is from `val` set of dataset `ffhq`.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``Loss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}

        is_mapping = isinstance(params, collections.abc.MutableMapping)

        if is_mapping:
            if "type" not in params:
                raise ValueError(
                    "Parameters to loss must have 'type' field to"
                    "specify type of loss to instantiate"
                )
            else:
                loss_name = params["type"]
        else:
            assert isinstance(
                params, str
            ), "loss must be a string or dictionary with 'type' key"
            loss_name = params

        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        log_class_usage("Loss", loss_class)

        if loss_class is None:
            raise ValueError(f"No loss named {loss_name} is registered to registry")

        if is_mapping:
            loss_params = params.get("params", {})
        else:
            loss_params = {}
        self.loss_criterion = loss_class(**loss_params)
        
        self.eval_applicable = loss_params.get('eval_applicable', True)

    def forward(self, sample: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        loss = self.loss_criterion(sample, model_output)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        if not torch.jit.is_scripting():
            key = "{}/{}/{}/{}".format(
                sample["dataset_type"], sample["dataset_name"], 
                self.name, sample["phase"]
            )
        else:
            key = f"{self.name}"
        return {key: loss}


@registry.register_loss("adv_video")
class StyleGANLoss(nn.Module):

    def __init__(self, **params):
        super().__init__()
        self.loss_disc_weight = params.get("loss_disc_weight", 1.0)
        self.loss_gen_weight = params.get("loss_gen_weight", 1.0)
        self.loss_disc_reg_weight = params.get("loss_disc_reg_weight", 16.0)
        self.loss_gen_reg_weight = params.get("loss_gen_reg_weight", 4.0)
        self.loss_density_reg_weight = params.get("loss_density_reg_weight", 0.25)
        self.loss_img_weight = params.get("loss_img_weight", 1.0)
        self.loss_vid_weight = params.get("loss_vid_weight", 1.0)
        self.r1_gamma = params.get("r1_gamma", 1.0)

    def forward(self, sample, model_output):
        assert "phase" in sample, "training phase not found"
        if sample["phase"] == "Gmain":
            loss_vid = self.loss_vid_weight*F.softplus(-model_output["generated_vid_scores"]).mean().mul(self.loss_gen_weight)
            loss_img = self.loss_img_weight*F.softplus(-model_output["generated_img_scores"]).mean().mul(self.loss_gen_weight)
            # loss_img = loss_img + 25.0 * (model_output["coarse_delta"]**2).mean() + 25.0 * (model_output["fine_delta"]**2).mean()
        elif sample["phase"] == "Greg":
            loss_img = F.l1_loss(model_output["sigma_initial"], model_output["sigma_perturbed"]).mul(self.loss_density_reg_weight).mul(self.loss_gen_reg_weight)
            loss_vid = torch.zeros_like(loss_img)
        elif sample["phase"] == "Dmain":
            loss_vid = self.loss_vid_weight*F.softplus(model_output["generated_vid_scores"]).mean().mul(self.loss_disc_weight) + \
                       self.loss_vid_weight*F.softplus(-model_output["real_vid_scores"]).mean().mul(self.loss_disc_weight)
            loss_img = self.loss_img_weight*F.softplus(model_output["generated_img_scores"]).mean().mul(self.loss_disc_weight) + \
                       self.loss_img_weight*F.softplus(-model_output["real_img_scores"]).mean().mul(self.loss_disc_weight)
        elif sample["phase"] == "Dreg":
            loss_vid = self.loss_vid_weight*model_output["grad_vid_penalty"].mul(self.r1_gamma/2).mean().mul(self.loss_disc_reg_weight)
            loss_img = self.loss_img_weight*model_output["grad_img_penalty"].mul(self.r1_gamma/2).mean().mul(self.loss_disc_reg_weight)
        else:
            raise RuntimeError("unknown training phase: {}".format(sample["phase"]))
        phase_loss_dict = {
            "{}/{}/adv_video/{}/img".format(sample["dataset_type"], sample["dataset_name"], sample["phase"]): loss_img.clone().detach(),
            "{}/{}/adv_video/{}/vid".format(sample["dataset_type"], sample["dataset_name"], sample["phase"]): loss_vid.clone().detach()
        }
        loss_dict = registry.get("losses", phase_loss_dict, no_warning=True)
        loss_dict.update(phase_loss_dict)
        registry.register("losses", loss_dict)
        loss = loss_vid + loss_img
        return loss


@registry.register_loss("adv_img")
class ImgLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.loss_disc_weight = params.get("loss_disc_weight", 1.0)
        self.loss_gen_weight = params.get("loss_gen_weight", 1.0)
        self.loss_disc_reg_weight = params.get("loss_disc_reg_weight", 16.0)
        self.loss_gen_reg_weight = params.get("loss_gen_reg_weight", 4.0)
        self.loss_density_reg_weight = params.get("loss_density_reg_weight", 0.25)
        self.loss_img_weight = params.get("loss_img_weight", 1.0)
        self.r1_gamma = params.get("r1_gamma", 1.0)

    def forward(self, sample, model_output):
        assert "phase" in sample, "training phase not found"
        if sample["phase"] == "Gmain":
            loss_img = self.loss_img_weight*F.softplus(-model_output["generated_img_scores"]).mean().mul(self.loss_gen_weight)
        elif sample["phase"] == "Greg":
            loss_img = F.l1_loss(model_output["sigma_initial"], model_output["sigma_perturbed"]).mul(self.loss_density_reg_weight).mul(self.loss_gen_reg_weight)
        elif sample["phase"] == "Dmain":
            loss_img = self.loss_img_weight*F.softplus(model_output["generated_img_scores"]).mean().mul(self.loss_disc_weight) + \
                       self.loss_img_weight*F.softplus(-model_output["real_img_scores"]).mean().mul(self.loss_disc_weight)
        elif sample["phase"] == "Dreg":
            loss_img = self.loss_img_weight*model_output["grad_img_penalty"].mul(self.r1_gamma/2).mean().mul(self.loss_disc_reg_weight)
        else:
            raise RuntimeError("unknown training phase: {}".format(sample["phase"]))
        phase_loss_dict = {
            "{}/{}/adv_img/{}/img".format(sample["dataset_type"], sample["dataset_name"], sample["phase"]): loss_img.clone().detach(),
        }
        loss_dict = registry.get("losses", phase_loss_dict, no_warning=True)
        loss_dict.update(phase_loss_dict)
        registry.register("losses", loss_dict)
        return loss_img