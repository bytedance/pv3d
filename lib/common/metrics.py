# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import subprocess
import collections
from lib.common.registry import registry
from lib.utils.env import get_root
from lib.utils.general import exec_process
from lib.utils.logger import log_class_usage
from lib.utils.distributed import is_master, synchronize, get_rank
from lib.utils.configuration import get_global_config
from lib.utils.evaluation import setup_evaluation, setup_fvd_evaluation
from lib.utils.frechet_video_distance import compute_fvd_5k
from lib.utils.identity_distance import compute_id


class Metrics:
    """Internally used, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (ListConfig): List of DictConfigs where each DictConfig
                                        specifies name and parameters of the
                                        metrics used.
    """

    def __init__(self, metric_list):
        if not isinstance(metric_list, collections.abc.Sequence):
            metric_list = [metric_list]

        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        self.required_params = {"dataset_name", "dataset_type"}
        for metric in metric_list:
            params = {}
            dataset_names = []
            if isinstance(metric, collections.abc.Mapping):
                if "type" not in metric:
                    raise ValueError(
                        f"Metric {metric} needs to have 'type' attribute "
                        + "or should be a string"
                    )
                metric_type = key = metric.type
                params = metric.get("params", {})
                # Support cases where uses need to give custom metric name
                if "key" in metric:
                    key = metric.key

                # One key should only be used once
                if key in metrics:
                    raise RuntimeError(
                        f"Metric with type/key '{metric_type}' has been defined more "
                        + "than once in metric list."
                    )

                # a custom list of dataset where this metric will be applied
                if "datasets" in metric:
                    dataset_names = metric.datasets
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )
                metric_type = key = metric

            metric_cls = registry.get_metric_class(metric_type)
            if metric_cls is None:
                raise ValueError(
                    f"No metric named {metric_type} registered to registry"
                )

            metric_instance = metric_cls(**params)
            metric_instance.name = key
            metric_instance.set_applicable_datasets(dataset_names)

            metrics[key] = metric_instance
            self.required_params.update(metrics[key].required_params)
            self.requires_generator = metrics[key].requires_generator

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}

        dataset_type = sample_list.dataset_type
        dataset_name = sample_list.dataset_name

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                if not metric_object.is_dataset_applicable(dataset_name):
                    continue
                key = f"{dataset_type}/{dataset_name}/{metric_name}"
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

                if not isinstance(values[key], torch.Tensor):
                    values[key] = torch.tensor(values[key], dtype=torch.float)
                else:
                    values[key] = values[key].float()

                if values[key].dim() == 0:
                    values[key] = values[key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values


class BaseMetric:
    """Base class to be inherited by all metrics registered. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.required_params = []
        self.requires_generator = False
        # the set of datasets where this metric will be applied
        # an empty set means it will be applied on *all* datasets
        self._dataset_names = set()
        log_class_usage("Metric", self.__class__)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``sample`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample (Dict): Sample provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value

    def set_applicable_datasets(self, dataset_names):
        self._dataset_names = set(dataset_names)

    def is_dataset_applicable(self, dataset_name):
        return len(self._dataset_names) == 0 or dataset_name in self._dataset_names


@registry.register_metric("fid50k")
class FrechetInceptionDistance(BaseMetric):
    """Metric for frechet inception distance
    """
    def __init__(self, **params):
        super().__init__('fid50k')
        self.required_params = ["image", "image_raw", "image_depth"]
        self.requires_generator = True
        setup_evaluation(**params)

    def calculate(self, sample, model_output):
        synchronize()
        fid = compute_fid_50k_full(model_output["generator"])
        torch.cuda.empty_cache()
        synchronize()
        device = sample["image"].device
        return torch.tensor(fid, device=device)


@registry.register_metric("fvd")
class FrechetInceptionDistance(BaseMetric):
    """Metric for frechet video inception distance
    """
    def __init__(self, **params):
        super().__init__('fvd')
        self.required_params = ["image", "image_raw", "image_depth"]
        self.requires_generator = True
        setup_fvd_evaluation(**params)
    
    def calculate(self, sample, model_output):
        torch.cuda.empty_cache()
        synchronize()
        # id_sim = compute_id(model_output["generator"])
        fvd = compute_fvd_5k(model_output["generator"])
        torch.cuda.empty_cache()
        synchronize()
        device = sample["image"].device
        return torch.tensor(fvd, device=device)


# @registry.register_metric("fid")
# class FrechetInceptionDistance(BaseMetric):
#     """Metric for frechet inception distance
#     """
#     def __init__(self, **params):
#         super().__init__('fid')
#         self.required_params = ["generated_imgs"]
#         setup_evaluation(**params)
    
#     def calculate(self, sample, model_output):
#         from torch_fidelity import calculate_metrics
#         metrics_dict = calculate_metrics(input1=model_output["real_dir"],
#                                          input2=model_output["fake_dir"],
#                                          cuda=True,
#                                          isc=False,
#                                          fid=True,
#                                          kid=False,
#                                          verbose=False)
#         fid = metrics_dict['frechet_inception_distance']
#         torch.cuda.empty_cache()
#         synchronize()
#         device = sample["generated_videos"].device if "generated_videos" in sample else sample["generated_imgs"].device
#         return torch.tensor(fid, device=device)
