# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


"""
Registry is central source of truth. Inspired from Redux's
concept of global store, Registry maintains mappings of various information
to unique keys. Special functions in registry can be used as decorators to
register different kind of classes.
Import the global registry object using
``from lib.common.registry import registry``
Various decorators for registry different kind of classes with unique keys
- Register a trainer: ``@registry.register_trainer``
- Register a metric: ``@registry.register_metric``
- Register a loss: ``@registry.register_loss``
- Register a model: ``@registry.register_model``
- Register a dataset builder: ``@registry.register_dataset_builder``
- Register a processor: ``@registry.register_processor``
- Register a sampler: ``@registry.register_sampler``
- Register a optimizer: ``@registry.register_optimizer``
- Register a scheduler: ``@registry.register_scheduler``
"""

class Registry:
    r"""Class for registry object which acts as central source of truth
    """
    mapping = {
        # Mappings of builder name to their respective classes
        # Use `registry.register_builder` to register a builder class
        # with a specific name
        # Further, use the name with the class is registered in the
        # command line or configuration to load that specific dataset
        "trainer_name_mapping": {},
        "scheduler_name_mapping": {},
        "model_name_mapping": {},
        "dataset_builder_name_mapping": {},
        "processor_name_mapping": {},
        "metric_name_mapping": {},
        "loss_name_mapping": {},
        "optimizer_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_trainer(cls, name):
        r"""Register a trainer to registry with key 'name'
        Args:
            name: Key with which the trainer will be registered.
        """

        def wrap(trainer_cls):
            cls.mapping["trainer_name_mapping"][name] = trainer_cls
            return trainer_cls

        return wrap

    @classmethod
    def register_metric(cls, name):
        r"""Register a metric to registry with key 'name'
        Args:
            name: Key with which the metric will be registered.
        """

        def wrap(func):
            from lib.common.metrics import BaseMetric

            assert issubclass(
                func, BaseMetric
            ), "All Metric must inherit BaseMetric class"
            cls.mapping["metric_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_loss(cls, name):
        r"""Register a loss to registry with key 'name'
        Args:
            name: Key with which the loss will be registered.
        """

        def wrap(func):
            from torch import nn

            assert issubclass(
                func, nn.Module
            ), "All loss must inherit torch.nn.Module class"
            cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a model to registry with key 'name'
        Args:
            name: Key with which the model will be registered.
        """

        def wrap(func):
            from lib.models.base_model import BaseModel

            assert issubclass(
                func, BaseModel
            ), "All models must inherit BaseModel class"
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap
    
    @classmethod
    def register_dataset_builder(cls, name):
        r"""Register a model to registry with key 'name'
        Args:
            name: Key with which the model will be registered.
        """

        def wrap(func):
            from lib.datasets.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                func, BaseDatasetBuilder
            ), "All models must inherit BaseDataset class"
            cls.mapping["dataset_builder_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'
        Args:
            name: Key with which the processor will be registered.
        """

        def wrap(func):
            from lib.datasets.processors.processors import BaseProcessor

            assert issubclass(
                func, BaseProcessor
            ), "All Processor classes must inherit BaseProcessor class"
            cls.mapping["processor_name_mapping"][name] = func
            return func

        return wrap
    
    @classmethod
    def register_sampler(cls, name):
        r"""Register a sampler to registry with key 'name'
        Args:
            name: Key with which the sampler will be registered.
        """

        def wrap(func):
            from torch.utils.data import Sampler

            assert issubclass(
                func, Sampler
            ), "All Processor classes must inherit Sampler class"
            cls.mapping["sampler_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_optimizer(cls, name):
        def wrap(func):

            cls.mapping["optimizer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_scheduler(cls, name):
        def wrap(func):

            cls.mapping["scheduler_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'
        Args:
            name: Key with which the item will be registered.
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_trainer_class(cls, name):
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)
    
    @classmethod
    def get_dataset_builder_class(cls, name):
        return cls.mapping["dataset_builder_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)
    
    @classmethod
    def get_sampler_class(cls, name):
        return cls.mapping["sampler_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def get_loss_class(cls, name):
        return cls.mapping["loss_name_mapping"].get(name, None)

    @classmethod
    def get_optimizer_class(cls, name):
        return cls.mapping["optimizer_name_mapping"].get(name, None)

    @classmethod
    def get_scheduler_class(cls, name):
        return cls.mapping["scheduler_name_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'
        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated.
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'
        Args:
            name: Key which needs to be removed.
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()