# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import os, warnings
from omegaconf import OmegaConf, DictConfig
from lib.common.registry import registry
from lib.datasets.samplers.infinite_sampler import InfiniteSampler
from lib.utils.general import BatchCollator, get_optimizer_parameters, get_batch_size
from lib.utils.distributed import get_world_size, is_dist_initialized, is_master, synchronize


def build_config(configuration):
    """Builder function for config. Freezes the configuration and registers
    configuration object and config DictConfig object to registry.

    Args:
        configuration (Configuration): Configuration object that will be
            used to create the config.

    Returns:
        (DictConfig): A config which is of type omegaconf.DictConfig
    """
    configuration.freeze()
    config = configuration.get_config()
    registry.register("config", config)
    registry.register("configuration", configuration)

    return config


def build_trainer(config):
    """Builder function for creating a trainer class. Trainer class name
    is picked from the config.

    Args:
        config (DictConfig): Configuration that will be used to create
            the trainer.

    Returns:
        (BaseTrainer): A trainer instance
    """
    trainer_type = config.training.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    trainer_obj = trainer_cls(config)

    return trainer_obj


def build_scheduler(optimizer, config):
    scheduler_config = config.get("scheduler", {})

    if "type" not in scheduler_config:
        warnings.warn(
            "No type for scheduler specified even though lr_scheduler is True, "
            "setting default to 'Pythia'"
        )
    scheduler_type = scheduler_config.get("type", "pythia")

    if "params" not in scheduler_config:
        warnings.warn("scheduler attributes has no params defined, defaulting to {}.")
    params = scheduler_config.get("params", {})
    scheduler_class = registry.get_scheduler_class(scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    return scheduler


def build_dataset(config):
    """Builder function for creating a dataset loader. Dataset builder class name
    is picked from the config.

    Args:
        config (DictConfig): Configuration that will be used to create
            the dataloader.

    Returns:
        (Dataloader): A dataloader instance
    """
    
    if type(config.datasets) == str:
        datasets = list(map(lambda x: x.strip(), config.datasets.split(",")))
    assert len(datasets) == 1, 'Multiple datasets not supported'
    dataset_name = datasets[0]
    dataset_config = config.dataset_config.get(dataset_name)
    dataset_builder_cls = registry.get_dataset_builder_class(dataset_name)
    dataset_instance = dataset_builder_cls(dataset_name)
    dataset_instance.build(dataset_config)
    
    return dataset_instance


def build_dataloader_and_sampler(dataset_instance, dataset_config):
    """Builds and returns a dataloader
    Args:
        dataset_instance (torch.utils.data.Dataset): Instance of dataset for which
            dataloader has to be created
        dataset_config (omegaconf.DictConfig): required
            for infering params for dataloader
    Returns:
        torch.utils.data.DataLoader
    """
    # from lib.utils.batch_collator import BatchCollator

    training_config = registry.get("config").get("training")
    # Support params coming in from dataloader params
    other_args = {
        "num_workers": dataset_config.get(
            "num_workers", training_config.get("num_workers", 4)
        ),
        "pin_memory": dataset_config.get(
            "pin_memory", training_config.get("pin_memory", False)
        ),
        "shuffle": dataset_config.get(
            "shuffle", training_config.get("shuffle", None)
        ),
        "batch_size": get_batch_size() if dataset_instance.dataset_type == "train" \
                      else 1, 
        "drop_last": dataset_config.get(
            "drop_last", training_config.get("drop_last", False)
        ),
        "prefetch_factor": dataset_config.get(
            "prefetch_factor", training_config.get("prefetch_factor", 2)
        )
    }

    # IterableDataset returns batches directly, so no need to add Sampler
    # or batch size as user is expected to control those. This is a fine
    # assumption for now to not support single item based IterableDataset
    # as it will add unnecessary complexity and config parameters
    # to the codebase
    if not isinstance(dataset_instance, torch.utils.data.IterableDataset):
        other_args = _add_extra_args_for_dataloader(dataset_instance, other_args)
    else:
        other_args.pop("shuffle")

    # Set drop_last=True when using XLA to have constant batch size.
    # In this case we also need to set drop_last=True in DistributedSampler.
    loader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        collate_fn=BatchCollator(
            dataset_instance.dataset_name, dataset_instance.dataset_type
        ),
        **other_args,
    )

    if other_args["num_workers"] >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    loader.dataset_type = dataset_instance.dataset_type

    return loader


def _add_extra_args_for_dataloader(dataset_instance, other_args = None):

    dataset_type = dataset_instance.dataset_type

    if other_args["shuffle"] is None:
        other_args["shuffle"] = False
        if dataset_type != "test":
            other_args["shuffle"] = True

    # In distributed mode, we use DistributedSampler from PyTorch
    elif is_dist_initialized():
        other_args["sampler"] = torch.utils.data.DistributedSampler(
            dataset_instance, shuffle=other_args["shuffle"]
        )
        # Shuffle is mutually exclusive with sampler, let DistributedSampler
        # take care of shuffle and pop from main args
        other_args.pop("shuffle")

    return other_args


def build_processors(processors_config, *args, **kwargs):
    """Given a processor config, builds the processors present and returns back
    a dict containing processors mapped to keys as per the config

    Args:
        processors_config (omegaconf.DictConfig): OmegaConf DictConfig describing
            the parameters and type of each processor passed here

    Returns:
        ProcessorDict: Dictionary containing key to
            processor mapping
    """
    from lib.datasets.processors.processors import Processor

    processor_dict = {}

    for processor_key, processor_params in processors_config.items():
        if not processor_params:
            continue

        processor_instance = None

        if processor_instance is None:
            processor_instance = Processor(processor_params, *args, **kwargs)
        processor_dict[processor_key] = processor_instance

    return processor_dict


def build_model(config):

    from lib.models.base_model import ModelConfigType
    # If it is not an OmegaConf object, create the object
    if not isinstance(config, DictConfig) and isinstance(config, ModelConfigType):
        config = OmegaConf.structured(config)

    model_name = config.model
    model_class = registry.get_model_class(model_name)

    if model_class is None:
        raise RuntimeError(f"No model registered for name: {model_name}")
    model = model_class(config)

    if hasattr(model, "build"):
        """Model build involves checkpoint loading.
        Let master build the model (download the checkpoints) while
        other ranks wait for the sync message
        Once the master has downloaded the checkpoint and built the
        model it sends the sync message, completing the synchronization
        now other cores can proceed to build the model
        using already downloaded checkpoint.
        """
        if is_master():
            model.download_checkpoint()
            model.build()
            synchronize()
        else:
            synchronize()
            model.build()
        model.init_losses()

    return model


def build_optimizer(model, config):
    optimizer_config = config.optimizer
    if "type" not in optimizer_config:
        raise ValueError(
            "Optimizer attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch, e.g. 'adam_w' or 'SGD')"
        )
    optimizer_type = optimizer_config.type

    if "params" not in optimizer_config:
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = optimizer_config.get("params", {})

    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                "No optimizer class of type {} present in "
                "either torch or registered to registry"
            )

    parameters, shared_params = get_optimizer_parameters(model, config)
    optimizer = dict()
    for group in parameters:
        optimizer[group] = optimizer_class(parameters[group], **shared_params)
    return optimizer