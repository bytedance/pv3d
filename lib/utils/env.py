# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import os
import glob
import importlib
from omegaconf import OmegaConf
import torch
import random
import numpy as np
from datetime import datetime


def get_root():

    from lib.common.registry import registry

    root = registry.get("root", no_warning=True)
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(root, "../.."))
        registry.register("root", root)
    return root


def set_seed(seed):
    if seed:
        if seed == -1:
            seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
            )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    return seed
    

def setup_imports():
    from lib.common.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("root", no_warning=True)
    if root_folder is None:
        root_folder = get_root()

    trainer_folder = os.path.join(root_folder, "lib/trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "lib/datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "lib/models")
    common_folder = os.path.join(root_folder, "lib/common")
    model_pattern = os.path.join(model_folder, "**", "*.py")
    common_pattern = os.path.join(common_folder, "**", "*.py")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
        + glob.glob(common_pattern, recursive=True)
    )

    relative_root = os.path.basename(root_folder)
    for f in files:
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == relative_root:
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(splits[import_prefix_index:-1] + [module_name])
            importlib.import_module(module)

    registry.register("imports_setup", True)