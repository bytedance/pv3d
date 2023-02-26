# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************


import torch
import random
import logging
import argparse
from lib.common.registry import registry
from lib.common.build import build_config, build_trainer
from lib.utils.configuration import Configuration
from lib.utils.options import options
from lib.utils.distributed import get_rank
from lib.utils.env import set_seed, setup_imports
from lib.utils.general import log_device_names
from lib.utils.logger import setup_logger, setup_very_basic_config


setup_very_basic_config()


def main(configuration, init_distributed=False, predict=False):
    # A reload might be needed for imports
    setup_imports()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    config = build_config(configuration)

    setup_logger(
        color=config.training.colored_logs, disable=config.training.should_not_log
    )
    logger = logging.getLogger("run")
    # Log args for debugging purposes
    logger.info(configuration.args)
    logger.info(f"Torch version: {torch.__version__}")
    log_device_names()
    logger.info(f"Using seed {config.training.seed}")

    trainer = build_trainer(config)
    trainer.load()
    trainer.rendering()


def run(opts=None, predict=True):

    setup_imports()

    if opts is None:
        parser = options.get_parser()
        args = parser.parse_args()
    else:
        args = argparse.Namespace(config_override=None)
        args.opts = opts

    configuration = Configuration(args)
    configuration.args = args
    config = configuration.get_config()
    config.device_id = 0
    config.run_type = "rendering"
    seed = config.training.seed
    config.training.seed = set_seed(seed if seed == -1 else seed + get_rank())
    registry.register("seed", config.training.seed)

    main(configuration, predict=predict)


if __name__ == '__main__':
    run()