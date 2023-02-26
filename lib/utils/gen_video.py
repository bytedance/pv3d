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
from lib.utils.distributed import distributed_init, infer_init_method


setup_very_basic_config()


def main(configuration, init_distributed=False, predict=False):
    # A reload might be needed for imports
    setup_imports()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    if init_distributed:
        distributed_init(config)

    seed = config.training.seed
    config.training.seed = set_seed(seed if seed == -1 else seed + get_rank())
    registry.register("seed", config.training.seed)

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
    trainer.generating()


def distributed_main(device_id, configuration, predict=False):
    config = configuration.get_config()
    config.device_id = device_id

    if config.distributed.rank is None:
        config.distributed.rank = config.start_rank + device_id

    main(configuration, init_distributed=True, predict=predict)


def run(opts=None, predict=False):

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
    config.start_rank = 0
    if config.distributed.init_method is None:
        infer_init_method(config)

    if config.distributed.init_method is not None:
        if torch.cuda.device_count() > 1 and not config.distributed.no_spawn:
            config.start_rank = config.distributed.rank
            config.distributed.rank = None
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(configuration, predict),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(0, configuration, predict)
    elif config.distributed.world_size > 1:
        assert config.distributed.world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        config.distributed.init_method = f"tcp://localhost:{port}"
        config.distributed.rank = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(configuration, predict),
            nprocs=config.distributed.world_size,
        )
    else:
        config.device_id = 0
        main(configuration, predict=predict)


if __name__ == '__main__':
    run()