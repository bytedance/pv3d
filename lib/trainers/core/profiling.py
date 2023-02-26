# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import logging
from abc import ABC
from typing import Type

from lib.utils.general import Timer


logger = logging.getLogger(__name__)


class TrainerProfilingMixin(ABC):
    profiler: Type[Timer] = Timer()

    def profile(self, text: str) -> None:
        if self.training_config.logger_level != "debug":
            return
        logging.debug(f"{text}: {self.profiler.get_time_since_start()}")
        self.profiler.reset()