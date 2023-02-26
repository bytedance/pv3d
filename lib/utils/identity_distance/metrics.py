# Copyright 2022 ByteDance and/or its affiliates.
#
# Copyright (2022) PV3D Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import scipy.linalg
import logging
from lib.common.registry import registry
from lib.common.build import build_dataset
from .metrics_utils import *


logger = logging.getLogger(__name__)


def _compute_id(generator, num_gen: int, pitch: int, yaw: int):

    id_sim = compute_feature_stats_for_generator(
             generator=generator, pitch=pitch, yaw=yaw, batch_size=2, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    return float(id_sim)


def compute_id(generator, pitch=0, yaw=30):
    return _compute_id(generator, num_gen=1000, pitch=pitch, yaw=yaw)