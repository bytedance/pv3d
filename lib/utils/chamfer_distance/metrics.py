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


def get_dataloader():
    # logger.info("Building evaluation dataloader")
    config = registry.get("config")
    dataloader = build_dataset(config).train_dataloader()
    dataloader.dataset.skip_processor = True
    if hasattr(dataloader.dataset, "bi_frame"):
        dataloader.dataset.bi_frame = False
    return dataloader


def _compute_fvd(generator, max_real: int, num_gen: int):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

    gt_dataloader = get_dataloader()
    mu_real, sigma_real = compute_feature_stats_for_dataset(
        dataloader=gt_dataloader, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = compute_feature_stats_for_generator(
        generator=generator, camera_dataset=gt_dataloader.dataset, batch_size=2, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)


def compute_fvd_512(generator):
    return _compute_fvd(generator, max_real=512, num_gen=512)


def compute_fvd_5k(generator):
    return _compute_fvd(generator, max_real=5000, num_gen=5000)


def compute_fvd_10k(generator):
    return _compute_fvd(generator, max_real=10000, num_gen=10000)


def compute_fvd_full(generator):
    return _compute_fvd(generator, max_real=None, num_gen=10000)