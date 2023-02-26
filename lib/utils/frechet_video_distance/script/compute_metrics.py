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


import numpy as np
import scipy.linalg
import logging, glob
from torch.utils.data import DataLoader
from lib.utils.frechet_video_distance.script.metrics_utils import compute_feature_stats_for_dataset, compute_feature_stats_for_generator
from lib.utils.frechet_video_distance.script.video_dataset import VideoDataset


logger = logging.getLogger(__name__)


def build_data_loader(path, batch_gen=4):
    dataset = VideoDataset(path)
    args = { "shuffle": True }
    camera_loader = DataLoader(
                    dataset, 
                    batch_size=batch_gen,
                    **args
                )
    return camera_loader


def _compute_fvd(path, dataset_name, img_size, max_real: int, num_gen: int):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

    dataloader = build_data_loader(path)
    mu_real, sigma_real = compute_feature_stats_for_dataset(
        dataset_name=dataset_name, img_size=img_size, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = compute_feature_stats_for_generator(
        dataloader=dataloader, batch_size=2, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fvd = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fvd)


def compute_fvd_512(generator):
    return _compute_fvd(generator, max_real=512, num_gen=512)


def compute_fvd_5k(path):
    return _compute_fvd(path, dataset_name="voxceleb", img_size=512, max_real=5000, num_gen=5000)


def compute_fvd_10k(generator):
    return _compute_fvd(generator, max_real=10000, num_gen=10000)


def compute_fvd_full(generator):
    return _compute_fvd(generator, max_real=None, num_gen=10000)