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
import os, glob, cv2
import logging, argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from lib.utils.frechet_video_distance.script.compute_metrics import compute_fvd_5k


if __name__ == "__main__":
    parser = argparse.ArgumentParser("compute frechet video distance")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    fvd_5k = compute_fvd_5k(args.path)
    print(fvd_5k)