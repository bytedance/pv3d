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
from lib.utils.identity_distance.arcface.model_irse import Backbone


def compute_identity_similarity(args, num_videos=1000, num_frames=4):
    device=torch.device('cuda:0')
    detector = get_feature_detector(device=device)
    scores = []
    scores_single_frame = []
    scores_cross_frame = []
    pairs = [[0, 2], [3, 1], [0, 1], [3, 2], [0, 3], [2, 1]]
    for idx in tqdm(range(num_videos)):
        batch = []
        for j in range(num_frames):
            img = cv2.cvtColor(cv2.imread(f"{args.path}/{idx:04d}_{j:02d}.png"), cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(torch.float32).to(device)
            batch.append(preprocess(img.permute(2, 0, 1).unsqueeze(0)))
        batch = torch.cat(batch, dim=0)
        with torch.no_grad():
            feat = detector(batch)
        all_scores = torch.mm(feat, feat.T)
        for i, pair in enumerate(pairs):
            score = all_scores[pair[0], pair[1]].item()
            scores.append(score)
            if i < 2:
                scores_single_frame.append(score)
            else:
                scores_cross_frame.append(score)
    print("overall ID scores:", np.mean(scores))
    return np.mean(scores)


def get_feature_detector(device=torch.device('cpu')):
    facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    facenet.load_state_dict(torch.load(os.path.expanduser('~/.cache')+"/model_ir_se50.pth", map_location=lambda storage, loc: storage))
    facenet.eval().to(device)
    return facenet


def preprocess(videos, target_resolution=112):
    # n, c, h, w = videos.shape
    videos = F.interpolate(videos, size=target_resolution, mode='bilinear',
                           align_corners=False)
    videos = videos / 127.5 - 1.0
    return videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser("compute identity consistency")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    compute_identity_similarity(args)