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


import glob, cv2, torch
import numpy as np
import imageio, warnings, argparse
from lib.utils.dataset import VideoReader


def make_image_grid(path):
    img_list = glob.glob(f"{path}/00*.jpg")
    assert len(img_list) > 0
    H, W = cv2.imread(img_list[0]).shape[1:]
    gw = np.clip(7680 // W, 7, 32)
    gh = np.clip(4320 // H, 4, 32)
    imgs = np.concatenate([cv2.imread(img)[None, :] for img in img_list[:gw*gh]], axis=0)

    _N, H, W, C = imgs.shape
    imgs = imgs.reshape([gh, gw, H, W, C])
    imgs = imgs.transpose(0, 2, 1, 3, 4)
    imgs = imgs.reshape([gh * H, gw * W, C])

    cv2.imwrite(f"{path}/image_grid.jpg", imgs)


def make_video_grid(args, num_frames=1e20):
    video_list = glob.glob(f"{args.path}/{args.regex}.mp4")
    assert len(video_list) > 0

    video_grid = []
    for video in video_list:
        V = VideoReader(video, num_frames)
        video_frames = V._read_video(offset=0)
        if args.resize is not None:
            video_frames = np.array([cv2.resize(np.uint8(f.to_rgb().to_ndarray()), (args.resize, args.resize)) for f in video_frames])
        else:
            video_frames = np.array([np.uint8(f.to_rgb().to_ndarray()) for f in video_frames])
        # assert video_frames.shape[0] == num_frames
        video_grid.append(video_frames[None, :])
    video_grid = np.concatenate(video_grid, axis=0).swapaxes(0, 1)

    H, W = video_grid.shape[-3:-1]
    gh, gw = list(map(int, args.grid.split(",")))
    if gw*gh != len(video_list):
        warnings.warn(f"{len(video_list)} videos cannot be devided by {gw}x{gh}, concatenate videos into one line")
        gw = len(video_list)
        gh = 1
    T, N, H, W, C = video_grid.shape
    video_grid = np.tile(video_grid, [args.repeat, 1, 1, 1, 1])
    video_grid = video_grid.reshape([T * args.repeat, gh, gw, H, W, C])
    video_grid = video_grid.transpose(0, 1, 3, 2, 4, 5)
    video_grid = video_grid.reshape([T * args.repeat, gh * H, gw * W, C])
    imageio.mimwrite(f"{args.path}/video_grid.mp4", torch.from_numpy(video_grid), fps=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("make video/image grid")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--regex", type=str, required=True)
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    make_video_grid(args)