# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Union
from einops import rearrange
from torchvision.transforms.functional import rotate, InterpolationMode


# Functional file similar to torch.nn.functional
def video_crop(vid: torch.tensor, i: int, j: int, h: int, w: int) -> torch.Tensor:
    return vid[..., i : (i + h), j : (j + w)]


def video_center_crop(vid: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return video_crop(vid, i, j, th, tw)


def video_hflip(vid: torch.Tensor) -> torch.Tensor:
    return vid.flip(dims=(-1,))


def cam_hflip(cams: torch.Tensor) -> torch.Tensor:
    if cams.ndim == 2:
        cams[:, [1, 2, 3, 4, 8]] *= -1.0
    else:
        cams[[1, 2, 3, 4, 8]] *= -1.0
    return cams


def video_random_flip(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x


def video_rotate(vid: torch.Tensor, angle: int) -> torch.Tensor:
    return rotate(vid, angle, interpolation=InterpolationMode.BILINEAR)


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def video_resize(
    vid: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
) -> torch.Tensor:
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False
    )


def video_pad(
    vid: torch.Tensor, padding: float, fill: float = 0, padding_mode: str = "constant"
) -> torch.Tensor:
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def video_to_normalized_float_tensor(vid: torch.Tensor) -> torch.Tensor:
    return vid.permute(0, 3, 1, 2).to(torch.float32) / 255


def video_normalize(vid: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


def warp_with_flip_batch(x):
    out = []
    for ii in range(x.shape[0]):
        out.append(warp_with_flip(x[ii]))
    return torch.cat(out, dim=0)


def warp_with_flip(x):
    num = random.randint(0, 1)
    if num == 1:
        return torch.flip(x, [-1]).unsqueeze(0)
    else:
        return x.unsqueeze(0)


def warp_with_color_batch(x):
    out = []
    for ii in range(x.shape[0]):
        out.append(warp_with_color(x[ii]))
    return torch.cat(out, dim=0)


def warp_with_color(x):
    c_shift = torch.rand(1) - 0.5
    c_shift = c_shift.cuda(x.get_device())
    m = torch.zeros_like(x)
    m = m.cuda(x.get_device())
    num = random.randint(0, 3)
    if num == 0:
        m.data += c_shift
    elif num == 1:
        m[0].data += c_shift
    elif num == 2:
        m[1].data += c_shift
    else:
        m[2].data += c_shift

    out = x + m
    return out.unsqueeze(0)


def warp_with_cutout_batch_real(x):
    out = []
    for ii in range(x.shape[0]):
        out.append(warp_with_cutout_real(x[ii]))
    return torch.cat(out, dim=0)


def warp_with_cutout_real(x, max_ratio=0.25):
    c, h, w = x.size()
    m = np.ones((c, h, w), np.float32)

    ratio = random.uniform(max_ratio / 2, max_ratio)
    num = random.randint(0, 3)
    if num == 0:
        h_start = random.uniform(0, max_ratio - ratio)
        w_start = random.uniform(0, 1 - max_ratio)
    elif num == 1:
        h_start = random.uniform(1 - max_ratio, 1 - ratio)
        w_start = random.uniform(0, 1 - max_ratio)
    elif num == 2:
        w_start = random.uniform(0, max_ratio - ratio)
        h_start = random.uniform(0, 1 - max_ratio)
    else:
        w_start = random.uniform(1 - max_ratio, 1 - ratio)
        h_start = random.uniform(0, 1 - max_ratio)

    h_s = round(h_start * (h - 1) - 0.5)
    w_s = round(w_start * (w - 1) - 0.5)
    length = round(h * ratio - 0.5)

    m[:, h_s:h_s + length, w_s:w_s + length] = 0.
    m = torch.from_numpy(m).cuda(x.get_device())
    out = x * m
    return out.unsqueeze(0)


def warp_with_affine(x, angle=180, trans=0.1, scale=0.05):
    angle = np.pi * angle / 180.

    pa = torch.FloatTensor(4)
    th = torch.FloatTensor(2, 3)

    pa[0].uniform_(-angle, angle)
    pa[1].uniform_(-trans, trans)
    pa[2].uniform_(-trans, trans)
    pa[3].uniform_(1. - scale, 1. + scale)

    th[0][0] = pa[3] * torch.cos(pa[0])
    th[0][1] = pa[3] * torch.sin(-pa[0])
    th[0][2] = pa[1]
    th[1][0] = pa[3] * torch.sin(pa[0])
    th[1][1] = pa[3] * torch.cos(pa[0])
    th[1][2] = pa[2]

    x = x.unsqueeze(0)
    th = th.unsqueeze(0)
    grid = F.affine_grid(th, x.size(), align_corners=False).cuda(x.get_device())
    out = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return out


def img_random_warp(x):
    out = warp_with_cutout_batch_real(x)
    out_list = []
    for ii in range(out.shape[0]):
        num = random.randint(0, 2)
        if num == 0:
            out_list.append(warp_with_flip(out[ii]))
        elif num == 1:
            out_list.append(warp_with_color(out[ii]))
        else:
            out_list.append(warp_with_affine(out[ii]))
    return torch.cat(out_list, dim=0)


def DiffAugment(x, policy=['color', 'translation', 'cutout'], channels_first=True):

    if x.size(1) == 7:
        x = rearrange(x[:, 0:6], 'b (t c) h w -> (b t) c h w', t=2)
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy:
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0)// 2, 1, 1, 1, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1,2,1,1,1).view(-1,1,1,1) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0) // 2, 1, 1, 1, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1,2,1,1,1).view(-1,1,1,1) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0) // 2, 1, 1, 1, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1,2,1,1,1).view(-1,1,1,1) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0) // 2, 1, 1], device=x.device).unsqueeze(1).repeat(1,2,1,1).view(-1,1,1)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0)// 2, 1, 1], device=x.device).unsqueeze(1).repeat(1,2,1,1).view(-1,1,1)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0) // 2, 1, 1], device=x.device).unsqueeze(1).repeat(1,2,1,1).view(-1,1,1)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0) // 2, 1, 1], device=x.device).unsqueeze(1).repeat(1,2,1,1).view(-1,1,1)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
