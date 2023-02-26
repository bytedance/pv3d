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


"""Generate lerp videos using pretrained network pickle."""

import os
import io
import argparse
import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lib.utils.render_utils import LookAtPoseSampler, xyz2mesh
from collections import defaultdict
from pytorch3d.ops.knn import knn_gather, knn_points


CANONICAL_CAMERA = torch.tensor([1.0, 0.0, 0.0, 0.0, 
                    0.0, -1.0, -0.0, 0.0, 
                    0.0, 0.0, -1.0, 2.70, 
                    0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).unsqueeze(0)


#----------------------------------------------------------------------------

def visualize_depth(tensor):
    tensor = tensor.cpu().permute(1,2,0).numpy()
    fig = plt.matshow(tensor, vmin=tensor.min(), vmax=tensor.max(), cmap="magma_r")
    plt.gca().set_axis_off()
    plt.gcf().set_dpi(100)
    io_buf = io.BytesIO()
    plt.savefig(io_buf, bbox_inches='tight', pad_inches=0)
    io_buf.seek(0)
    img_arr = imageio.imread(io_buf)
    io_buf.close()
    tensor_new = torch.tensor(img_arr, device="cpu").permute(2,0,1) / 127.5 - 1
    return tensor_new

def color_depth_map(depths, scale=None):
    """
    Color an input depth map.
    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)
    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """
    vmin, vmax = 0.8, 1.1
    depths = (depths.cpu().squeeze(0).numpy()-vmin) / (vmax-vmin)
    _color_map_depths = np.array([
      [0, 0, 0],  # 0.000
      [0, 0, 255],  # 0.114
      [255, 0, 0],  # 0.299
      [255, 0, 255],  # 0.413
      [0, 255, 0],  # 0.587
      [0, 255, 255],  # 0.701
      [255, 255, 0],  # 0.886
      [255, 255, 255],  # 1.000
      [255, 255, 255],  # 1.000
    ]).astype(float)
    _color_map_bincenters = np.array([
      0.0,
      0.114,
      0.299,
      0.413,
      0.587,
      0.701,
      0.886,
      1.000,
      2.000,  # doesn't make a difference, just strictly higher than 1
    ])
  
    if scale is None:
      scale = depths.max()
  
    values = np.clip(depths.flatten() / scale, 0, 1)
    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
      lower_bin + 1] * alphas.reshape(-1, 1)
    colored = colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)
    colored = torch.tensor(colored, device="cpu").permute(2,0,1) / 127.5 - 1
    return colored

#----------------------------------------------------------------------------

def get_rays(cam2world_matrix, intrinsics, resolution):
    """
    Create batches of rays and return origins and directions.

    cam2world_matrix: (N, 4, 4)
    intrinsics: (N, 3, 3)
    resolution: int

    ray_origins: (N, M, 3)
    ray_dirs: (N, M, 2)
    """
    N, M = cam2world_matrix.shape[0], resolution**2
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = torch.ones((N, M), device=cam2world_matrix.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

    world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    return ray_origins, ray_dirs

#----------------------------------------------------------------------------

def get_mask_dist(indices, dist, img_size):
    mask = torch.ones_like(dist).to(torch.bool)
    indices = ((indices + 1) / 2 * img_size).to(torch.uint8)
    dist_pixel = defaultdict(list)
    for i in range(img_size):
        for j in range(img_size):
            dist_pixel[(indices[:, i, j, 0].item(), indices[:, i, j, 1].item())].append([(i, j), dist[:, i, j]])
    for k, dists in dist_pixel.items():
        dists.sort(key=lambda x:x[1])
        mask[:, dists[0][0][0], dists[0][0][1]] = False
    return mask

#----------------------------------------------------------------------------

def warp(c_prim, c_aux, img_aux, depth_prim, depth_aux, img_size, coarse_size=64, cham_d_thres=0.015, visualize=False):
    # warp batch of aux_img to prim view given depth_prim
    device = c_prim.device
    batch_size = c_prim.shape[0]
    depth_prim = depth_prim.cpu()
    _depth_prim = F.interpolate(depth_prim.unsqueeze(0).cpu(), [coarse_size, coarse_size])
    ray_origins, ray_dirs = get_rays(c_prim[:, :16].reshape(batch_size, 4, 4), c_prim[:, 16:].reshape(batch_size, 3, 3), coarse_size)
    primary_points_3d = ray_origins + ray_dirs * _depth_prim.reshape(batch_size, coarse_size*coarse_size, 1)
    primary_points_3d = primary_points_3d.cuda()

    depth_aux = F.interpolate(depth_aux.unsqueeze(0).cpu(), [coarse_size, coarse_size], align_corners=False, mode='bilinear')
    ray_origins, ray_dirs = get_rays(c_aux[:, :16].reshape(batch_size, 4, 4), c_aux[:, 16:].reshape(batch_size, 3, 3), coarse_size)
    aux_points_3d = ray_origins + ray_dirs * depth_aux.reshape(batch_size, coarse_size*coarse_size, 1)
    aux_points_3d = aux_points_3d.cuda()

    x_nn = knn_points(primary_points_3d, aux_points_3d, norm=2, K=1)
    y_nn = knn_points(aux_points_3d, primary_points_3d, norm=2, K=1)
    cham_x = x_nn.dists[..., 0].reshape(batch_size, coarse_size, coarse_size)  # (N, P1)
    cham_y = y_nn.dists[..., 0].reshape(batch_size, coarse_size, coarse_size)  # (N, P2)

    distance = (cham_x + cham_y) * 0.5
    distance = F.interpolate(distance.unsqueeze(1), [img_size, img_size], align_corners=False, mode='bilinear')
    mask_occ = (distance[:, 0].sqrt() > cham_d_thres).repeat(1,3,1,1).cpu()

    depth_prim = F.interpolate(depth_prim.unsqueeze(0).cpu(), [img_size, img_size], align_corners=False, mode='bilinear')
    ray_origins, ray_dirs = get_rays(c_prim[:, :16].reshape(batch_size, 4, 4), c_prim[:, 16:].reshape(batch_size, 3, 3), img_size)
    primary_points_3d = ray_origins + ray_dirs * depth_prim.reshape(batch_size, img_size*img_size, 1)
    primary_points_homogeneous = torch.ones((batch_size, img_size*img_size, 4), device=device)
    primary_points_homogeneous[:, :, :3] = primary_points_3d
    primary_points_project_to_auxiliary = torch.bmm(torch.inverse(c_aux[:, :16].reshape(batch_size, 4, 4)), primary_points_homogeneous.permute(0, 2, 1)).permute(0, 2, 1).reshape(batch_size, img_size, img_size, 4)
    

    primary_grid_in_auxiliary = torch.cat((primary_points_project_to_auxiliary[..., 0:1] / primary_points_project_to_auxiliary[..., 2:3], primary_points_project_to_auxiliary[..., 1:2] / primary_points_project_to_auxiliary[..., 2:3]), -1)
    primary_grid_in_auxiliary = (primary_grid_in_auxiliary * c_prim[:, 16] + c_prim[:, 18]) * 2 - 1

    mask_range = (torch.abs(primary_grid_in_auxiliary).amax(-1) > 1.0).unsqueeze(0).repeat(1,3,1,1)

    warp_img = F.grid_sample(img_aux.permute(2, 0, 1).unsqueeze(0), primary_grid_in_auxiliary, align_corners=True)

    mask = torch.logical_or(mask_occ, mask_range)
    warp_img = warp_img.add(1.0).div(2.0).mul(255.0).to(torch.uint8)
    warp_img[mask] = torch.ones_like(warp_img[mask]) * 0
    warp_img = warp_img[0].permute(1,2,0)
    mask = mask[0].permute(1,2,0)

    if visualize:
        cv2.imwrite("warp.jpg", warp_img.cpu().numpy()[:,:,::-1])

    return warp_img, ~mask

#----------------------------------------------------------------------------

def warp_side(c_prim, c_aux, img_front, img_aux, depth_prim, depth_aux, img_size=512, visualize=False):
    # warp batch of aux_img to prim view given depth_prim
    device = c_prim.device
    batch_size = c_prim.shape[0]
    depth_aux = F.interpolate(depth_aux.unsqueeze(0).cpu(), [img_size, img_size], align_corners=False, mode='bilinear')

    ray_origins, ray_dirs = get_rays(c_aux[:, :16].reshape(batch_size, 4, 4), c_aux[:, 16:].reshape(batch_size, 3, 3), img_size)
    primary_points_3d = ray_origins + ray_dirs * depth_aux.reshape(batch_size, img_size*img_size, 1)
    # import pdb; pdb.set_trace()
    primary_points_homogeneous = torch.ones((batch_size, img_size*img_size, 4), device=device)
    primary_points_homogeneous[:, :, :3] = primary_points_3d

    primary_points_project_to_auxiliary = torch.bmm(torch.inverse(c_prim[:, :16].reshape(batch_size, 4, 4)), primary_points_homogeneous.permute(0, 2, 1)).permute(0, 2, 1).reshape(batch_size, img_size, img_size, 4)
    primary_grid_in_auxiliary = torch.cat((primary_points_project_to_auxiliary[..., 0:1] / primary_points_project_to_auxiliary[..., 2:3], primary_points_project_to_auxiliary[..., 1:2] / primary_points_project_to_auxiliary[..., 2:3]), -1)
    primary_grid_in_auxiliary = (primary_grid_in_auxiliary * c_aux[:, 16] + c_aux[:, 18]) * 2 - 1

    mask_grid = torch.abs(primary_grid_in_auxiliary).amax(-1) > 1.0
    mask_grid = mask_grid.repeat(3,1,1).permute(1, 2, 0)

    pixel_index = ((primary_grid_in_auxiliary + 1) / 2.0 * img_size).to(torch.int16).clamp(0, img_size-1)

    warp_img = img_front.clone() #torch.zeros_like(img_aux)
    pixel_index = pixel_index.reshape(-1, 2)
    prim_x = pixel_index[:, 0].long()
    prim_y = pixel_index[:, 1].long()
    orig_index = torch.stack(torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')).permute(1, 2, 0)
    orig_index = orig_index.reshape(-1, 2)
    orig_x = orig_index[:, 0].long()
    orig_y = orig_index[:, 1].long()
    warp_img[prim_y, prim_x] = img_aux[orig_x, orig_y]
    mask_index = torch.ones_like(img_aux).to(torch.bool)
    mask_index[prim_y, prim_x] = False
    mask = torch.logical_or(mask_index, mask_grid)
    warp_img[mask] = torch.zeros_like(warp_img[mask])

    if visualize:
        cv2.imwrite("warp.jpg", (warp_img.to(torch.uint8).numpy())[:,:,::-1])

    return warp_img, ~mask

#----------------------------------------------------------------------------

def warp_xyz(c_prim, c_aux, img_prim, img_aux, depth_prim, depth_aux=None, img_size=512):
    # warp batch of aux_img to prim view given depth_prim
    device = c_prim.device
    batch_size = c_prim.shape[0]
    primary_points_3d = F.interpolate(depth_prim.permute(2, 0, 1).unsqueeze(0).cpu(), [img_size, img_size], align_corners=False, mode='bilinear')

    primary_points_homogeneous = torch.ones((batch_size, img_size*img_size, 4), device=device)
    primary_points_homogeneous[:, :, :3] = primary_points_3d.permute(0, 2, 3, 1).reshape(batch_size, img_size*img_size, 3)

    primary_points_project_to_auxiliary = torch.bmm(torch.inverse(c_aux[:, :16].reshape(batch_size, 4, 4)), primary_points_homogeneous.permute(0, 2, 1)).permute(0, 2, 1).reshape(batch_size, img_size, img_size, 4)
    primary_grid_in_auxiliary = torch.cat((primary_points_project_to_auxiliary[..., 0:1] / primary_points_project_to_auxiliary[..., 2:3], primary_points_project_to_auxiliary[..., 1:2] / primary_points_project_to_auxiliary[..., 2:3]), -1)
    primary_grid_in_auxiliary = (primary_grid_in_auxiliary * c_prim[:, 16] + c_prim[:, 18])*2-1
    # primary_grid_in_auxiliary = torch.nn.functional.interpolate(primary_grid_in_auxiliary.permute(0,3,1,2), img_prim.shape[2:], mode='bilinear').permute(0,2,3,1)
    # depth_prim = torch.nn.functional.interpolate(depth_prim, img_prim.shape[2:], mode='bilinear')
    # compute mask if the projected pixel is out of range
    mask_grid = torch.abs(primary_grid_in_auxiliary).amax(-1)>1
    mask_grid = mask_grid.unsqueeze(0).repeat(1,3,1,1)
    # compute mask if the projected pixel is bg
    # mask = (depth_prim <= -0.23)

    # mask = mask.repeat(1,3,1,1)
    warp_img = F.grid_sample(img_aux.permute(2, 0, 1).unsqueeze(0), primary_grid_in_auxiliary, align_corners=True)
    
    # use black bg
    # warp_img[mask_grid] = torch.ones_like(warp_img[mask_grid])*(0-128)/127.5
    # warp_img[mask] = torch.ones_like(warp_img[mask])*(0-128)/127.5

    # use gray bg
    # warp_img[mask] = torch.ones_like(warp_img[mask]) * 128
    warp_img[mask_grid] = torch.ones_like(warp_img[mask_grid]) * 128

    _warp_img = warp_img[0].permute(1,2,0).to(torch.uint8).cpu().numpy()
    cv2.imwrite("warp.jpg", _warp_img[:,:,::-1])

    return warp_img

#----------------------------------------------------------------------------

def compute_error(args, num_videos=1000, num_frames=4, **video_kwargs):
    errors_mean = []
    errors_median = []
    errors_single_mean = []
    errors_single_median = []
    errors_cross_mean = []
    errors_cross_median = []
    for idx in tqdm(range(num_videos)):
        # load depth and images
        depths = torch.load(f"{args.path}/depth/{idx:04d}.pt")["depth"]
        pairs = [[0, 2], [3, 1], [0, 1], [3, 2]]
        imgs = []
        for i in range(num_frames):
            img = cv2.cvtColor(cv2.imread(f"{args.path}/images/{idx:04d}_{i:02d}.png"), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, [args.size, args.size])
            imgs.append(torch.from_numpy(img).to(torch.float32))

        # side view camera
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + args.yaw/180*np.pi,
                                                np.pi/2 -0.05 + args.pitch/180*np.pi,
                                                torch.tensor([0, 0, 0.2]), radius=2.7)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
        SIDE_CAMERA = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        for i, pair in enumerate(pairs):
            img_front = imgs[pair[0]]
            depth_front = depths[pair[0]]
            img_side = imgs[pair[1]]
            depth_side = depths[pair[1]]
            # img_warp, mask = warp(CANONICAL_CAMERA, SIDE_CAMERA, (img_side-127.5)/127.5, depth_front, depth_side, img_size=args.size, visualize=args.visualize)
            img_warp, mask = warp_side(CANONICAL_CAMERA, SIDE_CAMERA, img_front, img_side, depth_front, depth_side, img_size=args.size, visualize=args.visualize)
            abs_error = torch.abs(img_warp.clamp(0, 255) - img_front.clamp(0, 255))[mask]
            error_mean = torch.mean(abs_error)
            error_median = torch.median(abs_error)
            if torch.isnan(error_mean).any().item() or torch.isnan(error_median).any().item():
                # skip nan generated by failure cases
                continue
            error_mean = error_mean.item()
            error_median = error_median.item()
            errors_mean.append(error_mean)
            errors_median.append(error_median)
            if i < 2:
                errors_single_mean.append(error_mean)
                errors_single_median.append(error_median)
            else:
                errors_cross_mean.append(error_mean)
                errors_cross_median.append(error_median)
    print("avg mean error", np.mean(errors_mean))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("compute multi-view reprojection error")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--yaw", type=int, required=True)
    parser.add_argument("--pitch", type=int, required=True)
    parser.add_argument("--size", type=int, default=256, required=False)
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()
    compute_error(args)

#----------------------------------------------------------------------------
