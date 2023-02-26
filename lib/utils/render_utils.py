# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************


from curses import resizeterm
from matplotlib import image
import torch
import random
import imageio, os
import trimesh, pyrender, mcubes
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from scipy.spatial import Delaunay
# from skimage.measure import marching_cubes
from einops import rearrange
from tqdm import tqdm
import math
import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.blending import (
    BlendParams,
    softmax_rgb_blend
)

os.environ["PYOPENGL_PLATFORM"] = "egl"


######################### Dataset util functions ###########################
# Get data sampler
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

# Get data minibatch
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

############################## Model weights util functions #################
# Turn model gradients on/off
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

################# Camera parameters sampling ####################
def generate_camera_params(resolution, device, batch=1, locations=None, sweep=False,
                           uniform=False, azim_range=0.3, elev_range=0.15,
                           fov=12, dist_radius=0.12):
    if locations != None:
        azim = locations[:,0].view(-1,1)
        elev = locations[:,1].view(-1,1)

        # generate intrinsic parameters
        # fix distance to 1
        dist = torch.ones(azim.shape[0], 1, device=device)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov / 2 * torch.ones(azim.shape[0], 1, device=device).view(-1,1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
        # generate camera locations on the unit sphere
        azim = (-azim_range + (2 * azim_range / 7) * torch.arange(8, device=device)).view(-1,1).repeat(batch,1)
        elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device).repeat(1,8).view(-1,1))

        # generate intrinsic parameters
        dist = (torch.ones(batch, 1, device=device)).repeat(1,8).view(-1,1)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov / 2 * torch.ones(batch, 1, device=device).repeat(1,8).view(-1,1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    else:
        # sample camera locations on the unit sphere
        if uniform:
            azim = (-azim_range + 2 * azim_range * torch.rand(batch, 1, device=device))
            elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device))
        else:
            azim = (azim_range * torch.randn(batch, 1, device=device))
            elev = (elev_range * torch.randn(batch, 1, device=device))

        # generate intrinsic parameters
        dist = torch.ones(batch, 1, device=device) # restrict camera position to be on the unit sphere
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov / 2 * torch.ones(batch, 1, device=device) * np.pi / 180 # full fov is 12 degrees
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1,3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0,1,0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir, eps=1e-5) # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1, 2), T), -1)
    homo_coord = torch.Tensor([0., 0., 0., 1.]).to(device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1)
    extrinsics = torch.cat((extrinsics, homo_coord), dim=1)
    focal = focal.repeat(1, 1, 2)
    return extrinsics, focal, near, far, viewpoint

#################### Mesh generation util functions ########################
# Reshape sampling volume to camera frostum
def align_volume(volume, near=0.88, far=1.12):
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w),
                                torch.linspace(-1, 1, d))

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = torch.linspace(far / near, 1, d).view(1,1,1,-1,1).to(volume.device)
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[...,:2] = frostum_grid[...,:2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any((frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True)
    frostum_grid = frostum_grid.permute(0,3,1,2,4).contiguous()
    permuted_volume = volume.permute(0,4,3,1,2).contiguous()
    final_volume = F.grid_sample(permuted_volume, frostum_grid, padding_mode="border", align_corners=True)
    final_volume = final_volume.permute(0,3,4,2,1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 1

    return final_volume

# Extract mesh with marching cubes
def extract_mesh_with_marching_cubes(sdf):
    b, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    sdf_vol = sdf[0,...,0].permute(1,0,2).cpu().numpy()

    # scale vertices
    verts, faces, _, _ = marching_cubes(sdf_vol, 0)
    verts[:,0] = (verts[:,0]/float(w)-0.5)*0.24
    verts[:,1] = (verts[:,1]/float(h)-0.5)*0.24
    verts[:,2] = (verts[:,2]/float(d)-0.5)*0.24

    # fix normal direction
    verts[:,2] *= -1; verts[:,1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh

# Generate mesh from xyz point cloud
def xyz2mesh(xyz):
    b, _, h, w = xyz.shape
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    # Extract mesh faces from xyz maps
    tri = Delaunay(np.concatenate((x.reshape((h*w, 1)), y.reshape((h*w, 1))), 1))
    faces = tri.simplices

    # invert normals
    faces[:,[0, 1]] = faces[:,[1, 0]]

    # generate_meshes
    mesh = trimesh.Trimesh(xyz.squeeze(0).permute(1,2,0).view(h*w,3).cpu().numpy(), faces)

    return mesh


################# Mesh rendering util functions #############################
def add_textures(meshes:Meshes, vertex_colors=None) -> Meshes:
    verts = meshes.verts_padded()
    if vertex_colors is None:
        vertex_colors = torch.ones_like(verts) # (N, V, 3)
    textures = TexturesVertex(verts_features=vertex_colors)
    meshes_t = Meshes(
        verts=verts,
        faces=meshes.faces_padded(),
        textures=textures,
        verts_normals=meshes.verts_normals_padded(),
    )
    return meshes_t


def create_cameras(
    R=None, T=None,
    azim=0, elev=0., dist=1.,
    fov=12., znear=0.01,
    device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, at=((0, 0, 0.2),), device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=fov)
    return cameras


def create_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return phong_renderer


def NormalCalcuate(meshes, fragments):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    return pixel_normals


class NormalShader(nn.Module):
    def __init__(self, device="cuda", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        
    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        normals = NormalCalcuate(meshes, fragments)
        images = softmax_rgb_blend(normals, fragments, blend_params)[:,:,:,:3]

        images = F.normalize(images, dim=3)
        # control the color span
        images = images.add_(1.0).div_(2.0)
        return images


def create_mesh_normal_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=NormalShader(device=device)
    )

    return phong_renderer


def _create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        return _create_cam2world_matrix(forward_vectors, camera_origins)


## custom renderer
class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def create_depth_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=17,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=((-0.5, 1., 5.0),), **light_kwargs
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
            device=device,
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return renderer


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size


def pose_spherical(theta, phi, radius, gamma=0):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_gamma(gamma/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def trans_t(t):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
        ], dtype=np.float32)


def rot_phi(phi):
    return np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1],
        ], dtype=np.float32)


def rot_theta(th):
    return np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1],
        ], dtype=np.float32)


def rot_gamma(gamma):
    return np.array([
        [np.cos(gamma),-np.sin(gamma),0,0],
        [np.sin(gamma),np.cos(gamma),0,0],
        [0,0,1,0],
        [0,0,0,1],
        ], dtype=np.float32)


def ellipsoidal_trajectory(xdmax, ydmax, num_steps=120, offset=np.pi/2, is_azel=False):
    a = np.tan(xdmax/np.pi)
    b = np.tan(ydmax/np.pi)
    x = np.concatenate((np.linspace(0, a, num_steps//4), np.linspace(a, -a, num_steps//2), np.linspace(-a, 0., num_steps//4)))
    y = (b**2 - b**2*(x**2 / a**2))**0.5
    y[num_steps//4:num_steps//4+num_steps//2] = -1 * np.abs(y[num_steps//4:num_steps//4+num_steps//2])
    x = np.arctan(x) 
    y = np.arctan(y) 
    if is_azel:
        return -x[::-1], -y[::-1]
    return x + offset, y + offset


def thetaphi_to_azel(theta, phi):
    """ Az-El to Theta-Phi conversion.
  
    Args:
        theta (float or np.array): Theta angle, in radians
        phi (float or np.array): Phi angle, in radians
  
    Returns:
      (az, el): Tuple of corresponding (azimuth, elevation) angles, in radians
    """
    sin_el = np.sin(phi) * np.sin(theta)
    tan_az = np.cos(phi) * np.tan(theta)
    el = np.arcsin(sin_el)
    az = np.arctan(tan_az)
      
    return az, el


def sigma_render(sigmas, num_frames=16, render_frames=120, save_path=None, resize=None):
    threshold = 10
    scenes = []
    ncs = []
    nls = []
    for sigma in sigmas:
        vertices, triangles = mcubes.marching_cubes(sigma, threshold) # 1st frame
        mesh = trimesh.Trimesh(vertices / sigma.shape[1] - .5, triangles)

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.8)

        camera_pose = pose_spherical(-20., -40., 1.8)
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(nc)
        ncs.append(nc)

        # Set up the light -- a point light in the same spot as the camera
        light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
        nl = pyrender.Node(light=light, matrix=camera_pose)
        scene.add_node(nl)
        nls.append(nl)
        scenes.append(scene)

    imgs = []
    resolution = 1024 if resize is None else resize
    r = pyrender.OffscreenRenderer(resolution, resolution)
    angles = np.concatenate((np.linspace(0, 80., 30), np.linspace(80, -80., 60), np.linspace(-80, 0., 30)))
    for idx, gamma in enumerate(tqdm(angles)):
        camera_pose = pose_spherical(180, -90, 1.0)
        scene = scenes[idx%num_frames]
        nc = ncs[idx%num_frames]
        nl = nls[idx%num_frames]
        scene.set_pose(nc, pose=camera_pose)
        scene.set_pose(nl, pose=camera_pose)
        imgs.append(r.render(scene)[0])

    if save_path is not None:
        imageio.mimwrite(save_path, imgs, fps=25)
    imgs = torch.from_numpy(np.array(imgs))
    return imgs


def single_sigma_render(sigmas, render_frames=120, save_path=None, save_obj_path=None, resize=None):
    threshold = 10
    scenes = []
    ncs = []
    nls = []
    num_frames = sigmas.shape[0]
    for sigma in sigmas:
        vertices, triangles = mcubes.marching_cubes(sigma, threshold) # 1st frame
        mesh = trimesh.Trimesh(vertices / sigma.shape[1] - .5, triangles)

        if save_obj_path is not None:
            with open(save_obj_path, 'w') as f:
                mesh.export(f,file_type='obj')

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        camera_pose = pose_spherical(-20., -40., 1.)
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(nc)
        ncs.append(nc)

        # Set up the light -- a point light in the same spot as the camera
        light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
        nl = pyrender.Node(light=light, matrix=camera_pose)
        scene.add_node(nl)
        nls.append(nl)
        scenes.append(scene)

    imgs = []
    resolution = 1024 if resize is None else resize
    r = pyrender.OffscreenRenderer(resolution, resolution)
    angles = np.concatenate((np.linspace(0, 80., 30), np.linspace(80, -80., 60), np.linspace(-80, 0., 30)))
    for idx, gamma in enumerate(angles):
        camera_pose = pose_spherical(180, -90, 1., gamma=gamma)
        scene = scenes[idx%num_frames]
        nc = ncs[idx%num_frames]
        nl = nls[idx%num_frames]
        scene.set_pose(nc, pose=camera_pose)
        scene.set_pose(nl, pose=camera_pose)
        imgs.append(r.render(scene)[0])

    if save_path is not None:
        imageio.mimwrite(save_path, imgs, fps=25)
    imgs = torch.from_numpy(np.array(imgs))
    return imgs


def sigma_mesh_render(sigmas, num_frames=16, render_frames=120, fov=6.1, save_path=None, resize=None):
    threshold = 10
    imgs = []
    device = torch.device("cuda:0")

    trajectory = np.zeros((render_frames, 3), dtype=np.float32)

    t = np.zeros(render_frames)
    
    elev = 1.5 * 0.135 * np.sin((t * 2) * np.pi)
    azim = 1.5 * 0.3 * np.cos((t * 2 - 0.5) * np.pi)

    trajectory[:render_frames,0] = azim
    trajectory[:render_frames,1] = elev
    trajectory[:render_frames,2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    for idx in tqdm(range(render_frames)):
        vertices, triangles = mcubes.marching_cubes(sigmas[idx%num_frames], threshold)
        mesh = trimesh.Trimesh(vertices / sigmas[idx%num_frames].shape[1] - .5, triangles)

        # Render mesh
        mesh = Meshes(
            verts=[torch.from_numpy(np.asarray(mesh.vertices)).to(torch.float32).to(device)],
            faces = [torch.from_numpy(np.asarray(mesh.faces)).to(torch.float32).to(device)],
            textures=None,
            verts_normals=[torch.from_numpy(np.copy(np.asarray(mesh.vertex_normals))).to(torch.float32).to(device)],
        )
        mesh = add_textures(mesh)

        cameras = create_cameras(azim=np.rad2deg(trajectory[idx,0].cpu().numpy()),
                                 elev=np.rad2deg(trajectory[idx,1].cpu().numpy()),
                                 fov=2*trajectory[idx,2].cpu().numpy(),
                                 dist=2.7, device=device)

        renderer = create_mesh_renderer(cameras, image_size=1024 if resize is None else resize,
                                        light_location=((0.0,0.5,5.0),), specular_color=((0.15,0.15,0.15),),
                                        ambient_color=((0.05,0.05,0.05),), diffuse_color=((0.75,.75,.75),), device=device)
        
        mesh_image = renderer(mesh)
        mesh_image = mesh_image.mul(255.0).to(torch.uint8).cpu().numpy()
        mesh_image = mesh_image[...,:3][0]
        imgs.append(mesh_image)

    if save_path is not None:
        imageio.mimwrite(save_path, imgs, fps=25)
    imgs = torch.from_numpy(np.array(imgs))
    return imgs


def depth_render(xyzs, render_frames=120, fov=5.9, save_path=None, resize=None):
    imgs = []
    device = torch.device("cuda:0")

    trajectory = np.zeros((render_frames, 3), dtype=np.float32)

    t = np.linspace(0, 1, render_frames)
    
    elev = 1.5 * 0.135 * np.sin((t * 2 - 0.5) * np.pi)
    azim = 1.5 * 0.3 * np.cos((t * 2 - 0.5) * np.pi)

    trajectory[:render_frames,0] = azim
    trajectory[:render_frames,1] = elev
    trajectory[:render_frames,2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    for idx in tqdm(range(render_frames)):
        xyz = xyzs[idx]
        depth_mesh = xyz2mesh(xyz.permute(2, 0, 1).unsqueeze(0))

        # with open('save/debug/test.obj', 'w') as f:
        #     depth_mesh.export(f, file_type='obj')
                   
        # Render mesh
        mesh = Meshes(
            verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
            faces = [torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
            textures=None,
            verts_normals=[torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
        )
        mesh = add_textures(mesh)

        cameras = create_cameras(azim=np.rad2deg(trajectory[idx,0].cpu().numpy()),
                                 elev=np.rad2deg(trajectory[idx,1].cpu().numpy()),
                                 fov=2*trajectory[idx,2].cpu().numpy(),
                                 dist=2.7, device=device)

        # renderer = create_mesh_renderer(cameras, image_size=1024 if resize is None else resize,
        #                                 light_location=((0.0,0.5,5.0),), specular_color=((0.15,0.15,0.15),),
        #                                 ambient_color=((0.05,0.05,0.05),), diffuse_color=((0.75,.75,.75),), device=device)

        renderer = create_mesh_normal_renderer(cameras, image_size=1024 if resize is None else resize,
                                               light_location=((1.0,1.0,1.0),), specular_color=((0.2,0.2,0.2),),
                                               ambient_color=((0.2,0.2,0.2),), diffuse_color=((0.65,.65,.65),), device=device)

        mesh_image = renderer(mesh).mul(255.0).to(torch.uint8).cpu().numpy()
        mesh_image = mesh_image[...,:3][0]
        imgs.append(mesh_image)

    if save_path is not None:
        imageio.mimwrite(save_path, imgs, fps=25)
    imgs = torch.from_numpy(np.array(imgs))
    return imgs


def _depth_render(xyzs, cams, num_frames=16, render_frames=120, save_path=None, resize=None):
    scenes = []
    for idx in range(render_frames):
        xyz = xyzs[idx]
        cam = cams[idx]
        mesh = xyz2mesh(xyz.permute(2, 0, 1).unsqueeze(0))
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 12.0)

        camera_pose = cam.reshape(4, 4).numpy()
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(nc)

        # Set up the light -- a point light in the same spot as the camera
        light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
        nl = pyrender.Node(light=light, matrix=camera_pose)
        scene.add_node(nl)
        scenes.append(scene)

    imgs = []
    resolution = 1024 if resize is None else resize
    r = pyrender.OffscreenRenderer(resolution, resolution)
    for idx in range(render_frames):
        imgs.append(r.render(scenes[idx])[0])
    if save_path is not None:
        imageio.mimwrite(save_path, imgs, fps=25)
    imgs = torch.from_numpy(np.array(imgs))
    return imgs


def canonical_depth_render(xyzs, num_frames=16, render_frames=120, save_path=None, resize=None):
    threshold = 10
    scenes = []
    ncs = []
    nls = []
    for xyz in xyzs:
        mesh = xyz2mesh(xyz.permute(2, 0, 1).unsqueeze(0))

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 12.0)

        camera_pose = pose_spherical(-20., -40., 1.)
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(nc)
        ncs.append(nc)

        # Set up the light -- a point light in the same spot as the camera
        light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
        nl = pyrender.Node(light=light, matrix=camera_pose)
        scene.add_node(nl)
        nls.append(nl)
        scenes.append(scene)

    imgs = []
    resolution = 1024 if resize is None else resize
    r = pyrender.OffscreenRenderer(resolution, resolution)
    angles = np.concatenate((np.linspace(0, 80., 30), np.linspace(80, -80., 60), np.linspace(-80, 0., 30)))
    for idx, gamma in enumerate(angles):
        camera_pose = pose_spherical(180, -90, 1., gamma=angles[int(0.2*len(angles))])
        scene = scenes[idx%num_frames]
        nc = ncs[idx%num_frames]
        nl = nls[idx%num_frames]
        scene.set_pose(nc, pose=camera_pose)
        scene.set_pose(nl, pose=camera_pose)
        imgs.append(r.render(scene)[0])

    if save_path is not None:
        imageio.mimwrite(save_path, imgs, fps=25)
    imgs = torch.from_numpy(np.array(imgs))
    return imgs


if __name__ == "__main__":
    # sigma_render(torch.load("save/output.pt")["sigma"])
    outputs = torch.load("save/full_cond_late_ws_no_srt_reproduce/render/render/output_0000.pt")
    print(outputs.keys())
    # sigma_render(outputs["sigma"], save_path="./save/mcube.mp4")
    depth_render(outputs["xyz"], outputs["c"], resize=256, save_path="./save/depth.mp4")
    # _depth_render(outputs["xyz"], outputs["c"], resize=256, save_path="./save/_depth.mp4")
    # canonical_depth_render(outputs["canonical_depth"], save_path="./save/cano_depth.mp4")