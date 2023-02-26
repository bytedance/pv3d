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


import torch
import os, uuid
import logging
import hashlib
import numpy as np
import pickle
import math
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from lib.utils.general import PathManager, get_z, get_z_motion, get_t
from lib.common.registry import registry
from lib.utils.configuration import get_env
from lib.utils.distributed import broadcast_tensor, get_rank, get_world_size, synchronize, is_master
from lib.utils.frechet_video_distance.i3d.download import download
from lib.utils.frechet_video_distance.i3d.pytorch_i3d import InceptionI3d


logger = logging.getLogger(__name__)


_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'


def compute_feature_stats_for_dataset(dataloader, max_items=None, cache=True, **stats_kwargs):
    device=torch.device('cuda:{}'.format(get_rank()))
    # Try to lookup from cache.
    cache_file = None
    if cache:
        # Choose cache file name.
        args = dict(
                dataset=dataloader.dataset.dataset_name, 
                img_size=dataloader.dataset.config.img_size, 
                max_items=max_items, 
                detector=f"pytorch_i3d_{_I3D_PRETRAINED_ID}"
            )
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataloader.dataset.dataset_name}-{dataloader.dataset.config.img_size}-pytorch_i3d_{_I3D_PRETRAINED_ID}-{md5.hexdigest()}'
        cache_file = os.path.join(get_env("cache_dir"), 'gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        if PathManager.exists(cache_file):
            return FeatureStats.load(cache_file)
    logger.info("Extracting ground truth disctribution...")
    # Initialize.
    num_items = len(dataloader.dataset)
    if max_items is not None:
        if num_items < max_items:
            logger.warn(f"{num_items} samples in dataset, less than required {max_items}, using {num_items}")
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    detector = get_feature_detector(device=device)

    # Main loop.
    for batch in dataloader:
        frames = preprocess(batch["frames"])
        features = detector(frames.to(device))
        stats.append_torch(features, num_gpus=get_world_size(), rank=get_rank())
        if stats.is_full():
            break

    # Save to cache.
    if cache_file is not None and is_master():
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats


def compute_feature_stats_for_generator(generator, camera_dataset, batch_size=4, batch_gen=None, chunk=1, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0
    device=torch.device('cuda:{}'.format(get_rank()))
    # Setup generator and labels.
    generator.eval()

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    detector = get_feature_detector(device=device)
    synchronize()
    logger.info("Extracting generated disctribution...")
    # Main loop.
    while not stats.is_full():
        images = []
        for _ in range(batch_size // batch_gen):
            with torch.no_grad():
                Ts = get_t(batch_gen, device=device, test=True)
                num_frames = Ts.shape[1]
                Ts = Ts.reshape(-1, 1)
                z = get_z(batch_gen, device=device)
                z = z.unsqueeze(1).repeat(1, num_frames, 1).reshape(-1, z.shape[-1])
                gen_c = [camera_dataset.get_label(torch.randint(0, len(camera_dataset.idx_db), [1]).item()).unsqueeze(0) for _ in range(batch_gen)]
                gen_c = torch.cat(gen_c, dim=0).cuda()
                gen_c0 = gen_c[:, 0:1].repeat(1, num_frames, 1)
                gen_c = gen_c.reshape(-1, gen_c.shape[-1])
                gen_c0 = gen_c0.reshape(-1, gen_c0.shape[-1])
                z_motion = get_z_motion(batch_gen, device=device)
                z_motion = z_motion.unsqueeze(1).repeat(1, num_frames, 1).reshape(-1, z_motion.shape[-1])
                imgs = []
                for i in range(len(z)//chunk):
                    img = generator(z[i*chunk:(i+1)*chunk], z_motion[i*chunk:(i+1)*chunk], Ts[i*chunk:(i+1)*chunk], gen_c[i*chunk:(i+1)*chunk], c0=gen_c0[i*chunk:(i+1)*chunk], noise_mode='const')["image"]
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                image = rearrange(imgs, '(b t) c h w -> b t c h w', t=num_frames)
            image = image.permute(0, 1, 3, 4, 2).add_(1.0).div_(2).mul(255).clamp_(0, 255).to(torch.uint8)
            images.append(image)
        images = torch.cat(images)
        images = preprocess(images)
        features = detector(images.to(device))
        stats.append_torch(features, num_gpus=get_world_size(), rank=get_rank())
    return stats


def get_feature_detector(device=torch.device('cpu')):
    i3d = InceptionI3d(400, in_channels=3)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval().to(device)
    return i3d


def build_camera_loader(cameras, batch_gen):
    camera_dataset = CameraDataset(cameras)
    if get_world_size() > 1:
        args = {
            "sampler":DistributedSampler(camera_dataset, shuffle=True, seed=registry.get("seed"))
        }
    else:
        args = { "shuffle": True }
    camera_loader = DataLoader(
                    camera_dataset, 
                    batch_size=batch_gen,
                    **args
                )
    return camera_loader


def preprocess_single(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video


def preprocess(videos, target_resolution=224):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    videos = torch.stack([preprocess_single(video, target_resolution) for video in videos])
    return videos * 2


class CameraDataset(Dataset):
    def __init__(self, cameras):
        super().__init__()
        self.cameras = [cameras[k] for k in cameras]
    
    def __getitem__(self, index):
        return torch.tensor(self.cameras[index]).to(torch.float32)

    def __len__(self):
        return len(self.cameras)


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                broadcast_tensor(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = pickle.load(f)
        obj = FeatureStats(capture_all=s["capture_all"], max_items=s["max_items"])
        obj.__dict__.update(s)
        return obj