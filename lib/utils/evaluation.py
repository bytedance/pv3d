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


"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""
import logging
import os
import shutil
import glob
import torch
import imageio
from tqdm import tqdm
from PIL import Image
from lib.utils.general import updir
from lib.common.registry import registry
from torchvision.utils import save_image
from lib.common.build import build_dataset
from torchvision.io import read_video
from lib.utils.distributed import is_master, synchronize
from lib.utils.configuration import get_env, get_global_config


logger = logging.getLogger(__name__)


def setup_evaluation(real_dir,
                     fake_dir, 
                     target_size=128,
                     num_eval_imgs=8000,
                     **kwargs):

    save_dir = get_env(key="save_dir")
    real_dir = os.path.join(save_dir, real_dir)
    fake_dir = os.path.join(save_dir, fake_dir)
    registry.register("real_dir", real_dir)
    registry.register("fake_dir", fake_dir)

    if is_master():
        # remove fake dir if they have been made
        os.makedirs(fake_dir, exist_ok=True)
    synchronize()


def output_real_imgs(save_dir, 
                     real_dir,
                     target_size=128,
                     num_eval_imgs=8000,
                     **kwargs):
    # Only make real images if they haven't been made yet
    if os.path.exists(real_dir):
        img_list = glob.glob(f"{real_dir}/*.jpg") + glob.glob(f"{real_dir}/*.png")
        if len(img_list) > 0:
            img_path = img_list[0]
            img_pil = Image.open(img_path)
            if img_pil.size[0] != target_size:
                shutil.rmtree(real_dir)
                logger.info(f"Delete real images at {img_pil.size}")
    
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        logger.info("Writing real images")
        logger.info("Building evaluation dataloader")
        config = registry.get("config")
        dataloader = build_dataset(config).train_dataloader()
        save_real_images(dataloader, num_eval_imgs, real_dir)
        logger.info("Done")
    else:
        logger.info("Real images exist")


def save_real_images(dataloader, num_eval_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    pbar = tqdm(desc=f"Output real images", total=num_eval_imgs)
    for i in range(num_eval_imgs//batch_size+1):
        try:
            sample = next(dataloader)
        except StopIteration:
            break
        if len(sample["frames"].shape) >= 5:
            sample["frames"] = sample["frames"].squeeze(2)
        for img in sample["frames"]:
            save_image(img, os.path.join(real_dir, 
                        f'{img_counter:0>5}.jpg'), 
                        normalize=True, value_range=(-1, 1))
            pbar.update()
            img_counter += 1
            if img_counter >= num_eval_imgs:
                return


def setup_fvd_evaluation(real_dir,
                     fake_dir, 
                     target_size=128,
                     num_frames=16, 
                     num_eval_videos=8000,
                     **kwargs):

    save_dir = get_env(key="save_dir")
    real_dir = os.path.join(save_dir, real_dir)
    fake_dir = os.path.join(save_dir, fake_dir)
    registry.register("real_dir", real_dir)
    registry.register("fake_dir", fake_dir)

    if is_master():
        os.makedirs(fake_dir, exist_ok=True)
    synchronize()


def output_real_videos(save_dir, 
                     real_dir,
                     target_size=128, 
                     num_frames=16, 
                     num_eval_videos=8000,
                     **kwargs):
    # Only make real videos if they haven't been made yet
    if os.path.exists(real_dir):
        video_list = glob.glob(f"{real_dir}/*.mp4")
        if len(video_list) > 0:
            frames = read_video(video_list[0])[0]
            if (frames[0].shape[0] != target_size) or (len(frames) != num_frames):
                shutil.rmtree(real_dir)
                logger.info(f"deleting real videos at {len(frames)} X {list(frames[0].shape)}")
                cache_path = "{}/real.cache".format(updir(real_dir, 2))
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    logger.info("deleting real feat cache")
        else:
            shutil.rmtree(real_dir)

    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        logger.info("Writing real videos")
        logger.info("Building evaluation dataloader")
        config = registry.get("config")
        dataloader = build_dataset(config).train_dataloader()
        if hasattr(dataloader.dataset, "bi_frame"):
            dataloader.dataset.bi_frame = False
        save_real_videos(dataloader, num_eval_videos, real_dir)
        logger.info("Done")
    else:
        logger.info("Real videos exist")


def save_real_videos(dataloader, num_eval_videos, real_dir):
    video_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    pbar = tqdm(desc=f"Output real videos", total=num_eval_videos)
    for i in range(num_eval_videos//batch_size + 1):
        try:
            sample = next(dataloader)
        except StopIteration:
            break
        for video in sample["frames"]:
            video = video.clamp(-1, 1).cpu()
            video = ((video + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
            imageio.mimwrite(os.path.join(real_dir, f'{video_counter:0>5}.mp4'), video, fps=25)
            pbar.update()
            video_counter += 1
            if video_counter >= num_eval_videos:
                return
