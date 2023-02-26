# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************


import os.path, json
import random
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from lib.utils.dataset import *
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def preprocess(image):
    # [0, 1] => [-1, 1]
    img = image * 2.0 - 1.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    return img


class VideoDataset(data.Dataset):
    def load_video_frames(self, path):
        data_all = glob.glob(f"{path}/*.mp4")
        self.video_num = len(data_all)
        return data_all

    def __init__(self, path):
        self.path = path
        self.n_frames_G = 16
        self.time_step = 1
        self.video_frame_size = 512
        self.data_all = self.load_video_frames(path)

    def __getitem__(self, index):
        batch_data = self.getTensor(index)
        return batch_data

    def getTensor(self, index):
        n_frames = self.n_frames_G

        V = VideoReader(self.data_all[index], 1e10)
        video = V._read_video(0)
        video = np.array([np.uint8(f.to_rgb().to_ndarray()) for f in video])
        video_len = len(video)

        n_frames_interval = n_frames * self.time_step
        start_idx = random.randint(0, max(0, video_len - 1 - n_frames_interval))
        img = video[0]
        h, w = img.shape[:2]

        if h > w:
            half = (h - w) // 2
            cropsize = (0, half, w, half + w)  # left, upper, right, lower
        elif w > h:
            half = (w - h) // 2
            cropsize = (half, 0, half + h, h)

        images = []
        for i in range(start_idx, start_idx + n_frames_interval,
                       self.time_step):
            path = video[i]
            img = video[i]

            if h != w:
                img = img.crop(cropsize)

            img = cv2.resize(img, 
                (self.video_frame_size, self.video_frame_size))
            img = np.asarray(img, dtype=np.float32)
            img = torch.from_numpy(img).unsqueeze(0)
            images.append(img)

        video_clip = torch.cat(images)
        return video_clip

    def __len__(self):
        return self.video_num

    def name(self):
        return 'VideoDataset'
    
    # def get_label(self, index):

    #     camera_path = self.data_all[index].replace('aligned_faces_video', 'estimated_cameras').replace('mp4', 'json')

    #     with open(camera_path, 'r') as f:
    #         cameras = json.load(f)["labels"]
    #         camera = torch.FloatTensor(list(map(float, cameras[torch.randint(len(cameras), [1]).item()][1])))

    #     if random.random() > 0.5:
    #         camera[[1,2,3,4,8]] *= -1
        
    #     camera = torch.from_numpy(np.array(camera, dtype=np.float32))
    #     return camera
