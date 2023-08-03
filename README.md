## PV3D: A 3D Generative Model for Portrait Video Generation
[Zhongcong Xu*](https://scholar.google.com/citations?user=-4iADzMAAAAJ&hl=en), [Jianfeng Zhang*](http://jeff95.me), [Jun Hao Liew](https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ&hl=en), Wenqing Zhang, [Song Bai](https://songbai.site/), [Jiashi Feng](https://sites.google.com/site/jshfeng/home), [Mike Zheng Shou](https://sites.google.com/view/showlab)

[[Project Page]](https://showlab.github.io/pv3d/), [[OpenReview]](https://openreview.net/forum?id=o3yygm3lnzS)

- [x] codebase
- [x] pretrained checkpoints

## Demo Videos
**Unconditional Generation**
</table>
<table style="border:0px">
   <tr>
       <td><img src="release_docs/assets/teaser_demo.gif" width="400px" frame=void rules=none></td>
       <td><img src="release_docs/assets/vc_128_stitched_demo.gif" width="400px"  frame=void rules=none></td>
   </tr>
</table>

## Installation
```
conda create -n pv3d python=3.7
conda activate pv3d
```
Install pytorch>=1.11.0 based on your CUDA version (e.g. CUDA 11.3):
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install other dependencies:
```
pip install -r requirements.txt
```
Install pytorch3d: 
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
Download pretrained i3d checkpoints from [VideoGPT](https://github.com/wilson1yan/VideoGPT) for [Frechet Video Distance](https://arxiv.org/abs/1812.01717) evaluation:
```
gdown 1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT
mkdir -p ~/.cache/videogpt && mv i3d_pretrained_400.pt ~/.cache/videogpt
```
Download pretrained ArcFace from [InsightFace](https://github.com/deepinsight/insightface) for multi-view identity consistency evaluation:
```
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O ~/.cache/model_ir_se50.pth
```

## Preparing Dataset
Download the processed [voxceleb](https://drive.google.com/drive/folders/1UJtZYPOdD8Rs0g2nz3UXV_o9sBc5_TgF?usp=share_link) dataset, uncompress the video clips and camera sequences.

We do not have the plan to release the video clips for other two datasets due to the potential copyright issues. Instead, we provide the list for selected video clips. Please follow the preprocessing pipeline described in our paper to process them.

## Training
Start training using the following script and override the arguments:
```
python3 run.py \
config=configs/projects/pv3d.yaml \
run_type=train \
model=pv3d \
dataset=voxceleb \
env.save_dir=./save/${save_dir} \
training.batch_size=16 \
training.num_workers=2 \
model_config.pv3d.density_reg=0.6 \
model_config.pv3d.losses[0].params.loss_vid_weight=0.65 \
model_config.pv3d.losses[0].params.r1_gamma=2.0 \
dataset_config.voxceleb.data_dir=${voxceleb_data_path}
```
## Inference
Download the pretrained checkpoints from release page.

**Note:** We use a rendering resolution of 64 for all the experiments except for the geometry visualization in qualitative comparison, where we used a rendering resolution of 128. Please download [voxceleb_res64](https://github.com/bytedance/pv3d/releases/download/v1.0.0/vc_video_reg_06_vid_065_gamma_2_dual_layer_4_res64_reproduce.ckpt) for reproducing the results. As for the teasers, please use the 128 models trained on [voxceleb](https://github.com/bytedance/pv3d/releases/download/v1.0.0/vc_video_reg_06_vid_065_gamma_2_dual_layer_4_res128.ckpt) and [mixed data](https://github.com/bytedance/pv3d/releases/download/v1.0.0/mix_reg_005_vid_065_gamma_4_dual_layer_4_res128_mix.ckpt).

Generating video clips:
```
python3 lib/utils/gen_video.py \
config=configs/projects/pv3d.yaml \
run_type=val model=pv3d dataset=voxceleb \
env.save_dir=./save/${save_dir} \
training.num_workers=2 \
training.rendering_resolution=${rendering_resolution} \
checkpoint.resume_file=save/${checkpoint} \
dataset_config.voxceleb.data_dir=${voxceleb_data_path}
```

## Evaluation
Compute Frechet Video Distance:
```
python3 lib/utils/frechet_video_distance/script/scritpt.py --path save/${save_dir}/videos
```
Generating multi-view results for evaluation:
```
CUDA_VISIBLE_DEVICES=${gpu} python3 lib/utils/gen_multiview.py \
config=configs/projects/pv3d.yaml \
run_type=val model=pv3d dataset=voxceleb \
env.save_dir=./save/${save_dir} \
training.num_workers=2 \
training.rendering_resolution=${rendering_resolution} \
checkpoint.resume_file=save/${checkpoint} \
dataset_config.voxceleb.data_dir=${voxceleb_data_path}
```
Compute Chamfer Distance:
```
python3 lib/utils/chamfer_distance/script.py --path save/${save_dir}/multi/depth/
```
Compute Identity (ID) Consistency:
```
python3 lib/utils/identity_distance/script.py --path save/${save_dir}/multi/images/
```
Compute Multi-view Warping Error:
```
python3 lib/utils/multiview_error/compute_multiview_error.py --path save/${save_dir}/multi --yaw 30 --pitch 0 --size 256
```

## Citation
If you find this codebase useful for your research, please use the following entry.
```
@inproceedings{xu2022pv3d,
    author = {Xu, Eric Zhongcong and Zhang, Jianfeng and Liew, Junhao and Zhang, Wenqing and Bai, Song and Feng, Jiashi and Shou, Mike Zheng},
    title = {PV3D: A 3D Generative Model for Portrait Video Generation},
    booktitle={The Tenth International Conference on Learning Representations},
    year = {2023},
    url = {https://openreview.net/forum?id=o3yygm3lnzS}
}
```
