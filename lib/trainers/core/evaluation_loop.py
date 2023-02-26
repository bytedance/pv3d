# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


from json import load
import logging, imageio
from operator import is_
from abc import ABC
from typing import Any, Dict, Tuple, Type
from einops import rearrange
import torch
import json
import math, cv2, mcubes, trimesh, os
import numpy as np
from tqdm import tqdm
from lib.common.meter import Meter
from lib.common.report import Report
from lib.common.registry import registry
from lib.utils.configuration import get_env
from lib.utils.distributed import gather_tensor, get_world_size, get_rank, is_dist_initialized, synchronize
from lib.utils.general import get_t, move_to_device, infer_batch_size, PathManager, get_z, get_z_motion
from lib.utils.render_utils import *


logger = logging.getLogger(__name__)


CANONICAL_CAMERA = [1.0, 0.0, 0.0, 0.0, 
                    0.0, -1.0, -0.0, 0.0, 
                    0.0, 0.0, -1.0, 2.70, 
                    0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, dataset_type: str, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()
        use_cpu = self.config.evaluation.get("use_cpu", False)
        loaded_batches = 0
        skipped_batches = 0
        img_viz_grid = self.config.training.img_viz_grid

        model = self.model.module if is_dist_initialized() else self.model
        generator = model.generator_ema if self.training_config.use_ema else model.generator
        # get dataset for real cameras
        dataset = getattr(getattr(self, f"train_loader"), "dataset")
        grid_z = get_z(img_viz_grid[0]*img_viz_grid[1], z_dim=generator.z_dim)
        grid_z_motion = get_z_motion(img_viz_grid[0]*img_viz_grid[1], z_dim=generator.z_dim)
        grid_t = get_t(img_viz_grid[0]*img_viz_grid[1])
        grid_c = [dataset.get_label(torch.randint(0, len(dataset), [1]).item()).unsqueeze(0) for _ in range(img_viz_grid[0])]
        # grid_c = torch.cat(grid_c, dim=0).unsqueeze(1).repeat(1, img_viz_grid[1], 1).reshape(grid_z.shape[0], -1)
        grid_c = torch.cat(grid_c, dim=0).unsqueeze(1).repeat(1, img_viz_grid[1], 1, 1).reshape(grid_z.shape[0], 2, -1)

        with torch.no_grad():
            self.model.eval()
            torch.cuda.empty_cache()
            combined_report = None
            for z, z_motion, Ts, c in zip(grid_z, grid_z_motion, grid_t, grid_c):
                batch = dict(z=z.unsqueeze(0).repeat(1, 1), c=c, z_motion=z_motion.unsqueeze(0).repeat(1, 1), Ts=Ts.unsqueeze(0),
                             dataset_type=dataset_type, dataset_name=dataset.dataset_name)
                prepared_batch = move_to_device(batch, self.device)
                model_output = model.forward_evaluation(prepared_batch)
                report = Report(prepared_batch, model_output)
                report = report.detach()
                moved_report = report
                # Move to CPU for metrics calculation later if needed
                # Explicitly use `non_blocking=False` as this can cause
                # race conditions in next accumulate
                if use_cpu:
                    moved_report = report.copy().to("cpu", non_blocking=False)
                # accumulate necessary params for metric calculation
                if combined_report is None:
                    # make a copy of report since `reporter.add_to_report` will
                    # change some of the report keys later
                    combined_report = moved_report.copy()
                    combined_report.real_dir = registry.get("real_dir", None)
                    combined_report.fake_dir = registry.get("fake_dir", None)
                else:
                    combined_report._accumulate_tensor_fields(
                        moved_report, self.metrics.required_params
                    )
                loaded_batches += batch["z"].shape[0]
                
            combined_report.num_samples = self.num_samples
            combined_report.num_updates = self.num_updates
            combined_report.nrow = img_viz_grid[1]
            combined_report.save_generated_images()
            # combined_report.save_generated_videos(loaded_batches)

            logger.info(f"Finished inferrence. Loaded {loaded_batches}")
            logger.info(f" -- skipped {skipped_batches} batches.")

            assert (
                combined_report is not None
            ), "Please check if your validation set is empty!"
            # add prediction_report is used for set-level metrics
            combined_report.generator = generator
            combined_report.metrics = self.metrics(combined_report, combined_report)

            # Since update_meter will reduce the metrics over GPUs, we need to
            # move them back to GPU but we will only move metrics and losses
            # which are needed by update_meter to avoid OOM
            # Furthermore, do it in a non_blocking way to avoid any issues
            # in device to host or host to device transfer
            if use_cpu:
                combined_report = combined_report.to(
                    self.device, fields=["metrics", "losses"], non_blocking=False
                )

            meter.update_from_report(combined_report, should_update_loss=False)

        # enable train mode again
        self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        skipped_batches = 0
        loaded_batches = 0
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()
                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm(dataloader)
                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = move_to_device(prepared_batch, self.device)
                    loaded_batches += 1
                    if not validate_batch_sizes(prepared_batch.get_batch_size()):
                        logger.info("Skip batch due to unequal batch sizes.")
                        skipped_batches += 1
                        continue
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)
                    report.detach()

                reporter.postprocess_dataset_report()

            logger.info(f"Finished predicting. Loaded {loaded_batches}")
            logger.info(f" -- skipped {skipped_batches} batches.")
            self.model.train()

    def img_rendering_loop(self, dataset_type: str) -> None:

        # img_viz_grid = self.config.training.img_viz_grid

        model = self.model.module if is_dist_initialized() else self.model
        generator = model.generator_ema if self.training_config.use_ema else model.generator

        with torch.no_grad():
            for i in tqdm(range(8)):
                rand_seed = torch.Generator(device=self.device)
                rand_seed.manual_seed(i+88888000008)
                z = torch.randn([1, 512], device=self.device, generator=rand_seed)
                z_motion = torch.randn([1, 512], device=self.device, generator=rand_seed)
                canonical_c = torch.tensor(CANONICAL_CAMERA, device=self.device).reshape(-1, 25)
                Ts = get_t(1, device=self.device)

                scale = 0.5
                neural_rendering_resolution = 128
                num_steps = 128
                sx, sy, sz = torch.meshgrid(torch.linspace(-scale, scale, neural_rendering_resolution, device=self.device),
                                torch.linspace(-scale, scale, neural_rendering_resolution, device=self.device), 
                                torch.linspace(-scale, scale, num_steps, device=self.device), indexing="ij")
                sxyz = torch.stack([sx, sy, sz], dim=-1)
                sxyz = sxyz.view(neural_rendering_resolution*neural_rendering_resolution*num_steps, 3).unsqueeze(0).expand(2, -1, -1)
                outputs = generator.sample(sxyz, torch.zeros_like(sxyz), z.repeat(2, 1), canonical_c.repeat(2, 1), z_motion=z_motion.repeat(2, 1), Ts=Ts)
                sigma = outputs["sigma"][0].detach().cpu().reshape(neural_rendering_resolution, neural_rendering_resolution, num_steps).numpy()
                
                sigma_vid_save_dir = "{}/sigmas".format(get_env(key="save_dir"))
                if not PathManager.exists(sigma_vid_save_dir):
                    PathManager.mkdirs(sigma_vid_save_dir)
                
                sigma_obj_save_dir = "{}/objs".format(get_env(key="save_dir"))
                if not PathManager.exists(sigma_obj_save_dir):
                    PathManager.mkdirs(sigma_obj_save_dir)

                single_sigma_render(sigma[None, :], save_path=f"{sigma_vid_save_dir}/sigma_{i}.mp4", 
                                    save_obj_path=f"{sigma_obj_save_dir}/sigma_{i}.obj")

        logger.info(f"Finished rendering")

        # enable train mode again
        self.model.train()

    def rendering_loop(self, dataset_type: str) -> None:
        num_videos = 16
        num_frames = 16
        render_frames = 120
        batch_gen = 1 # int(render_frames / 20)
        assert render_frames%batch_gen == 0
        scale = 0.5
        neural_rendering_resolution = 128
        num_steps = 128
        sx, sy, sz = torch.meshgrid(torch.linspace(-scale, scale, neural_rendering_resolution, device=self.device),
                        torch.linspace(-scale, scale, neural_rendering_resolution, device=self.device), 
                        torch.linspace(-scale, scale, num_steps, device=self.device), indexing="ij")
        sxyz = torch.stack([sx, sy, sz], dim=-1)
        sxyz = sxyz.view(neural_rendering_resolution*neural_rendering_resolution*num_steps, 3).unsqueeze(0).expand(batch_gen, -1, -1)

        with torch.no_grad():
            self.model.eval()
            generator = self.model.generator_ema if self.training_config.use_ema else self.model.generator
            logger.info(f"Starting rendering")

            for batch_idx in range(num_videos):
                rand_seed = torch.Generator(device=self.device)
                rand_seed.manual_seed(batch_idx + 80000000000)
                z = torch.randn([1, 512], device=self.device, generator=rand_seed)
                z = z.unsqueeze(1).repeat(1, render_frames, 1).reshape(-1, z.shape[-1])
                Ts = torch.linspace(0, 1., steps=num_frames).view(num_frames, 1, 1).unsqueeze(0)
                Ts = Ts.repeat(1, 1, 1, 1).view(-1, 1, 1, 1).to(self.device)
                Ts = Ts.repeat(render_frames//num_frames+1, 1, 1, 1)[:render_frames]
                z_motion = torch.randn([1, 512], device=self.device, generator=rand_seed)
                z_motion = z_motion.unsqueeze(1).repeat(1, render_frames, 1).reshape(-1, z_motion.shape[-1])

                pitch_range = 0.25
                yaw_range = 0.35
                c = []
                for i in range(render_frames):
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * i / render_frames),
                                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * i / render_frames),
                                                            torch.tensor([0, 0, 0.2], device=self.device), radius=2.7, device=self.device)
                    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device)
                    c.append(torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1))
                c = torch.cat(c, dim=0)
                canonical_c = torch.tensor(CANONICAL_CAMERA, device=self.device).unsqueeze(0).repeat(render_frames, 1).reshape(-1, 25)
                # with open("/mnt/bd/bytedrive-zcxu-hl01/Data/voxceleb/smoothed_camera2world/id00803_jZBNgt4kafg_00063.json", "r") as f:
                #     cams = json.load(f)
                #     gen_cam = torch.tensor([cam[1] for cam in cams], device=self.device)[:num_frames].repeat(120//16+1, 1)
                #     gen_cam = gen_cam[:render_frames]

                ws = generator.mapping(z, canonical_c)

                outputs = dict()
                canonical_video = []
                sigmas = []
                for i in range(render_frames // batch_gen):
                    bc = c[i*batch_gen:(i+1)*batch_gen]
                    bTs = Ts[i*batch_gen:(i+1)*batch_gen]
                    bz_motion = z_motion[i*batch_gen:(i+1)*batch_gen]
                    bws = ws[i*batch_gen:(i+1)*batch_gen]
                    output = generator.synthesis(bws, bz_motion, bTs, bc, noise_mode='const', return_depth=True)
                    bcanonical_c = canonical_c[i*batch_gen:(i+1)*batch_gen]
                    canonical_outputs = generator.synthesis(bws, bz_motion, bTs, bcanonical_c, noise_mode='const')
                    canonical_video.append(canonical_outputs["image"])

                    for k in output:
                        if k not in outputs:
                            outputs[k] = [output[k]]
                        else:
                            outputs[k].append(output[k])
                    output = generator.sample_mixed(sxyz, torch.zeros_like(sxyz), bws, bz_motion, bTs)
                    sigma = output["sigma"].detach().cpu().reshape(batch_gen, neural_rendering_resolution, neural_rendering_resolution, num_steps).numpy()
                    sigmas.append(sigma)

                model_output = { k:torch.cat(v, dim=0).cpu() for k, v in outputs.items() }
                model_output = { k:v.cpu() for k, v in model_output.items() }
                video = model_output["image"].clamp(-1, 1).add_(1).div_(2).mul_(255).permute(0,2,3,1).type(torch.uint8)
                depth_video = depth_render(model_output["depth"], resize=video.shape[1])
                
                canonical_video = torch.cat(canonical_video, dim=0).cpu()
                canonical_video = canonical_video.clamp(-1, 1).add_(1).div_(2).mul_(255).permute(0,2,3,1).type(torch.uint8)
                canonical_shape = sigma_render(np.concatenate(sigmas, axis=0), resize=video.shape[1])
                
                render_save_dir = "{}/render".format(get_env(key="save_dir"))
                if not PathManager.exists(render_save_dir):
                    PathManager.mkdirs(render_save_dir)

                rgb_save_path = "{}/rgb_{:04d}.mp4".format(render_save_dir, batch_idx)
                N, H, W, C = video.shape
                coupled_video = torch.cat([
                    canonical_video.unsqueeze(2),
                    canonical_shape.unsqueeze(2),
                    video.unsqueeze(2),
                    depth_video.unsqueeze(2)
                ], dim=2).reshape(N, H, 4*W, C)
                imageio.mimwrite(rgb_save_path, coupled_video, fps=25)

            logger.info(f"Finished rendering")
            self.model.train()

    def generating_loop(self, dataset_type: str) -> None:
        num_videos = 5000 // get_world_size()
        num_frames = 16
        render_frames = 16
        batch_gen = 1 # int(render_frames / 20)
        rank = get_rank()
        configuration = registry.get("configuration", no_warning=True)
        config = configuration.get_config()
        assert render_frames%batch_gen == 0

        with torch.no_grad():
            self.model.eval()
            model = self.model.module if is_dist_initialized() else self.model
            generator = model.generator_ema if self.training_config.use_ema else model.generator
            dataset = getattr(getattr(self, f"{dataset_type}_loader"), "dataset", None)
            assert dataset is not None
            dataset.bi_frame = False
            logger.info(f"Starting generating")

            rand_seed = torch.Generator(device=self.device)
            rand_seed.manual_seed(10000000)
            Ts = torch.linspace(0, 1., steps=num_frames).view(num_frames, 1, 1).unsqueeze(0)
            Ts = Ts.repeat(1, 1, 1, 1).view(-1, 1, 1, 1).to(self.device)

            z = torch.randn([1, 512], device=self.device, generator=rand_seed)
            z = z.unsqueeze(1).repeat(1, render_frames, 1).reshape(-1, z.shape[-1])
            z_motion = torch.randn([1, 512], device=self.device, generator=rand_seed)
            z_motion = z_motion.unsqueeze(1).repeat(1, render_frames, 1).reshape(-1, z_motion.shape[-1])
            gen_c = torch.tensor(CANONICAL_CAMERA, device=self.device).unsqueeze(0).repeat(render_frames, 1).reshape(-1, 25)

            for batch_idx in tqdm(range(num_videos)):
                rand_seed = torch.Generator(device=self.device)
                global_index = (rank * num_videos) + batch_idx
                rand_seed.manual_seed(global_index + 80000000000)
                z = torch.randn([1, 512], device=self.device, generator=rand_seed)
                z = z.unsqueeze(1).repeat(1, render_frames, 1).reshape(-1, z.shape[-1])
                z_motion = torch.randn([1, 512], device=self.device, generator=rand_seed)
                z_motion = z_motion.unsqueeze(1).repeat(1, render_frames, 1).reshape(-1, z_motion.shape[-1])
                gen_c = dataset.get_label(torch.randint(0, len(dataset), [1]).item()).to(self.device)

                # use frame 0 for ws
                gen_c_0 = gen_c[0:1].repeat(gen_c.shape[0], 1)
                ws = generator.mapping(z, gen_c_0)

                outputs = dict()
                for i in range(render_frames // batch_gen):
                    bc = gen_c[i*batch_gen:(i+1)*batch_gen]
                    bTs = Ts[i*batch_gen:(i+1)*batch_gen]
                    bz_motion = z_motion[i*batch_gen:(i+1)*batch_gen]
                    bws = ws[i*batch_gen:(i+1)*batch_gen]
                    output = generator.synthesis(bws, bz_motion, bTs, bc, neural_rendering_resolution=config.training.rendering_resolution, noise_mode='const', return_depth=True)

                    for k in output:
                        if k not in outputs:
                            outputs[k] = [output[k]]
                        else:
                            outputs[k].append(output[k])

                model_output = { k:torch.cat(v, dim=0).cpu() for k, v in outputs.items() }
                model_output = { k:v.cpu() for k, v in model_output.items() }
                video = model_output["image"].clamp(-1, 1).add_(1).div_(2).mul_(255).permute(0, 2, 3, 1).type(torch.uint8)
                
                render_save_dir = "{}/videos".format(get_env(key="save_dir"))
                if not PathManager.exists(render_save_dir):
                    PathManager.mkdirs(render_save_dir)
                rgb_save_path = "{}/rgb_{:04d}.mp4".format(render_save_dir, global_index)
                imageio.mimwrite(rgb_save_path, video, fps=25)

            synchronize()
            logger.info(f"Finished generating")
            self.model.train()
            dataset.bi_frame = True
    
    def generating_multiview_loop(self, dataset_type: str) -> None:
        num_videos = 1000
        batch_gen = 2
        num_frames = 16
        yaw = 30
        pitch = 0
        chunk = 1
        count = 0

        self.model.eval()
        generator = self.model.generator_ema if self.training_config.use_ema else self.model.generator
        dataset = getattr(getattr(self, f"{dataset_type}_loader"), "dataset", None)
        assert dataset is not None
        dataset.bi_frame = False
        logger.info(f"Starting generating")

        # side view camera
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw/180*np.pi,
                                                np.pi/2 -0.05 + pitch/180*np.pi,
                                                torch.tensor([0, 0, 0.2], device=self.device), radius=2.7, device=self.device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device)
        SIDE_CAMERA = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        img_save_dir = "{}/multi/images".format(get_env(key="save_dir"))
        if not PathManager.exists(img_save_dir):
            PathManager.mkdirs(img_save_dir)

        depth_save_dir = "{}/multi/depth".format(get_env(key="save_dir"))
        if not PathManager.exists(depth_save_dir):
            PathManager.mkdirs(depth_save_dir)

        logger.info("Extracting generated identity...")

        for _ in tqdm(range(num_videos // batch_gen)):
            with torch.no_grad():
                Ts = get_t(batch_gen, device=self.device, test=True)
                rand_indices = torch.randint(0, Ts.shape[1], [batch_gen, 2], device=self.device).reshape(batch_gen, 2)
                Ts = torch.gather(Ts, 1, rand_indices)
                Ts = Ts.repeat(1, 2)
                num_frames = Ts.shape[1]
                Ts = Ts.reshape(-1, 1)
                z = get_z(batch_gen, device=self.device)
                z = z.unsqueeze(1).repeat(1, num_frames, 1).reshape(-1, z.shape[-1])
                gen_c0 = torch.tensor(CANONICAL_CAMERA, device=self.device).repeat(batch_gen, num_frames, 1)
                gen_ch0 = gen_c0.clone()
                gen_ch0[:, 1::2] = SIDE_CAMERA
                gen_ch1 = gen_c0.clone()
                gen_ch1[:, 0::2] = SIDE_CAMERA
                gen_c = torch.cat([gen_ch0[:, :num_frames//2], gen_ch1[:, num_frames//2:]], dim=1)
                gen_c = gen_c.reshape(-1, gen_c.shape[-1])
                gen_c0 = gen_c0.reshape(-1, gen_c0.shape[-1])
                z_motion = get_z_motion(batch_gen, device=self.device)
                z_motion = z_motion.unsqueeze(1).repeat(1, num_frames, 1).reshape(-1, z_motion.shape[-1])
                imgs = []
                depths = []
                xyzs = []
                weight_totals = []
                for i in range(len(z)//chunk):
                    outputs = generator(z[i*chunk:(i+1)*chunk], z_motion[i*chunk:(i+1)*chunk], Ts[i*chunk:(i+1)*chunk], gen_c[i*chunk:(i+1)*chunk], 
                                        c0=gen_c0[i*chunk:(i+1)*chunk], neural_rendering_resolution=self.training_config.rendering_resolution, return_depth=True, noise_mode='const')
                    imgs.append(outputs["image"].cpu())
                    xyzs.append(outputs["depth"].cpu())
                    depths.append(outputs["image_depth"].cpu())
                    weight_totals.append(outputs["weight_total"].cpu())
                image = torch.cat(imgs, dim=0).clamp(-1, 1).add_(1).div_(2).mul_(255).permute(0, 2, 3, 1).type(torch.uint8)
                image = rearrange(image, "(b t) h w c -> b t h w c", b=batch_gen)
                depths = torch.cat(depths, dim=0)
                depths = rearrange(depths, "(b t) h w c -> b t h w c", b=batch_gen)
                xyzs = torch.cat(xyzs, dim=0)
                xyzs = rearrange(xyzs, "(b t) h w c -> b t h w c", b=batch_gen)
                weight_totals = torch.cat(weight_totals, dim=0)
                weight_totals = rearrange(weight_totals, "(b t) h w c -> b t h w c", b=batch_gen)
                for ii in range(batch_gen):
                    for idx, f in enumerate(image[ii]):
                        cv2.imwrite(f"{img_save_dir}/{(count+ii):04d}_{idx:02d}.png", (f.cpu().numpy())[:,:,::-1])
                    torch.save({"xyz": xyzs[ii], "depth": depths[ii], "weights_total": weight_totals[ii]}, f"{depth_save_dir}/{(count+ii):04d}.pt")

            count += batch_gen
            torch.cuda.empty_cache()
            del outputs

        logger.info(f"Finished generating")
        self.model.train()
        dataset.bi_frame = True


    def _can_use_tqdm(self, dataloader: torch.utils.data.DataLoader):
        """
        Checks whether tqdm can be gracefully used with a dataloader
        1) should have `__len__` property defined
        2) calling len(x) should not throw errors.
        """
        use_tqdm = hasattr(dataloader, "__len__")

        try:
            _ = len(dataloader)
        except (AttributeError, TypeError, NotImplementedError):
            use_tqdm = False
        return use_tqdm


def validate_batch_sizes(my_batch_size: int) -> bool:
    """
    Validates all workers got the same batch size.
    """
    batch_size_tensor = torch.IntTensor([my_batch_size])
    if torch.cuda.is_available():
        batch_size_tensor = batch_size_tensor.cuda()
    all_batch_sizes = gather_tensor(batch_size_tensor)
    for j, oth_batch_size in enumerate(all_batch_sizes.data):
        if oth_batch_size != my_batch_size:
            logger.error(f"Node {j} batch {oth_batch_size} != {my_batch_size}")
            return False
    return True