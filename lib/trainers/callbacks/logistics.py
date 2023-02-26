# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import logging
from einops import rearrange
from lib.trainers.callbacks.base import Callback
from lib.utils.configuration import get_env
from lib.utils.logger import (
    TensorboardLogger,
    calculate_time_left,
    summarize_report, 
    setup_output_folder
)
from lib.utils.general import Timer
from torchvision.utils import save_image
from lib.utils.distributed import is_master


logger = logging.getLogger(__name__)


class LogisticsCallback(Callback):
    """Callback for handling train/validation logistics, report summarization,
    logging etc.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)
        self.total_timer = Timer()
        self.log_interval = self.training_config.log_interval
        self.visualize_interval = self.training_config.visualize_interval
        self.evaluation_interval = self.training_config.evaluation_interval
        self.checkpoint_interval = self.training_config.checkpoint_interval
        self.visualization_online = self.training_config.visualization_online
        
        # Total iterations for snapshot
        # len would be number of batches per GPU == max updates
        if self.trainer.val_loader is not None:
            self.snapshot_iterations = len(self.trainer.val_loader)
        else:
            self.snapshot_iterations = 0

        self.tb_writer = None

        if self.training_config.tensorboard:
            log_dir = setup_output_folder(folder_only=True)
            env_tb_logdir = get_env(key="tensorboard_logdir")
            if env_tb_logdir:
                log_dir = env_tb_logdir

            self.tb_writer = TensorboardLogger(log_dir, self.trainer.current_iteration)

    def on_train_start(self):
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

    def on_batch_end(self, **kwargs):
        self.on_img_batch_end(**kwargs)

    def on_img_batch_end(self, **kwargs):
        report = kwargs["report"]
        meter = kwargs["meter"]
        num_images = report["batch_size"]

        discriminator_acc = dict()
        generated_img_scores = report["generated_img_scores"].sigmoid() < 0.5
        generated_img_pred = generated_img_scores.sum()
        real_img_scores = report["real_img_scores"].sigmoid() > 0.5
        real_img_pred = real_img_scores.sum()
        discriminator_acc["acc/img"] = (generated_img_pred+real_img_pred)/(2*num_images)
        report.discriminator_acc = discriminator_acc
        meter.update_from_report(report)

        if self.tb_writer:
            scalar_dict = meter.get_scalar_dict()
            self.tb_writer.add_scalars(scalar_dict, self.trainer.current_iteration)
            if self.trainer.current_iteration % self.visualize_interval == 0:
                fake_images = []
                fake_render = []
                fake_depth = []
                real_images = []

                generated_imgs = report["generated_imgs"].clamp(-1, 1).sub_(-1).div_(2).cpu()
                fake_images.extend([generated_imgs])

                render_imgs = report["render_imgs"].clamp(-1, 1).sub_(-1).div_(2).cpu()
                fake_render.extend([render_imgs])

                depth = report["depth"]
                generated_depth = ((depth-depth.mean())/(depth.min()-depth.max())).clamp(-1, 1).sub_(-1).div_(2).cpu()
                fake_depth.extend([generated_depth])

                real_frames = report["frames"].clamp(-1, 1).sub_(-1).div_(2).cpu()
                real_images.append(real_frames)

                fake_images = torch.cat(fake_images, dim=-1)[:16]
                fake_render = torch.cat(fake_render, dim=-1)[:16]
                fake_depth = torch.cat(fake_depth, dim=-1)[:16]
                real_images = torch.cat(real_images, dim=-1)[:16]
                self.tb_writer.add_images(fake_images, self.trainer.current_iteration, tag=f"fake")
                self.tb_writer.add_images(fake_render, self.trainer.current_iteration, tag=f"fake_render")
                self.tb_writer.add_images(fake_depth, self.trainer.current_iteration, tag=f"fake_depth")
                self.tb_writer.add_images(real_images, self.trainer.current_iteration, tag=f"real")

                if self.visualization_online and is_master():
                    save_dir = get_env(key="save_dir")
                    online_images = torch.cat([fake_images, real_images], dim=-2)
                    save_image(online_images, f"{save_dir}/online_image.jpg", value_range=(-1, 1))

            self.tb_writer.flush()

    def on_video_batch_end(self, **kwargs):
        report = kwargs["report"]
        meter = kwargs["meter"]
        num_images = report["batch_size"]

        discriminator_acc = dict()
        generated_vid_scores = report["generated_vid_scores"].sigmoid() < 0.5
        generated_vid_pred = generated_vid_scores.sum()
        real_vid_scores = report["real_vid_scores"].sigmoid() > 0.5
        real_vid_pred = real_vid_scores.sum()
        discriminator_acc["acc/vid"] = (generated_vid_pred+real_vid_pred)/(2*num_images)
        generated_img_scores = report["generated_img_scores"].sigmoid() < 0.5
        generated_img_pred = generated_img_scores.sum()
        real_img_scores = report["real_img_scores"].sigmoid() > 0.5
        real_img_pred = real_img_scores.sum()
        discriminator_acc["acc/img"] = (generated_img_pred+real_img_pred)/(4*num_images)
        report.discriminator_acc = discriminator_acc
        meter.update_from_report(report)

        if self.tb_writer:
            scalar_dict = meter.get_scalar_dict()
            self.tb_writer.add_scalars(scalar_dict, self.trainer.current_iteration)
            if self.trainer.current_iteration % self.visualize_interval == 0:
                fake_images = []
                fake_render = []
                fake_depth = []
                real_images = []

                generated_imgs = report["generated_imgs"].clamp(-1, 1).sub_(-1).div_(2).cpu()
                fake_images.extend([generated_imgs])

                render_imgs = report["render_imgs"].clamp(-1, 1).sub_(-1).div_(2).cpu()
                fake_render.extend([render_imgs])

                depth = report["depth"]
                generated_depth = ((depth-depth.mean())/(depth.max()-depth.min())).clamp(-1, 1).sub_(-1).div_(2).cpu()
                fake_depth.extend([generated_depth])

                real_videos = report["frames"].clamp(-1, 1).sub_(-1).div_(2).cpu()
                real_videos = rearrange(real_videos, 'b t c h w -> (b t) c h w', t=2)
                real_images.append(real_videos)

                fake_images = torch.cat(fake_images, dim=-1)[:16]
                fake_render = torch.cat(fake_render, dim=-1)[:16]
                fake_depth = torch.cat(fake_depth, dim=-1)[:16]
                real_images = torch.cat(real_images, dim=-1)[:16]
                self.tb_writer.add_images(fake_images, self.trainer.current_iteration, tag=f"fake")
                self.tb_writer.add_images(fake_render, self.trainer.current_iteration, tag=f"fake_render")
                self.tb_writer.add_images(fake_depth, self.trainer.current_iteration, tag=f"fake_depth")
                self.tb_writer.add_images(real_images, self.trainer.current_iteration, tag=f"real")

                if self.visualization_online and is_master():
                    save_dir = get_env(key="save_dir")
                    online_images = torch.cat([fake_images, real_images], dim=-2)
                    save_image(online_images, f"{save_dir}/online_image.jpg", value_range=(-1, 1))

            self.tb_writer.flush()

    def on_update_end(self, **kwargs):
        if not kwargs["should_log"]:
            return
        extra = {}
        if "cuda" in str(self.trainer.device):
            extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
            extra["max mem"] //= 1024

        if self.training_config.experiment_name:
            extra["experiment"] = self.training_config.experiment_name

        max_updates = getattr(self.trainer, "max_updates", None)
        num_updates = getattr(self.trainer, "num_updates", None)
        extra.update(
            {
                "epoch": self.trainer.current_epoch,
                "num_updates": num_updates,
                "iterations": self.trainer.current_iteration,
                "max_updates": max_updates,
                "lr_generator": "{:.5f}".format(
                    self.trainer.optimizer["generator"].param_groups[0]["lr"]
                ).rstrip("0"),
                "ups": "{:.2f}".format(
                    self.log_interval / self.train_timer.unix_time_since_start()
                ),
                "time": self.train_timer.get_time_since_start(),
                "time_since_start": self.total_timer.get_time_since_start(),
                "eta": calculate_time_left(
                    max_updates=max_updates,
                    num_updates=num_updates,
                    timer=self.train_timer,
                    num_snapshot_iterations=self.snapshot_iterations,
                    log_interval=self.log_interval,
                    eval_interval=self.evaluation_interval,
                ),
            }
        )
        self.train_timer.reset()
        summarize_report(
            current_iteration=self.trainer.current_iteration,
            num_updates=num_updates,
            max_updates=max_updates,
            meter=kwargs["meter"],
            extra=extra,
            tb_writer=self.tb_writer,
        )

    def on_validation_start(self, **kwargs):
        self.snapshot_timer.reset()

    def on_validation_end(self, **kwargs):
        max_updates = getattr(self.trainer, "max_updates", None)
        num_updates = getattr(self.trainer, "num_updates", None)
        extra = {
            "num_updates": num_updates,
            "epoch": self.trainer.current_epoch,
            "iterations": self.trainer.current_iteration,
            "max_updates": max_updates,
            "val_time": self.snapshot_timer.get_time_since_start(),
        }
        extra.update(self.trainer.early_stop_callback.early_stopping.get_info())
        self.train_timer.reset()
        summarize_report(
            current_iteration=self.trainer.current_iteration,
            num_updates=num_updates,
            max_updates=max_updates,
            meter=kwargs["meter"],
            extra=extra,
            tb_writer=self.tb_writer,
        )

    def on_test_end(self, **kwargs):
        prefix = "{}: full {}".format(
            kwargs["report"].dataset_name, kwargs["report"].dataset_type
        )
        summarize_report(
            current_iteration=self.trainer.current_iteration,
            num_updates=getattr(self.trainer, "num_updates", None),
            max_updates=getattr(self.trainer, "max_updates", None),
            meter=kwargs["meter"],
            should_print=prefix,
            tb_writer=self.tb_writer,
        )
        logger.info(f"Finished run in {self.total_timer.get_time_since_start()}")
