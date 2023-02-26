# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import gc
import logging
from abc import ABC
from typing import Any, Dict

import torch
from torch import Tensor
from lib.models.base_model import BaseModel
from lib.utils.configuration import get_global_config
from lib.utils.distributed import is_dist_initialized
from lib.common.report import Report
from lib.common.meter import Meter
from lib.common.registry import registry
from lib.utils.general import (clip_gradients, 
                               get_max_updates, 
                               move_to_device, 
                               extract_loss, 
                               is_fixed_generator, 
                               set_gradient_requirement)


logger = logging.getLogger(__name__)


class TrainerTrainingLoopMixin(ABC):
    current_epoch: int = 0
    current_iteration: int = 0
    num_updates: int = 0
    num_samples: int = 0
    meter: Meter = Meter()

    def training_loop(self) -> None:
        self.max_updates = self._calculate_max_updates()
        torch.autograd.set_detect_anomaly(self.training_config.detect_anomaly)
        self.start_updates = self.num_updates
        logger.info("Starting training...")
        self.model.train()
        self.run_training_epoch()
        self.after_training_loop()

    def after_training_loop(self) -> None:
        logger.info("Stepping into final validation check")
        # Only do when run_type has train as it shouldn't happen on validation and
        # inference runs. Inference will take care of this anyways. Also, don't run
        # if current iteration is divisble by snapshot interval as it will just
        # be a repeat
        if (
            "train" in self.run_type
            and "val" in self.run_type
            and self.num_updates % self.training_config.evaluation_interval != 0
        ):
            # Create a new meter for this case
            report, meter = self.evaluation_loop("val")

            # Validation end callbacks
            self.on_validation_end(report=report, meter=meter)

    def run_training_epoch(self) -> None:
        should_break = False
        while self.num_updates < self.max_updates and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            # if is_dist_initialized():
            #     self.train_loader.sampler.set_epoch(self.current_epoch)

            # For iterable datasets we cannot determine length of dataset properly.
            # For those cases we set num_remaining_batches to be the (number of
            # updates remaining x update_frequency)
            num_remaining_batches = (
                (
                    (self.max_updates - self.num_updates)
                    * self.training_config.update_frequency
                )
                if isinstance(
                    self.train_loader.dataset, torch.utils.data.IterableDataset
                )
                else len(self.train_loader)
            )

            self.train_discriminator = True
            should_start_update = True
            for idx, batch in enumerate(self.train_loader):
                if should_start_update:
                    combined_report = None
                    self._start_update()
                    num_batches_for_this_update = min(
                        self.training_config.update_frequency, num_remaining_batches
                    )
                    should_start_update = False

                self.current_iteration += 1
                # batch execution starts here
                self.on_batch_start()
                self.profile("Batch load time")

                run_training_batch = getattr(self, "run_{}_training_batch".format(get_global_config("model")))
                report = run_training_batch(batch, num_batches_for_this_update)
                report = report.detach()

                # accumulate necessary params (including loss) for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields_and_loss(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                # batch execution ends here
                self.on_batch_end(report=combined_report, meter=self.meter)

                # check if an update has finished or if it is the last, if no continue
                if (
                    (idx + 1) % self.training_config.update_frequency
                    and num_remaining_batches != num_batches_for_this_update
                ):
                    continue

                should_start_update = True

                should_log = False
                if self.num_updates % self.logistics_callback.log_interval == 0:
                    should_log = True
                    # Calculate metrics every log interval for debugging
                    if self.training_config.evaluate_metrics:
                        combined_report.metrics = self.metrics(
                            combined_report, combined_report
                        )
                    self.meter.update_from_report(combined_report)

                self.on_update_end(
                    report=combined_report, meter=self.meter, should_log=should_log
                )

                num_remaining_batches -= num_batches_for_this_update

                # Check if training should be stopped
                should_break = False

                if (
                    self.num_updates % self.training_config.evaluation_interval == 0
                    and self.num_updates > self.training_config.warmup_iterations
                ):
                    # Validation begin callbacks
                    self.on_validation_start()

                    logger.info("Evaluation time. Running on full validation set...")
                    # Validation and Early stopping
                    # Create a new meter for this case
                    report, meter = self.evaluation_loop("val")

                    # Validation end callbacks
                    stop = self.early_stop_callback.on_validation_end(
                        report=report, meter=meter
                    )
                    self.on_validation_end(report=report, meter=meter)

                    gc.collect()

                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()

                    if stop is True:
                        logger.info("Early stopping activated")
                        should_break = True

                if self.num_updates >= self.max_updates:
                    should_break = True

                if should_break:
                    break

    def run_pv3d_training_batch(self, batch: Dict[str, Tensor], loss_divisor: int) -> None:
        batch_size = batch["frames"].shape[0]
        dataset = self.train_loader.dataset
        gen_cams = [dataset.get_label(torch.randint(0, len(dataset), [1]).item()).unsqueeze(0) for _ in range(4*batch_size)]
        gen_cams = torch.cat(gen_cams, dim=0).pin_memory()
        gen_cams = gen_cams.split(batch_size)
        combined_report = Report(batch)
        combined_report.losses = {}
        generators = ["generator"]
        discriminators = ["img_discriminator", "vid_discriminator"]
        # train generator
        # Gmain
        self.train_discriminator = False
        batch["phase"] = "Gmain"
        batch["gen_cam"] = gen_cams[0]
        set_gradient_requirement(self.model, 
                                 self.train_discriminator,
                                 discriminators=discriminators)
        report = self._forward_eg3d(batch)
        loss = extract_loss(report, loss_divisor)
        self._eg3d_backward(loss)
        self._step(groups=generators)
        combined_report.generated_imgs = report["generated_imgs"]
        combined_report.render_imgs = report["render_imgs"]
        combined_report.depth = report["depth"]

        # Greg
        if self.num_updates % self.config.training.gen_reg_interval == 0:
            batch["phase"] = "Greg"
            batch["gen_cam"] = gen_cams[1]
            report = self._forward_eg3d(batch)
            loss = extract_loss(report, loss_divisor)
            self._eg3d_backward(loss)
            self._step(groups=generators)

        # train discriminator
        # Dmain
        self.train_discriminator = True
        batch["phase"] = "Dmain"
        batch["gen_cam"] = gen_cams[2]
        set_gradient_requirement(self.model, 
                                 self.train_discriminator,
                                 discriminators=discriminators)
        report = self._forward_eg3d(batch)
        loss = extract_loss(report, loss_divisor)
        self._eg3d_backward(loss)
        self._step(groups=discriminators)
        combined_report.real_img_scores = report["real_img_scores"]
        combined_report.generated_img_scores = report["generated_img_scores"]

        # Dreg
        if self.num_updates % self.config.training.disc_reg_interval == 0:
            batch["phase"] = "Dreg"
            batch["gen_cam"] = gen_cams[3]
            report = self._forward_eg3d(batch)
            loss = extract_loss(report, loss_divisor)
            self._eg3d_backward(loss)
            self._step(groups=discriminators)

        self._update_ema(self.config.training.batch_size)
        self._finish_eg3d_update(self.config.training.batch_size) # multi-GPU
        combined_report.losses = registry.get("losses")
        return combined_report

    def run_eg3d_training_batch(self, batch: Dict[str, Tensor], loss_divisor: int) -> None:
        batch_size = batch["frames"].shape[0]
        dataset = self.train_loader.dataset
        gen_cams = [dataset.get_label(torch.randint(0, len(dataset), [1]).item()).unsqueeze(0) for _ in range(4*batch_size)]
        gen_cams = torch.cat(gen_cams, dim=0).pin_memory()
        gen_cams = gen_cams.split(batch_size)
        combined_report = Report(batch)
        combined_report.losses = {}
        generators = ["generator"]
        discriminators = ["img_discriminator"]
        # train generator
        # Gmain
        self.train_discriminator = False
        batch["phase"] = "Gmain"
        batch["gen_cam"] = gen_cams[0]
        set_gradient_requirement(self.model, 
                                 self.train_discriminator,
                                 discriminators=discriminators)
        report = self._forward_eg3d(batch)
        loss = extract_loss(report, loss_divisor)
        self._eg3d_backward(loss)
        self._step(groups=generators)
        combined_report.generated_imgs = report["generated_imgs"]
        combined_report.render_imgs = report["render_imgs"]
        combined_report.depth = report["depth"]

        # Greg
        if self.num_updates % self.config.training.gen_reg_interval == 0:
            batch["phase"] = "Greg"
            batch["gen_cam"] = gen_cams[1]
            report = self._forward_eg3d(batch)
            loss = extract_loss(report, loss_divisor)
            self._eg3d_backward(loss)
            self._step(groups=generators)

        # train discriminator
        # Dmain
        self.train_discriminator = True
        batch["phase"] = "Dmain"
        batch["gen_cam"] = gen_cams[2]
        set_gradient_requirement(self.model, 
                                 self.train_discriminator,
                                 discriminators=discriminators)
        report = self._forward_eg3d(batch)
        loss = extract_loss(report, loss_divisor)
        self._eg3d_backward(loss)
        self._step(groups=discriminators)
        combined_report.real_img_scores = report["real_img_scores"]
        combined_report.generated_img_scores = report["generated_img_scores"]

        # Dreg
        if self.num_updates % self.config.training.disc_reg_interval == 0:
            batch["phase"] = "Dreg"
            batch["gen_cam"] = gen_cams[3]
            report = self._forward_eg3d(batch)
            loss = extract_loss(report, loss_divisor)
            self._eg3d_backward(loss)
            self._step(groups=discriminators)

        self._update_ema(self.config.training.batch_size)
        self._finish_eg3d_update(self.config.training.batch_size) # multi-GPU
        combined_report.losses = registry.get("losses")
        return combined_report

    def _forward_eg3d(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        # Move the sample list to device if it isn't as of now.
        prepared_batch = move_to_device(batch, self.device)
        prepared_batch["num_updates"] = self.num_updates
        prepared_batch["num_samples"] = self.num_samples
        # prepared_batch["return_depth"] = self.current_iteration % self.training_config.visualize_interval == 0
        
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point
        with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
            model_output = self.model(prepared_batch)
            report = Report(prepared_batch, model_output)
        self.profile("Forward time")
        return report
    
    def _update_ema(self, batch_size):
        model = self.model.module if is_dist_initialized() else self.model

        ema_nimg = self.training_config.batch_size * 10 / 32 * 1000
        if self.training_config.ema_rampup is not None:
            ema_nimg = min(ema_nimg, self.num_samples * self.training_config.ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))

        for p_ema, p in zip(model.generator_ema.parameters(), model.generator.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(model.generator_ema.buffers(), model.generator.buffers()):
            b_ema.copy_(b)
        model.generator_ema.neural_rendering_resolution = model.generator.neural_rendering_resolution
        model.generator_ema.rendering_kwargs = model.generator.rendering_kwargs.copy()

    def _start_update(self):
        logger.debug(self.num_updates + 1)
        self.on_update_start()
        for group in self.optimizer:
            self.optimizer[group].zero_grad(set_to_none=True)

    def _backward(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.profile("Backward time")
    
    def _eg3d_backward(self, loss: Tensor) -> None:
        loss.backward()
        self.profile("Backward time")
    
    def _step(self, groups):
        clip_gradients(
            self.model,
            self.optimizer,
            self.num_updates,
            self.logistics_callback.tb_writer,
            self.config,
            scale=self.scaler.get_scale(),
        )
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        for group in groups:
            self.optimizer[group].step()
            self.optimizer[group].zero_grad(set_to_none=True)
        self.profile("Finished step")
    
    def _finish_eg3d_update(self, batch_size):
        self.num_updates += 1
        self.num_samples += batch_size
        self.profile("Finished update")

    def _calculate_max_updates(self):
        config_max_updates = self.training_config.max_updates
        config_max_epochs = self.training_config.max_epochs
        max_updates, _ = get_max_updates(
            config_max_updates,
            config_max_epochs,
            self.train_loader,
            self.training_config.update_frequency,
        )

        return max_updates
