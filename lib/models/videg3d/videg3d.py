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


import torch
import copy
import logging
import numpy as np
from einops import rearrange
from lib.models.base_model import BaseModel
from lib.common.registry import registry
from lib.utils.checkpoint import load_pretrained_model
from lib.utils.general import get_batch_size, get_z, get_t, get_z_motion
from lib.models.videg3d.torch_utils.ops import conv2d_gradfix, upfirdn2d
from lib.models.videg3d.networks.triplane import TriPlaneGenerator
from lib.models.videg3d.networks.dual_discriminator import DualDiscriminator, VideoDiscriminator
from lib.models.videg3d.networks.volumetric_rendering import *
from lib.models.eg3d.networks.dual_discriminator import filtered_resizing


logger = logging.getLogger(__name__)


@registry.register_model("pv3d")
class PV3D(BaseModel):

    def __init__(self, config, *args, **kwds):
        super().__init__(config, *args, **kwds)
    
    def build(self):
        self.generator = TriPlaneGenerator(**self.config.get("generator"))
        self.img_discriminator = DualDiscriminator(**self.config.get("img_discriminator"))
        self.vid_discriminator = VideoDiscriminator(**self.config.get("vid_discriminator"))
        
        if self.config.use_ema:
            self.generator_ema = copy.deepcopy(self.generator).eval()
            for p in self.generator_ema.parameters():
                p.requires_grad = False

        self.style_mixing_prob = self.config.get("style_mixing_prob", 0.0)
        self.pl_weight = self.config.get("pl_weight", 0.0)
        self.pl_batch_shrink = self.config.get("pl_batch_shrink", 2)
        self.pl_decay = self.config.get("pl_decay", 0.01)
        self.pl_no_weight_grad = self.config.get("pl_no_weight_grad", False)
        self.blur_fade_kimg = self.config.get("blur_fade_kimg", 200) * get_batch_size()/ 32
        self.blur_init_sigma = self.config.get("blur_init_sigma", 10)
        self.gpc_reg_fade_kimg = self.config.get("gpc_reg_fade_kimg", 1000)
        self.gpc_reg_prob = self.config.get("gpc_reg_prob", 0.5)
        self.r1_gamma_init = self.config.get("r1_gamma_init", 0)
        self.r1_gamma_fade_kimg = self.config.get("r1_gamma_fade_kimg", 0)
        self.dual_discrimination = self.config.get("dual_discrimination", True)
        self.neural_rendering_resolution_final = self.config.get("neural_rendering_resolution_final", None)
        self.neural_rendering_resolution_fade_kimg = self.config.get("neural_rendering_resolution_fade_kimg", 1000)
        self.neural_rendering_resolution_initial = self.config.get("neural_rendering_resolution_initial", 64)
        self.filter_mode = self.config.get("filter_mode", 'antialiased')
        self.timesteps = self.config.get("timesteps", 2)
        self.resample_filter = None
        self.augment_pipe = None
        self.blur_raw_target = True
        self.register_buffer("pl_mean", torch.zeros([]))

        ckpt_path = self.config.pretrained_ckpt_path
        if ckpt_path:
            logger.info(f"initializing pretrained model from {ckpt_path}")
            checkpoint = load_pretrained_model(ckpt_path, init = True)["G_ema"]
            checkpoint.pop("dataset_label_std")
            self.generator.load_state_dict(checkpoint, strict=True)
            self.generator_ema.load_state_dict(checkpoint, strict=True)
    
    def train(self, mode=True):
        super().train(mode)
        if self.config.use_ema:
            self.generator_ema.eval()
    
    def forward(self, sample, return_xyz=False):
        sample["blur_sigma"] = max(1 - sample["num_samples"] / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        alpha = min(sample["num_samples"] / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        sample["c_swap_prob"] = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(sample["num_samples"] / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial
        sample["neural_rendering_resolution"] = neural_rendering_resolution

        if self.resample_filter is None:
            self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1], device=sample["frames"].device)
        sample["frames"] = rearrange(sample["frames"], "b t c h w -> (b t) c h w", t=sample["frames"].shape[1])
        real_img_raw = filtered_resizing(sample["frames"], size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
        if self.blur_raw_target:
            blur_size = np.floor(sample["blur_sigma"] * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(sample["blur_sigma"]).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())
        sample["frames_raw"] = real_img_raw

        forward_fn = getattr(self, "_forward_"+sample["phase"])
        return forward_fn(sample)
    
    def _run_generator(self, z, z_motion, Ts, c, c_swap_prob, neural_rendering_resolution, update_emas=False):
        if c_swap_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where((torch.rand((c.shape[0], 1, 1), device=c.device) < c_swap_prob).repeat(1, 2, 1), c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)
        c = c.reshape(-1, c.shape[-1])
        c_gen_conditioning = c_gen_conditioning.reshape(-1, c.shape[-1])
        # c_gen_conditioning = c_gen_conditioning[:, 0:1].repeat(1, 2, 1).reshape(-1, c.shape[-1])
        ws = self.generator.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.generator.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        output = self.generator.synthesis(ws, z_motion, Ts, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        output["ws"] = ws
        return output

    def _run_discriminator(self, img, Ts, c, return_input=False, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        c = c.reshape(-1, c.shape[-1])
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
            img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits_img = self.img_discriminator(img, c, update_emas=update_emas)
        logits_vid, inp = self.vid_discriminator(img, Ts, c, return_input=return_input, update_emas=update_emas)
        generated_scores = {"logits_img": logits_img, "logits_vid": logits_vid, "input": inp}
        return generated_scores

    def _forward_Gmain(self, sample):
        model_output = dict(phase=sample["phase"])
        # forward generator
        batch_size = sample["frames"].shape[0] // self.timesteps
        gen_z = get_z(batch_size, timesteps=self.timesteps, device=sample["frames"].device)
        gen_z_motion = get_z_motion(batch_size, timesteps=self.timesteps, device=sample["frames"].device)
        gen_t = get_t(batch_size, device=sample["frames"].device)
        gen_c = sample["gen_cam"]
        
        generated = self._run_generator(gen_z, gen_z_motion, gen_t, gen_c, c_swap_prob=sample["c_swap_prob"], neural_rendering_resolution=sample["neural_rendering_resolution"])
        generated_scores = self._run_discriminator(generated, gen_t, gen_c, blur_sigma=sample["blur_sigma"])

        model_output["generated_img_scores"] = generated_scores["logits_img"]
        model_output["generated_vid_scores"] = generated_scores["logits_vid"]
        model_output["generated_imgs"] = generated["image"].clone().detach().cpu()
        model_output["render_imgs"] = generated["image_raw"].clone().detach().cpu()
        model_output["depth"] = generated["image_depth"].clone().detach().cpu() if "image_depth" in generated else None
        return model_output

    def _forward_Greg(self, sample):
        model_output = dict(phase=sample["phase"])
        batch_size = sample["frames"].shape[0] // self.timesteps
        gen_z = get_z(batch_size, timesteps=self.timesteps, device=sample["frames"].device)
        gen_z_motion = get_z_motion(batch_size, timesteps=self.timesteps, device=sample["frames"].device)
        gen_t = get_t(batch_size, device=sample["frames"].device)
        gen_c = sample["gen_cam"]

        # generator gradient penalty
        if sample["c_swap_prob"] is not None:
            c_swapped = torch.roll(gen_c.clone(), 1, 0)
            c_gen_conditioning = torch.where((torch.rand((gen_c.shape[0], 1, 1), device=gen_c.device) < sample["c_swap_prob"]).repeat(1, 2, 1), c_swapped, gen_c)
        else:
            c_gen_conditioning = torch.zeros_like(gen_c)
        gen_c = gen_c.reshape(-1, gen_c.shape[-1])
        c_gen_conditioning = c_gen_conditioning.reshape(-1, gen_c.shape[-1])
        # c_gen_conditioning = c_gen_conditioning[:, 0:1].repeat(1, 2, 1).reshape(-1, gen_c.shape[-1])

        ws = self.generator.mapping(gen_z, c_gen_conditioning, update_emas=False)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(gen_z), gen_c, update_emas=False)[:, cutoff:]
        initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.generator.rendering_kwargs['density_reg_p_dist']
        all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        sigma = self.generator.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, gen_z_motion, gen_t, update_emas=False)['sigma']
        sigma_initial = sigma[:, :sigma.shape[1]//2]
        sigma_perturbed = sigma[:, sigma.shape[1]//2:]
        model_output["sigma_initial"] = sigma_initial
        model_output["sigma_perturbed"] = sigma_perturbed

        return model_output
    
    def _forward_Dmain(self, sample):
        model_output = dict()
        batch_size = sample["frames"].shape[0] // self.timesteps
        gen_z = get_z(batch_size, timesteps=self.timesteps, device=sample["frames"].device)
        gen_z_motion = get_z_motion(batch_size, timesteps=self.timesteps, device=sample["frames"].device)
        gen_t = get_t(batch_size, device=sample["frames"].device)
        gen_c = sample["gen_cam"]
        real_t = sample["real_t"].detach().requires_grad_(False)
        real_c = sample["real_cam"]

        generated = self._run_generator(gen_z, gen_z_motion, gen_t, gen_c, c_swap_prob=sample["c_swap_prob"], neural_rendering_resolution=sample["neural_rendering_resolution"], update_emas=True)
        generated_scores = self._run_discriminator(generated, gen_t, gen_c, blur_sigma=sample["blur_sigma"], update_emas=True)
        model_output["generated_img_scores"] = generated_scores["logits_img"]
        model_output["generated_vid_scores"] = generated_scores["logits_vid"]

        real_imgs = {"image": sample["frames"].detach().requires_grad_(False), "image_raw": sample["frames_raw"].detach().requires_grad_(False)}
        real_scores = self._run_discriminator(real_imgs, real_t, real_c, blur_sigma=sample["blur_sigma"])
        model_output["real_img_scores"] = real_scores["logits_img"]
        model_output["real_vid_scores"] = real_scores["logits_vid"]

        return model_output

    def _forward_Dreg(self, sample):
        model_output = dict()
        real_t = sample["real_t"].detach().requires_grad_(True)
        real_c = sample["real_cam"]
        real_imgs = {"image": sample["frames"].detach().requires_grad_(True), "image_raw": sample["frames_raw"].detach().requires_grad_(True)}
        # gradient penalty
        real_output = self._run_discriminator(real_imgs, real_t, real_c, return_input=True, blur_sigma=sample["blur_sigma"])
        if self.dual_discrimination:
            with conv2d_gradfix.no_weight_gradients():
                r1_grads = torch.autograd.grad(outputs=[real_output["logits_img"].sum()], inputs=[real_imgs["image"], real_imgs["image_raw"]], create_graph=True, only_inputs=True)
                r1_grads_image = r1_grads[0]
                r1_grads_image_raw = r1_grads[1]
            r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
        else:
            with conv2d_gradfix.no_weight_gradients():
                r1_grads = torch.autograd.grad(outputs=[real_output["logits_img"].sum()], inputs=[real_imgs['image']], create_graph=True, only_inputs=True)
                r1_grads_image = r1_grads[0]
            r1_penalty = r1_grads_image.square().sum([1,2,3])

        with conv2d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(outputs=[real_output["logits_vid"].sum()], inputs=[real_output["input"]], create_graph=True, only_inputs=True)
            r1_grads_video = r1_grads[0]
        r1_penalty_vid = r1_grads_video.square().sum([1,2,3])

        model_output["grad_img_penalty"] = r1_penalty
        model_output["grad_vid_penalty"] = r1_penalty_vid
        return model_output
    
    def forward_evaluation(self, sample, return_xyz=False):
        z = sample["z"].repeat(2, 1)
        z_motion = sample["z_motion"].repeat(2, 1)
        Ts = sample["Ts"]
        c = sample["c"]
        with torch.no_grad():
            if self.config.use_ema:
                generated = self.generator_ema(z, z_motion, Ts, c, noise_mode='const')
            else:
                generated = self.generator(z, z_motion, Ts, c, noise_mode='const')
        generated = {k:v[0:1] for k, v in generated.items() if v is not None}
        return generated
