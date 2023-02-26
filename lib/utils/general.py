# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import torch
import time, math, os
import PIL.Image
import logging, warnings
import torch.nn as nn
import numpy as np
from PIL import Image
from bisect import bisect
from typing import Dict, Any
from subprocess import Popen, PIPE
from lib.common.registry import registry
from iopath.common.file_io import PathManager as pm
from lib.utils.distributed import get_rank, is_dist_initialized, get_world_size


logger = logging.getLogger(__name__)


class Timer:
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y/%m/%d %H:%M:%S"
    DEFAULT_TIME_FORMAT = ["%03dms", "%02ds", "%02dm", "%02dh"]

    def __init__(self):
        self.start = time.time() * 1000

    def get_current(self):
        return self.get_time_hhmmss(self.start)

    def reset(self):
        self.start = time.time() * 1000

    def get_time_since_start(self, format=None):
        return self.get_time_hhmmss(self.start, format)

    def unix_time_since_start(self, in_seconds=True):
        gap = time.time() * 1000 - self.start

        if in_seconds:
            gap = gap // 1000

        # Prevent 0 division errors
        if gap == 0:
            gap = 1
        return gap

    def get_time_hhmmss(self, start=None, end=None, gap=None, format=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None and gap is None:

            if format is None:
                format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(format)

        if end is None:
            end = time.time() * 1000
        if gap is None:
            gap = end - start

        s, ms = divmod(gap, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        if format is None:
            format = self.DEFAULT_TIME_FORMAT

        items = [ms, s, m, h]
        assert len(items) == len(format), "Format length should be same as items"

        time_str = ""
        for idx, item in enumerate(items):
            if item != 0:
                time_str = format[idx] % item + " " + time_str

        # Means no more time is left.
        if len(time_str) == 0:
            time_str = "0ms"

        return time_str.strip()


PathManager = pm()


def log_device_names():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        logger.info(f"CUDA Device {get_rank()} is: {device_name}")


def scalarize_dict_values(dict_with_tensors):
    """
    this method returns a new dict where the values of
    `dict_with_tensors` would be a scalar

    Returns:
        Dict: a new dict with scalarized values
    """
    dict_with_scalar_tensors = {}
    for key, val in dict_with_tensors.items():
        if torch.is_tensor(val):
            if val.dim() != 0:
                val = val.mean()
        dict_with_scalar_tensors[key] = val
    return dict_with_scalar_tensors


def print_model_parameters(model, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        logger.info(
            f"Total Parameters: {total_params}. Trained Parameters: {trained_params}"
        )
    return total_params, trained_params


def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )

    if is_parallel and hasattr(model.module, "get_optimizer_parameters"):
        parameters = model.module.get_optimizer_parameters(config)

    if len(parameters) == 0:
        raise ValueError("optimizer got an empty parameter list")

    for group in parameters:
        for p in parameters[group]:
            p["params"] = list(p["params"])

    check_unused_parameters(parameters, model, config)
    params = get_extra_params(config)

    return parameters, params


def check_unused_parameters(parameters, model, config):
    optimizer_param_set = {p for group in parameters for pa in parameters[group] for p in pa["params"]}
    unused_param_names = []
    for n, p in model.named_parameters():
        if p.requires_grad and p not in optimizer_param_set:
            unused_param_names.append(n)
    if len(unused_param_names) > 0:
        logger.info(
            "Model parameters not used by optimizer: {}".format(
                " ".join(unused_param_names)
            )
        )
        if not config.optimizer.allow_unused_parameters:
            raise Exception(
                "Found model parameters not used by optimizer. Please check the "
                "model's get_optimizer_parameters and add all parameters. If this "
                "is intended, set optimizer.allow_unused_parameters to True to "
                "ignore it."
            )


def get_extra_params(config):
    valid_args = { "Adam": ["lr", "weight_decay", "betas", "eps", "amsgrad"] }
    params = config.optimizer.get("params", {})
    params = { k:v for k, v in params.items() if k in valid_args[config.optimizer.type]}
    return params


def lr_lambda_update(i_iter, cfg):
    if cfg.training.use_warmup is True and i_iter <= cfg.training.warmup_iterations:
        alpha = float(i_iter) / float(cfg.training.warmup_iterations)
        return cfg.training.warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg.training.lr_steps, i_iter)
        return pow(cfg.training.lr_ratio, idx)


def clip_gradients(model, optimizer, i_iter, writer, config, scale=1.0):
    max_grad_l2_norm = config.training.max_grad_l2_norm
    clip_norm_mode = config.training.clip_norm_mode

    if config.training.clip_gradients and max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            if hasattr(optimizer, "clip_grad_norm"):
                norm = optimizer.clip_grad_norm(max_grad_l2_norm * scale)
            else:
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_l2_norm * scale
                )
            if writer is not None:
                writer.add_scalars({"grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )
    if config.training.nan_to_num:
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if get_world_size() > 1:
                torch.distributed.all_reduce(flat)
                flat /= get_world_size()
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)


def get_max_updates(config_max_updates, config_max_epochs, train_loader, update_freq):
    if config_max_updates is None and config_max_epochs is None:
        raise ValueError("Neither max_updates nor max_epochs is specified.")

    if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
        warnings.warn(
            "max_epochs not supported for Iterable datasets. Falling back "
            + "to max_updates."
        )
        return config_max_updates, config_max_epochs

    if config_max_updates is not None and config_max_epochs is not None:
        warnings.warn(
            "Both max_updates and max_epochs are specified. "
            + f"Favoring max_epochs: {config_max_epochs}"
        )

    if config_max_epochs is not None:
        assert (
            hasattr(train_loader, "__len__") and len(train_loader) != 0
        ), "max_epochs can't be used with IterableDatasets"
        max_updates = math.ceil(len(train_loader) / update_freq) * config_max_epochs
        max_epochs = config_max_epochs
    else:
        max_updates = config_max_updates
        if hasattr(train_loader, "__len__") and len(train_loader) != 0:
            max_epochs = max_updates / len(train_loader)
        else:
            max_epochs = math.inf

    return max_updates, max_epochs


def set_gradient_requirement(model, train_discriminator, 
                             generators=["generator"], 
                             discriminators=["discriminator"],
                             generator_modules=None):

    if is_dist_initialized():
        model = model.module

    for generator in generators:
        module = getattr(model, generator)
        if generator_modules is None:
            module.requires_grad_(not train_discriminator)
        else:
            for generator_module in generator_modules:
                submodule = getattr(module, generator_module)
                submodule.requires_grad_(not train_discriminator)

    for discriminator in discriminators:
        module = getattr(model, discriminator)
        module.requires_grad_(train_discriminator)


def get_model_config():
    config = registry.get("config")
    return config.model_config.get(config.get("model"))


def is_fixed_generator():
    model_config = get_model_config()
    return model_config.pretrained_ckpt_path != ""


def move_to_device(batch, device):
    assert isinstance(batch, Dict), "invalid batch sample type"
    assert isinstance(device, torch.device), "invalid device type"
    return { k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def extract_loss(model_output: Dict[str, Any], loss_divisor: int) -> torch.Tensor:
    loss_dict = model_output["losses"]
    assert len(loss_dict) != 0, (
        "Model returned an empty loss dict. "
        "Did you forget to (i) define losses in your model configuration or"
        "(ii) return losses dict from your model?"
    )

    # Since losses are batch averaged, this makes sure the
    # scaling is right.
    for key, value in loss_dict.items():
        value = value.mean() / loss_divisor
        model_output["losses"][key] = value

    loss = sum(loss.mean() for loss in loss_dict.values())
    return loss


def get_batch_size():
    """
    get batch size from config
    """
    from lib.utils.configuration import get_global_config

    batch_size = get_global_config("training.batch_size")
    world_size = get_world_size()

    batch_size_per_device = get_global_config("training.batch_size_per_device")

    if batch_size_per_device is not None:
        logger.info(
            f"training.batch_size_per_device has been used as {batch_size_per_device} "
            + "This will override training.batch_size and set the global batch size to "
            + f"{batch_size_per_device} x {world_size} = "
            + f"{batch_size_per_device * world_size}"
        )
        batch_size = batch_size_per_device * world_size

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    if (
        get_global_config("model") in ["cips3d", "vnerf"] and 
        (batch_size // world_size) > 4 and 
        (batch_size // world_size) % 4 != 0
    ):
        raise RuntimeError(
            "Batch size per device {} must be divisible by 4 "
            "to be compatible with {} multiscale discriminator.".format(batch_size // world_size,
                                                                        get_global_config("model"))
        )

    return batch_size // world_size


def infer_batch_size(sample: Dict):
    """
    infer batch size from a sample returned by "Dataloader"
    """
    if "Ts" in sample:
        return sample["Ts"].shape[0]
    else:
        return sample["frames"].shape[0]
    # batch_sizes = np.array([v.shape[0] for _, v in sample.items() if isinstance(v, torch.Tensor)])
    # assert len(batch_sizes) > 0, "no tensor found in sample"
    # assert np.all(batch_sizes == batch_sizes[0])
    # return batch_sizes[0]


def updir(d, n):
    """Given path d, go up n dirs from d and return that path"""
    ret_val = d
    for _ in range(n):
        ret_val = os.path.dirname(ret_val)
    return ret_val


def write_gif_videos(path, video, fps):
    imgs = [Image.fromarray(img.numpy().astype(np.uint8)).convert('RGB') for img in video]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], optimize=False, duration=len(video), loop=0, fps=fps)


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def exec_process(cmd, verbose=False):
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if verbose:
        logger.info(stdout.decode("utf-8"))
        logger.info(stderr.decode("utf-8"))
    return stdout, stderr


class BatchCollator():

    def __init__(self, dataset_name, dataset_type) -> None:
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
    
    def __call__(self, batch):
        samples = {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type
        }
        if isinstance(batch, list):
            for key in batch[0]:
                samples[key] = torch.cat([sample[key].unsqueeze(0) \
                                          for sample in batch])
            return samples
        else:
            raise TypeError("Batch samples returned by \"Dataloader\" must be a list")


def get_z(batch_size, z_dim=512, timesteps=None, device=torch.device("cpu"), dist="gaussian"):
    if dist == 'gaussian':
        z = torch.randn((batch_size, z_dim), device=device)
    elif dist == 'uniform':
        z = torch.rand((batch_size, z_dim), device=device) * 2 - 1
    if timesteps is not None:
        z = z.unsqueeze(1).repeat(1, timesteps, 1).reshape(-1, z_dim)
    return z


def get_z_motion(batch_size, z_dim=512, timesteps=None, device=torch.device("cpu"), dist="gaussian"):
    if dist == 'gaussian':
        z = torch.randn((batch_size, z_dim), device=device)
    elif dist == 'uniform':
        z = torch.rand((batch_size, z_dim), device=device) * 2 - 1
    if timesteps is not None:
        z = z.unsqueeze(1).repeat(1, timesteps, 1).reshape(-1, z_dim)
    return z


beta_dist1 = torch.distributions.beta.Beta(2., 1., validate_args=None)
beta_dist2 = torch.distributions.beta.Beta(1., 2., validate_args=None)

def get_t(batch_size, device=torch.device("cpu"), test=False):
    if not test:
        Ts = torch.cat([beta_dist1.sample((batch_size, 1)),
                        beta_dist2.sample((batch_size, 1))], dim=1).to(device)
        Ts = torch.cat([Ts.min(dim=1, keepdim=True)[0], Ts.max(dim=1, keepdim=True)[0]], dim=1)
    else:
        NUM_FRAMES = registry.get("config").dataset_config.voxceleb.num_frames
        Ts = torch.linspace(0, 1., steps=NUM_FRAMES).unsqueeze(0).repeat(batch_size, 1).to(device)
    return Ts