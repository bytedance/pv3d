# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.

import os
import socket
import warnings
import logging
import subprocess
import torch
from torch import distributed as dist


logger = logging.getLogger(__name__)


def infer_init_method(config):
    if config.distributed.init_method is not None:
        return

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        config.distributed.init_method = "env://"
        config.distributed.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed.rank = int(os.environ["RANK"])
        config.distributed.no_spawn = True

    # we can determine the init method automatically for Slurm
    elif config.distributed.port > 0:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config.distributed.init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config.distributed.port,
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert config.distributed.world_size % nnodes == 0
                    gpus_per_node = config.distributed.world_size // nnodes
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    config.distributed.rank = node_id * gpus_per_node
                else:
                    assert ntasks_per_node == config.distributed.world_size // nnodes
                    config.distributed.no_spawn = True
                    config.distributed.rank = int(os.environ.get("SLURM_PROCID"))
                    config.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def distributed_init(config):
    if config.distributed.world_size == 1:
        raise ValueError("Cannot initialize distributed with distributed_world_size=1")

    if dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
        config.distributed.rank = dist.get_rank()
    else:
        logger.info(
            f"Distributed Init (Rank {config.distributed.rank}): "
            f"{config.distributed.init_method}"
        )
        dist.init_process_group(
            backend=config.distributed.backend,
            init_method=config.distributed.init_method,
            world_size=config.distributed.world_size,
            rank=config.distributed.rank,
        )
        logger.info(
            f"Initialized Host {socket.gethostname()} as Rank "
            f"{config.distributed.rank}"
        )

        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            # Set for onboxdataloader support
            split = config.distributed.init_method.split("//")
            assert len(split) == 2, (
                "host url for distributed should be split by '//' "
                + "into exactly two elements"
            )

            split = split[1].split(":")
            assert (
                len(split) == 2
            ), "host url should be of the form <host_url>:<host_port>"
            os.environ["MASTER_ADDR"] = split[0]
            os.environ["MASTER_PORT"] = split[1]

        # perform a dummy all-reduce to initialize the NCCL communicator
        dist.all_reduce(torch.zeros(1).cuda())

        suppress_output(is_master())
        config.distributed.rank = dist.get_rank()
    return config.distributed.rank


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_current_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


def is_master():
    return get_rank() == 0


def synchronize():
    if dist.is_initialized():
        dist.barrier()


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def broadcast_tensor(tensor, src=0):
    world_size = get_world_size()
    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.broadcast(tensor, src=0)

    return tensor


def broadcast_scalar(scalar, src=0, device="cpu"):
    if get_world_size() < 2:
        return scalar
    scalar_tensor = torch.tensor(scalar).long().to(device)
    scalar_tensor = broadcast_tensor(scalar_tensor, src)
    return scalar_tensor.item()


def reduce_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if dist.get_rank() == 0:
            tensor = tensor.div(world_size)

    return tensor


def gather_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.stack(tensor_list, dim=0)

    return tensor_list


def gather_tensor_along_batch(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


def reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        if len(dictionary) == 0:
            return dictionary

        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        dist.reduce(values, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)