import os

import torch
from loguru import logger
from pydantic import BaseModel, computed_field
from torch.distributed import init_process_group


class DDPCoord(BaseModel):
    rank: int
    local_rank: int
    world_size: int

    @computed_field
    @property
    def master_process(self) -> bool:
        return self.rank == 0


def init_compute_device(*, use_mps: bool = False) -> tuple[str, DDPCoord]:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and use_mps:
        # MPS does not support all of the pytorch functions. It needs to be explicitly
        # allowed, even when the hardware platform supports it
        device = "mps"

    # Calculate the distributed data parallel coordinate
    if not (
        device == "cuda"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        ddp_coord = DDPCoord(
            rank=0,
            local_rank=0,
            world_size=1,
        )
    else:
        # Initialize DDP and the specific CUDA device
        init_process_group(backend="nccl")
        ddp_coord = DDPCoord(
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
        device = f"cuda:{ddp_coord.local_rank}"
        torch.cuda.set_device(device)

    logger.info(f"Using device: {device} with coordinates {ddp_coord}")
    return device, ddp_coord


def fix_random_seeds(compute_device: str) -> None:
    # Set the seed to 42
    torch.manual_seed(42)
    if compute_device == "cuda":
        torch.cuda.manual_seed(42)
    elif compute_device == "mps":
        torch.mps.manual_seed(42)
