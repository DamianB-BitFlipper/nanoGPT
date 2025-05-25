import torch
from loguru import logger


def get_compute_device(*, use_mps: bool = False) -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and use_mps:
        # MPS does not support all of the pytorch functions. It needs to be explicitly
        # allowed, even when the hardware platform supports it
        device = "mps"

    logger.info(f"Using device: {device}")
    return device
