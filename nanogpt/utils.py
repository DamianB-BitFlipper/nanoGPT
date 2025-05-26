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


def fix_random_seeds(compute_device: str) -> None:
    # Set the seed to 42
    torch.manual_seed(42)
    if compute_device == "cuda":
        torch.cuda.manual_seed(42)
    elif compute_device == "mps":
        torch.mps.manual_seed(42)
