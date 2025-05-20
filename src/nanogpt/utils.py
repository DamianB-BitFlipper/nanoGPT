from typing import TextIO

import torch
from loguru import logger
from tiktoken import Encoding


def get_compute_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Using device: {device}")
    return device


def get_databatch(
    encoder: Encoding,
    infile: TextIO,
    *,
    n_batches: int,
    tokens_per_sample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = infile.read()

    # As an optimization, we only need roughly `4 * n_batches * tokens_per_sample` characters
    text = text[: (4 * n_batches * tokens_per_sample)]
    tokens = encoder.encode(text)

    # Fetch all of the tokens used in the batch data plus 1 for
    # the expected completion of the last token
    buf = torch.tensor(tokens[: (n_batches * tokens_per_sample + 1)])

    x = buf[:-1].view(n_batches, tokens_per_sample)  # Inputs
    y = buf[1:].view(n_batches, tokens_per_sample)  # Expected outputs
    return x, y
