from pathlib import Path

import torch
from loguru import logger
from tiktoken import Encoding


class DataLoaderLite:
    def __init__(
        self,
        B: int,  # noqa: N803
        T: int,  # noqa: N803
        *,
        data_file: Path,
        encoder: Encoding,
    ) -> None:
        self.B = B
        self.T = T

        # Load all of the tokens in to memory
        with open(data_file) as f:
            text = f.read()
        tokens = encoder.encode(text)
        self.tokens = torch.Tensor(tokens)

        logger.info(f"Loaded {len(self.tokens)} tokens")
        logger.info(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # State
        self.current_position = 0

    def next_batch(self) -> None:
        buf = self.tokens[self.current_position : self.current_position + (self.B * self.T) + 1]
