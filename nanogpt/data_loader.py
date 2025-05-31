from pathlib import Path

import torch
from tiktoken import Encoding

from nanogpt.logging import get_master_logger
from nanogpt.utils import DDPCoord

logger = get_master_logger()


class DataLoader:
    def __init__(
        self,
        B: int,  # noqa: N803
        T: int,  # noqa: N803
        ddp_coord: DDPCoord,
        *,
        data_file: Path,
        encoder: Encoding,
    ):
        self.B = B
        self.T = T
        self.ddp_coord = ddp_coord

        # Load the data in to memory
        with open(data_file) as f:
            text = f.read()
        tokens = encoder.encode(text)
        self.tokens = torch.tensor(tokens)
        logger.info(f"Loaded {len(self.tokens)} tokens")

        n_micro_batches = len(self.tokens) // (self.B * self.T * self.ddp_coord.world_size)
        logger.info(f"1 epoch = {n_micro_batches} micro-batches")

        # Initialize the `current_position` by resetting it
        self.reset_current_position()

    def next_microbatch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Test if the batch goes out of bounds, if so, reset the `current_position`
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.reset_current_position()

        buf = self.tokens[self.current_position : self.current_position + (self.B * self.T + 1)]
        x = (buf[:-1]).view(self.B, self.T)  # inputs
        y = (buf[1:]).view(self.B, self.T)  # targets

        # Advance the current position of our data
        self.current_position += self.B * self.T * self.ddp_coord.world_size
        return x, y

    def reset_current_position(self) -> None:
        # Reset the `current_position` strided out based on the GPU's rank
        self.current_position = self.B * self.T * self.ddp_coord.rank
