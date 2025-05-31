from pathlib import Path

import torch
from tiktoken import Encoding

from nanogpt.logging import get_master_logger

logger = get_master_logger()


class DataLoader:
    def __init__(self, B: int, T: int, *, data_file: Path, encoder: Encoding):  # noqa: N803
        self.B = B
        self.T = T

        # Load the data in to memory
        with open(data_file) as f:
            text = f.read()
        tokens = encoder.encode(text)
        self.tokens = torch.tensor(tokens)
        logger.info(f"Loaded {len(self.tokens)} tokens")
        logger.info(f"1 epoch = {len(self.tokens) // (self.B * self.T)} micro-batches")

        # State
        self.current_position = 0

    def next_microbatch(self) -> tuple[torch.Tensor, torch.Tensor]:
        buf = self.tokens[self.current_position : self.current_position + (self.B * self.T + 1)]
        x = (buf[:-1]).view(self.B, self.T)  # inputs
        y = (buf[1:]).view(self.B, self.T)  # targets

        # Advance the current position of our data
        self.current_position += self.B * self.T

        # If loading the next batch would go out of bounds, reset
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
