from pathlib import Path

import torch
from loguru import logger
from tiktoken import Encoding


class DataLoader:
    def __init__(self, B: int, T: int, *, data_file: Path, encoder: Encoding):
        self.B = B
        self.T = T

        # Load the data in to memory
        with open(data_file) as f:
            text = f.read()
        tokens = encoder.encode(text)
        self.tokens = torch.tensor(tokens)
        logger.info(f"Loaded {len(self.tokens)} tokens")
        logger.info(f"1 epoch = {len(self.tokens) // (self.B * self.T)} batches")

        # State
        self.current_position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        buf = self.tokens[self.current_position : self.current_position + (self.B * self.T + 1)]
        x = (buf[:-1]).view(self.B, self.T)  # inputs
        y = (buf[1:]).view(self.B, self.T)  # targets

        # Advance the current position of our data
        self.current_position += self.B * self.T

        # If loading the next batch would go out of bounds, reset
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
