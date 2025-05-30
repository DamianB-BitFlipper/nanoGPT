"""Run with DDP using `torchrun --standalone --nproc-per-node=<N_GPUS> train.py`"""

import math
import time
from pathlib import Path
from typing import cast

import tiktoken
import torch
from torch.nn.parallel import DistributedDataParallel

from nanogpt.data_loader import DataLoader
from nanogpt.logging import get_all_logger, get_master_logger
from nanogpt.model import GPT2, GPT2Config
from nanogpt.utils import (
    fix_random_seeds,
    get_compute_device,
    init_ddp,
    is_ddp_enabled,
    register_cleanup_ddp,
)

logger = get_master_logger()
all_logger = get_all_logger()

COMPUTE_DEVICE, DDP_COORD = get_compute_device()
all_logger.info(f"Using device: {COMPUTE_DEVICE} with coordinates {DDP_COORD}")

# Initialize distribtued data parallelism for this `COMPUTE_DEVICE`
init_ddp(COMPUTE_DEVICE)

# Register DDP cleanup whenever the process exits
register_cleanup_ddp()

# Disable logging if not the master process
if not DDP_COORD.master_process:
    logger.disable("__main__")


def get_gpt3_lr(
    step: int, *, max_lr: float, min_lr: float, warmup_steps: int, max_decay_steps: int
) -> float:
    # 1) Linear warm up for `warmup_steps` steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) If `step > max_decay_steps`, return a minimum learning rate
    if step > max_decay_steps:
        return min_lr
    # 3) In between, use a cosine decay down to the minimum learning rate
    decay_ratio = (step - warmup_steps) / (max_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    # The `coeff` starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def main_train() -> None:
    # Use tensor float 32 for matrix multiplication
    torch.set_float32_matmul_precision("high")

    enc = tiktoken.get_encoding("gpt2")

    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    gpt2 = GPT2(GPT2Config(vocab_size=50304))
    gpt2.to(COMPUTE_DEVICE)
    gpt2 = torch.compile(gpt2)

    # Wrap `gpt2` with the `DistributedDataParallel` class to utilize the distributed training
    if is_ddp_enabled():
        gpt2 = DistributedDataParallel(gpt2, device_ids=[DDP_COORD.local_rank])

    # Force a type cast to `GPT2` to keep pyright satisfied
    gpt2 = cast(GPT2, gpt2)

    logger.info("Model loaded")

    total_batch_size = 524288  # 2**19, ~0.5M tokens
    B = 16  # Microbatch size
    T = 1024  # Context length
    assert total_batch_size % (B * T * DDP_COORD.world_size) == 0, (
        "Make sure the `total_batch_size` is divisible by B * T * DDP_COORD.world_size"
    )
    grad_accum_steps = total_batch_size // (B * T * DDP_COORD.world_size)
    logger.info(f"Total desired batch size: {total_batch_size}")
    logger.info(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoader(
        B=B,
        T=T,
        ddp_coord=DDP_COORD,
        data_file=Path("./data/my_tiny_shakespeare.txt"),
        encoder=enc,
    )

    # Build the optimizer with GPT-3 hyper-parameters
    optimizer = gpt2.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        device=COMPUTE_DEVICE,
    )

    # Train!
    for step in range(50):
        t0 = time.time()

        optimizer.zero_grad()

        loss_accum = torch.tensor(0.0, device=COMPUTE_DEVICE)
        for _micro_step in range(grad_accum_steps):
            # Get a batch of training data
            x, y = train_loader.next_microbatch()
            x = x.to(COMPUTE_DEVICE)
            y = y.to(COMPUTE_DEVICE)

            # Use reduced precision for the forward pass
            with torch.autocast(device_type=COMPUTE_DEVICE, dtype=torch.bfloat16):
                logits, loss = gpt2(x, y)  # (B, T, vocab_size)

            # Normalize the `loss` since we want it in the end to be the average
            # across the entire batch. Without the normalization, it would be the sum
            # of the averages of the micro-batches, which is not equivalent
            loss /= grad_accum_steps
            loss_accum += loss.detach()

            # Deposit the gradients during the backward pass
            loss.backward()

        # Clip the global gradient norm to 1 to prevent huge updates
        norm = torch.nn.utils.clip_grad_norm_(gpt2.parameters(), 1.0)

        # Utilize a custom learning rate schedule from GPT-3's paper
        lr = get_gpt3_lr(
            step,
            max_lr=6e-4,
            min_lr=6e-5,
            warmup_steps=10,
            max_decay_steps=50,
        )

        # Set this `lr` on all of the optimizer param groups
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Apply the gradient
        optimizer.step()

        # Wait for the GPU kernels to finish their scheduled jobs
        torch.cuda.synchronize()
        t1 = time.time()

        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
        logger.info(
            f"step: {step} | "
            f"loss: {loss_accum.item():.6f} | "
            f"lr: {lr:.5f} | "
            f"norm: {norm:.4f} | "
            f"dt: {(t1 - t0) * 1000:.2f}ms | "
            f"tok/sec: {tokens_per_sec:.2f}"
        )


if __name__ == "__main__":
    fix_random_seeds(COMPUTE_DEVICE)
    main_train()
