from __future__ import annotations

import sys

import loguru
from loguru import logger

from nanogpt.utils import get_compute_device

_, DDP_COORD = get_compute_device()

# Remove default handler from the main logger
logger.remove()

# Create master logger - only add handler if master process
if DDP_COORD.master_process:
    master_handler_id = logger.add(
        sys.stderr,
        format="[MASTER] {time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: record["extra"].get("logger_type") == "master",
    )

# Create all-processes logger - always add handler
all_handler_id = logger.add(
    sys.stderr,
    format="[RANK-{extra[rank]}] {time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    filter=lambda record: record["extra"].get("logger_type") == "all",
)


def get_master_logger() -> loguru.Logger:
    """Logger that only logs from master process"""
    return logger.bind(logger_type="master")


def get_all_logger() -> loguru.Logger:
    """Logger that logs from any process"""
    return logger.bind(logger_type="all", rank=DDP_COORD.rank)
