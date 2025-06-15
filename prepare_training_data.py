"""
Inspired from https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py

FineFineWeb dataset (for GPT2 pretraining)
https://huggingface.co/datasets/m-a-p/FineFineWeb

First download (a subset) of the dataset to disk. The follow command downloads
the first 10 shards (217GB) of the dataset:

$ for i in {0..9}; do shard_num=$(printf "%06d" $i); echo "Downloading shard $shardnum..."; huggingface-cli download m-a-p/FineFineWeb --repo-type dataset --include "*${shard_num}.jsonl"; done

To process and prepare this training data, simply run:
$ python prepare_training_data.py

Will tokenize the dataset and save shards as a new dataset in the HF_HOME directory.
"""  # noqa: E501

import multiprocessing as mp
import os
from collections.abc import Iterator
from typing import cast

import datasets
import numpy as np
import tiktoken
from datasets import Dataset, IterableDataset, load_dataset
from numpy.typing import NDArray
from tqdm import tqdm

from nanogpt.logging import get_all_logger

logger = get_all_logger()

# Divide by 2 to avoid using hyperthreaded cores, just the physical cores
N_PROCS = max(1, (os.cpu_count() or 0) // 2)

# 100M tokens per shard
SHARD_SIZE = 100_000_000

encoder = tiktoken.get_encoding("gpt2")
EOT = encoder._special_tokens["<|endoftext|>"]  # End of text token


def tokenize(row: dict[str, str]) -> NDArray[np.uint16]:
    """Token a single row of data and return it as a numpy array of uint16 tokens."""
    tokens = encoder.encode_ordinary(row["text"])
    tokens.append(EOT)
    tokens_np = np.array(tokens)

    # Sanity check that all tokens fit in a uint16
    if not ((0 <= tokens_np).all() and (tokens_np < 2**16).all()):
        raise ValueError("Token dictionary too large for uint16")

    # Convert the type and return
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def tokenized_rows_generator(
    *, dataset: IterableDataset
) -> Iterator[dict[str, NDArray[np.uint16]]]:
    with mp.Pool(N_PROCS) as pool:
        progress_bar = tqdm(desc="Tokenizing", unit="tokens", unit_scale=True)

        try:
            # Tokenize the dataset in parallel with each process processing 256 rows at a time
            for tokens in pool.imap(tokenize, dataset, chunksize=256):
                progress_bar.update(len(tokens))

                yield {"tokens": tokens}
        finally:
            progress_bar.close()


def main() -> None:
    # Check if HF_HOME environment variable is defined
    hf_home = os.getenv("HF_HOME")
    if hf_home is None:
        raise OSError("HF_HOME environment variable is not defined")

    # Lazily load the dataset with the rows randomly shuffled
    dataset = (
        cast(
            IterableDataset,
            load_dataset(
                f"{hf_home}/hub/datasets--m-a-p--FineFineWeb/",
                split="train",
                streaming=True,
            ),
        )
        .shuffle(seed=42)
        .take(
            int(16e6)
        ),  # 650 tokens per row on average, 16M rows is a bit more than 10B tokens
    )

    # Load the shards from a generator to avoid loading the entire dataset to RAM
    dataset = cast(
        Dataset,
        Dataset.from_generator(
            tokenized_rows_generator,
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value("uint16"))}),
            gen_kwargs={"dataset": dataset},
            writer_batch_size=int(25e3),  # Write to disk every 25k rows
        ),
    )

    # Split the dataset in to 99% train and 1% test
    train_test_split = dataset.train_test_split(test_size=0.01, seed=42)

    logger.info("Saving dataset to disk...")
    train_test_split.save_to_disk(
        f"{hf_home}/hub/datasets--m-a-p--FineFineWeb-tokenized/",
        num_proc=N_PROCS,
    )
    logger.info("Dataset saved successfully!")


if __name__ == "__main__":
    main()
