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
"""

import os
import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Divide by 2 to avoid using hyperthreaded cores, just the physical cores
N_PROCS = max(1, os.cpu_count() // 2)

# 100M tokens per shard
SHARD_SIZE = 100_000_000

encoder = tiktoken.get_encoding("gpt2")
EOT = encoder._special_tokens['<|endoftext|>'] # End of text token

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

def main():
    # Check if HF_HOME environment variable is defined
    hf_home = os.getenv('HF_HOME')
    if hf_home is None:
        raise EnvironmentError("HF_HOME environment variable is not defined")

    # Lazily load the dataset with the rows randomly shuffled
    dataset = load_dataset(
        f"{hf_home}/hub/datasets--m-a-p--FineFineWeb/",
        split="train",
        streaming=True,
    ).shuffle(seed=42)

    with mp.Pool(N_PROCS) as pool:
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)

        shard_index = 0
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, dataset, chunksize=128):
            # Determine if this is a validation or train split
            split = "val" if shard_index == 0 else "train"

            # Initialize the progress bar if not already
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < SHARD_SIZE:
                # Simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # Update progress bar
                progress_bar.update(len(tokens))
            else:
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
                break


if __name__ == "__main__":
    main()
