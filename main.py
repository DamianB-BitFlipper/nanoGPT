from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger

from nanogpt.data_loader import DataLoader
from nanogpt.model import GPT2, GPT2Config
from nanogpt.utils import get_compute_device

COMPUTE_DEVICE = get_compute_device()


def fix_random_seeds() -> None:
    # Set the seed to 42
    torch.manual_seed(42)
    if COMPUTE_DEVICE == "cuda":
        torch.cuda.manual_seed(42)
    elif COMPUTE_DEVICE == "mps":
        torch.mps.manual_seed(42)


def main_hello_world() -> None:
    num_return_sequences = 5
    max_length = 30
    top_k = 50

    enc = tiktoken.get_encoding("gpt2")
    gpt2 = GPT2.from_pretrained("gpt2")
    gpt2.to(COMPUTE_DEVICE)
    logger.info("Model loaded")

    # Configures the model for inference
    gpt2.eval()

    # Repeat the tokenized input string `num_return_sequences` times
    x = (
        torch.tensor(enc.encode("Hello, I'm a language model,"), dtype=torch.long)
        .unsqueeze(0)
        .repeat(num_return_sequences, 1)
    ).to(COMPUTE_DEVICE)

    # Generate! Right now x is (B, T) where B = 5, T = 8
    while x.size(1) < max_length:
        # Forward the model to get the logits
        with torch.no_grad():
            logits, _ = gpt2(x)  # (B, T, vocab_size)
            # Take the logits at the final position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Get the probabilities
            probs = F.softmax(logits, dim=-1)
            # Do top-k sampling of `top_k` (huggingface pipeline default)
            # topk_probs here becomes (5, top_k), topk_indicies is (5, top_k)
            topk_probs, topk_indicies = torch.topk(probs, top_k, dim=-1)
            # Select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # Gather the corresponding indices
            xcol = torch.gather(topk_indicies, -1, ix)  # (B, 1)
            # Append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # Print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        logger.info(f"> {decoded}")


def main_train() -> None:
    enc = tiktoken.get_encoding("gpt2")

    gpt2 = GPT2(GPT2Config())
    gpt2.to(COMPUTE_DEVICE)
    logger.info("Model loaded")

    train_loader = DataLoader(
        B=4,
        T=32,
        data_file=Path("./data/my_tiny_shakespeare.txt"),
        encoder=enc,
    )

    # Optimize!
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=3e-4)
    for i in range(50):
        optimizer.zero_grad()

        # Get a batch of training data
        x, y = train_loader.next_batch()
        x = x.to(COMPUTE_DEVICE)
        y = y.to(COMPUTE_DEVICE)

        logits, loss = gpt2(x, y)  # (B, T, vocab_size)
        loss.backward()
        optimizer.step()
        logger.info(f"step: {i}, loss: {loss.item()}")


if __name__ == "__main__":
    fix_random_seeds()
    # main_hello_world()
    main_train()
