import tiktoken
import torch
import torch.nn.functional as F  # noqa: N812

from nanogpt.logging import get_master_logger
from nanogpt.model import GPT2
from nanogpt.utils import fix_random_seeds, get_compute_device

logger = get_master_logger()

COMPUTE_DEVICE, _ = get_compute_device()


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


if __name__ == "__main__":
    fix_random_seeds(COMPUTE_DEVICE)
    main_hello_world()
