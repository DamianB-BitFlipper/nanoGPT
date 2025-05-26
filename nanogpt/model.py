from __future__ import annotations

from typing import ClassVar, Self

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator
from transformers.models.gpt2 import GPT2LMHeadModel


class GPT2Config(BaseModel):
    model_config = ConfigDict(frozen=True)  # Makes the model immutable like a dataclass

    block_size: int = 1024  # Max sequence length
    # Number of tokens, 50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    vocab_size: int = 50257
    n_layer: int = 12  # Number of layers
    n_head: int = 12  # Number of heads
    n_embd: int = 768  # Embedding dimension

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"Embedding dimension ({self.n_embd}) must be divisible by "
                f"number of heads ({self.n_head})",
            )
        return self


class CausalSelfAttention(nn.Module):
    bias: ClassVar[torch.Tensor]

    def __init__(self, config: GPT2Config):
        super().__init__()
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Scale the initialization because of the residual connection
        self.c_proj.NANOGPT_SCALE_INIT = True  # pyright: ignore[reportArgumentType]

        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Not really a `bias`, more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1,
                1,
                config.block_size,
                config.block_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = (  # noqa: N806
            x.size()
        )  # Batch size, sequence length, embedding dimensionality (n_embd)
        # Calculate query, key, values for all heads in batch and move head forward
        # to be the batch dim `nh` is "number of heads", hs is "head size" and C
        # (number of channels) = nh * hs. e.g. in GPT-2 (124M), n_head=12, hs=64,
        # so nh*hs=C=768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )  # (B, nh, T, hs)

        # Attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Flash attention pytorch implementation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # Re-assemble all head outputs side by side
        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Scale the initialization because of the residual connection
        self.c_proj.NANOGPT_SCALE_INIT = True  # pyright: ignore[reportArgumentType]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            ),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme since embedding and output matrices should be identical.
        # This is since both the input and outspaces are similar, so it makes sense to
        # codify them similarly. This produces better results
        self.transformer.wte.weight = self.lm_head.weight  # pyright: ignore[reportAttributeAccessIssue]

        # Initialize the tensors. The `self.apply` recursively iterates all modules and
        # sub-modules and applies the target function
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize tensors according to the original GPT-2 implementation."""
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale the initialization randomness for the layers marked
            # with `NANOGPT_SCALE_INIT` since they form a residual chain
            # to prevent gradient explosion/collapse
            if getattr(module, "NANOGPT_SCALE_INIT", None):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The `idx` is of shape (B, T)
        B, T = idx.size()  # noqa: N806
        assert T <= self.config.block_size, (
            f"Cannot forward sequences of length {T}, block size is {self.config.block_size}"
        )

        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shake (T)
        # Position embeddings of shape (T, n_embed)
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        # Token embeddings of shape (B, T, n_embed)
        tok_emb = self.transformer.wte(idx)  # type: ignore
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer
        for block in self.transformer.h:  # type: ignore
            x = block(x)

        # Forward the final `LayerNorm` and the classifier
        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
        device: str,
    ) -> torch.optim.Optimizer:
        # Start with all candidate parameters that require a gradient
        param_dict = {pname: p for pname, p in self.named_parameters() if p.requires_grad}

        # Create optimizer groups. Any parameters that are 2D (matrices) will be
        # weight decayed. Otherwise no. I.e. all weight tensors in matmuls + embeddings
        # decay, all biases and layernorms do not
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"Num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        logger.info(
            f"Num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        # Create an AdamW optimizer and use the fused version if on CUDA
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            fused="cuda" in device,
        )

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type: str) -> GPT2:
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        logger.info(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # Always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # Always 1024 for GPT model checkpoints

        # Create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # Discard this mask / buffer, not a param

        # Initialize a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # Ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # Same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # Basically the openai checkpoints use a "Conv1D" module, but we only want
        # to use a vanilla nn.Linear this means that we have to transpose these
        # weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
