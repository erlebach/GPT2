import math
import time
from dataclasses import dataclass
from typing import cast

import lightning as pl
import torch

# import torch._dynamo
import torch.nn as nn
from beartype import beartype
from data_utils import TextDataModule
from jaxtyping import Float, Integer
from torch import Tensor
from torch.nn import MultiheadAttention, TransformerDecoderLayer
from torch.nn import functional as F

# torch._dynamo.config.suppress_errors = True


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    # GE (2025-07-24)
    block_size = 32  # max seq length
    n_layer: int = 2  # number of layers
    n_head: int = 4  # number of heads
    n_embd: int = 16  # embedding dimension
    device: str = "cpu"


@beartype
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.key_padding_mask = None
        self.attn_mask = None

    def forward(
        self,
        x: Float[Tensor, "b seq emb"],
    ) -> Float[Tensor, "b seq emb"]:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # reassemble head outputs side by side
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # self.c_proj.NANOGPT_SCALE_INIT = 1  # not used

    def forward(
        self,
        x: Float[Tensor, "b seq emb"],
    ) -> Float[Tensor, "b seq emb"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1: nn.Module = nn.LayerNorm(config.n_embd)
        self.attn: nn.Module = CausalSelfAttention(config)
        self.ln_2: nn.Module = nn.LayerNorm(config.n_embd)
        self.mlp: nn.Module = MLP(config)

    def forward(
        self,
        x: Float[Tensor, "b seq emb"],
    ) -> Float[Tensor, "b seq emb"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SuperBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block1 = Block(config)
        self.block2 = Block(config)
        self.concat_proj = nn.Linear(config.n_embd * 2, config.n_embd)

        # MoE-style gating (optional enhancement)
        self.gate = nn.Linear(config.n_embd, 2)  # Learn which path to prefer
        self._init_moe_style()

    def forward(self, x):
        x1 = x + self.block1(x)
        x2 = x + self.block2(x)

        # Optional: Add gating mechanism
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [B, T, 2]
        x1_weighted = x1 * gate_weights[:, :, 0:1]
        x2_weighted = x2 * gate_weights[:, :, 1:2]

        concat = torch.cat((x1_weighted, x2_weighted), dim=-1)
        return self.concat_proj(concat)

    def _init_moe_style(self):
        """Initialize the SuperBlock using MoE principles.

        This ensures:
        1. Diverse initialization of parallel paths
        2. Conservative initialization of combination layer
        3. Balanced contribution from both paths
        """
        # Initialize the combination layer with smaller weights
        # This prevents one path from dominating early in training
        torch.nn.init.normal_(self.concat_proj.weight, mean=0.0, std=0.01)
        if self.concat_proj.bias is not None:
            torch.nn.init.zeros_(self.concat_proj.bias)

        # Ensure both blocks start with different random states
        # This promotes diversity in learned representations
        for i, block in enumerate([self.block1, self.block2]):
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    # Use different initialization scales for diversity
                    scale = 0.02 * (1.0 + 0.1 * i)  # Slightly different scales
                    torch.nn.init.normal_(module.weight, mean=0.0, std=scale)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([SuperBlock(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            },
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer["wte"].weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, SuperBlock):
            # Let SuperBlock handle its own MoE-style initialization
            module._init_moe_style()

    def forward(
        self,
        idx: Integer[Tensor, "b seq"],
        targets: Tensor | None,
    ):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        # position embeddings of shape (T, n_embd)
        pos_emb = self.transformer["wpe"](pos)
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer["wte"](idx)
        x = tok_emb + pos_emb
        mod_list = cast(nn.ModuleList, self.transformer["h"])
        # forward the blocks of the transformer
        for block in mod_list:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        print("model_type: ", model_type)  # GE
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(
            f"configuring optimizers with fused={use_fused} (fused_available={fused_available}) on device {device}"
        )
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


def train():
    if torch.cuda.is_available():
        device = "cuda"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Use TextDataModule from data_utils.py
    batch_size = 8
    block_size = 32
    data_path = "input.txt"
    dm = TextDataModule(data_path, block_size=block_size, batch_size=batch_size)
    dm.setup()
    train_loader = dm.train_dataloader()
    # Optionally, set grad_accum_steps if you want gradient accumulation
    grad_accum_steps = 1

    torch.set_float32_matmul_precision("high")

    model = GPT(GPTConfig(vocab_size=50304))
    device = "cpu"  # GE
    model.to(device)
    # Compilation creates beartype issues
    # model = torch.compile(model)

    # lr scheduler
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 500

    def get_lr(it):
        # 1 linear warmup for warmup_iter steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2 return min lr if it > max_steps
        if it > max_steps:
            return min_lr
        # 3 use cosine decay if inbetween
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and then goes to 0 in a cosine like fashion
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    for step in range(max_steps):
        t0 = time.time()

        optimizer.zero_grad()
        loss_accum = 0.0

        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):  # <<< ????
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps  # to normalize across grad accum steps
            loss_accum += loss.detach()
            loss.backward()  # accumulate grads
            # Only one batch per step for simplicity; remove break for full epoch
            break

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # grad clipping

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()  # <<< ???
        t1 = time.time()
        dt = (t1 - t0) * 1000
        print(
            f"step: {step:4d}, loss: {loss_accum.item()}, norm: {norm:.4f}, dt: {dt:.2f}ms"
        )


if __name__ == "__main__":
    train()
