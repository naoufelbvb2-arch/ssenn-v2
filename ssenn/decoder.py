"""One pre-normalized, causal decoder block shared by all C02-A learners."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DecoderConfig:
    vocab: int
    context: int
    width: int
    heads: int
    ff: int

    def __post_init__(self):
        if min(self.vocab, self.context, self.width, self.heads, self.ff) < 1:
            raise ValueError("Decoder dimensions must be positive")
        if self.width % self.heads or self.width % 2:
            raise ValueError("Width must be even and divisible by the head count")


def split_heads(x: Tensor, heads: int) -> Tensor:
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).transpose(1, 2)


def join_heads(x: Tensor) -> Tensor:
    return x.transpose(1, 2).flatten(2)


def rms(x: Tensor) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True) + 1e-5).rsqrt()


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, seed: int, trainable_by_autograd: bool = False):
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        rng = torch.Generator().manual_seed(seed)
        shapes = {"embedding": (cfg.vocab, d), "q": (d, d), "k": (d, d), "v": (d, d),
                  "attention_out": (d, d), "ff_in": (d, cfg.ff), "ff_out": (cfg.ff, d),
                  "head": (d, cfg.vocab)}
        self.weights = nn.ParameterDict()
        for name, shape in shapes.items():
            std = 0.25 if name == "embedding" else 1 / math.sqrt(shape[0])
            self.weights[name] = nn.Parameter(torch.randn(shape, generator=rng) * std,
                                              requires_grad=trainable_by_autograd)
        for name in ("norm_attention", "norm_ff", "norm_final"):
            self.weights[name] = nn.Parameter(torch.ones(d), requires_grad=trainable_by_autograd)
        angle = torch.arange(cfg.context)[:, None] * (10000 ** (-torch.arange(0, d, 2) / d))
        self.register_buffer("positions", 0.125 * torch.stack((angle.sin(), angle.cos()), -1).flatten(1))
        self.register_buffer("mask", torch.ones(cfg.context, cfg.context, dtype=torch.bool).tril())

    def compute(self, ids: Tensor, weights, self_only: bool = False, capture: bool = False):
        t = ids.size(1)
        if t < 1 or t > self.cfg.context:
            raise ValueError("Sequence length outside decoder context")
        x = weights["embedding"][ids] + self.positions[:t]
        ra = rms(x)
        a = ra * weights["norm_attention"]
        q, k, v = (split_heads(a @ weights[n], self.cfg.heads) for n in ("q", "k", "v"))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        mask = torch.eye(t, dtype=torch.bool, device=ids.device) if self_only else self.mask[:t, :t]
        prob = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        mixed = join_heads(prob @ v)
        mid = x + mixed @ weights["attention_out"]
        rf = rms(mid)
        f = rf * weights["norm_ff"]
        hidden = (f @ weights["ff_in"]).tanh()
        last = mid + hidden @ weights["ff_out"]
        rl = rms(last)
        head_input = rl * weights["norm_final"]
        logits = head_input @ weights["head"]
        trace = dict(ra=ra, a=a, q=q, k=k, v=v, prob=prob, mixed=mixed, rf=rf, f=f,
                     hidden=hidden, rl=rl, head_input=head_input) if capture else None
        return logits, trace

    def forward(self, ids: Tensor, self_only: bool = False) -> Tensor:
        return self.compute(ids, self.weights, self_only)[0]
