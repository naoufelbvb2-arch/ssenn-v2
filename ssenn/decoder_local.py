"""Fixed direct feedback to local decoder sites. No backward pass across sublayers."""

import math
import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot

from .decoder import Decoder, DecoderConfig, join_heads, split_heads


class LocalDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, seed: int, rate: float, anchor_rate: float | None = None):
        super().__init__()
        if not math.isfinite(rate) or rate <= 0 or (anchor_rate is not None and not 0 < anchor_rate <= 1):
            raise ValueError("Invalid local learning rate")
        self.net = Decoder(cfg, seed)
        self.rate, self.anchor_rate = rate, anchor_rate
        rng = torch.Generator().manual_seed(seed + 1)
        self.feedback = nn.Module()
        sites = {"embedding": cfg.width, "norm_attention": cfg.width, "attention_mix": cfg.width,
                 "attention_out": cfg.width, "norm_ff": cfg.width, "ff_in": cfg.ff,
                 "ff_out": cfg.width, "norm_final": cfg.width}
        for site, width in sites.items():
            self.feedback.register_buffer(site, torch.randn(cfg.vocab, width, generator=rng) / math.sqrt(cfg.vocab))
        self.anchors = nn.Module()
        if anchor_rate is not None:
            for name, w in self.net.weights.items():
                self.anchors.register_buffer(name, w.detach().clone())

    @torch.no_grad()
    def forward(self, ids: Tensor, self_only: bool = False) -> Tensor:
        w = self.net.weights if self.anchor_rate is None else dict(self.anchors.named_buffers())
        return self.net.compute(ids, w, self_only)[0]

    @torch.no_grad()
    def train_step(self, x: Tensor, y: Tensor) -> None:
        w = self.net.weights
        logits, c = self.net.compute(x, w, capture=True)
        # Divide once at the source. Training minibatches contain only valid targets.
        e = (one_hot(y, logits.size(-1)) - logits.softmax(-1)) / y.numel()
        def direct(site):
            return e @ getattr(self.feedback, site)
        def outer(a, b):
            return a.flatten(0, 1).T @ b.flatten(0, 1)

        m = split_heads(direct("attention_mix"), self.net.cfg.heads)
        p, v = c["prob"], c["v"]
        affinity = m @ v.transpose(-2, -1)
        ds = p * (affinity - (affinity * p).sum(-1, keepdim=True))
        ds = ds / math.sqrt(v.size(-1))
        drives = {
            "q": outer(c["a"], join_heads(ds @ c["k"])),
            "k": outer(c["a"], join_heads(ds.transpose(-2, -1) @ c["q"])),
            "v": outer(c["a"], join_heads(p.transpose(-2, -1) @ m)),
            "attention_out": outer(c["mixed"], direct("attention_out")),
            "ff_in": outer(c["f"], direct("ff_in") * (1 - c["hidden"].square())),
            "ff_out": outer(c["hidden"], direct("ff_out")),
            "head": outer(c["head_input"], e),
        }
        for name, key in (("norm_attention", "ra"), ("norm_ff", "rf"), ("norm_final", "rl")):
            drives[name] = (c[key] * direct(name)).sum((0, 1))
        emb = torch.zeros_like(w["embedding"])
        emb.index_add_(0, x.flatten(), direct("embedding").flatten(0, 1))
        drives["embedding"] = emb
        next_w = {name: w[name] + self.rate * drives[name] for name in w}
        next_a = {}
        for name, old in self.anchors.named_buffers():
            next_a[name] = old + self.anchor_rate * (next_w[name] - old)
        if not all(t.isfinite().all() for t in (*next_w.values(), *next_a.values())):
            raise FloatingPointError("Nonfinite local decoder update")
        for name, t in next_w.items():
            w[name].copy_(t)
        for name, t in next_a.items():
            getattr(self.anchors, name).copy_(t)
