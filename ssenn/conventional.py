"""Autograd and AdamW comparator with the same forward and initialization."""

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from .network import forward


class AdamNetwork:
    def __init__(self, weights: tuple[Tensor, Tensor], lr: float, weight_decay: float):
        self.weights = tuple(torch.nn.Parameter(w.clone()) for w in weights)
        self.optimizer = torch.optim.AdamW(self.weights, lr=lr, weight_decay=weight_decay)

    def train_step(self, x: Tensor, y: Tensor) -> None:
        loss = cross_entropy(forward(x, *self.weights)[2], y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def logits(self, x: Tensor) -> Tensor:
        return forward(x, *self.weights)[2]

    deployed_logits = logits

    def counts(self) -> tuple[int, int]:
        extra = sum(t.numel() for state in self.optimizer.state.values() for t in state.values())
        return sum(w.numel() for w in self.weights), extra
