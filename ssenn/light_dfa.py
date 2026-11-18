"""Fixed-feedback DFA with optional moving anchors and no stiffness allocation."""

import math
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from .credit import Credit
from .network import forward


@torch.no_grad()
def local_drives(x: Tensor, y: Tensor, weights: tuple[Tensor, Tensor],
                 credit: Credit) -> tuple[Tensor, Tensor]:
    inputs, hidden, logits = forward(x, *weights)
    error = one_hot(y, logits.shape[1]) - logits.softmax(dim=1)
    delta = credit.project(error) * (1 - hidden[:, :-1].square())
    return inputs.T @ delta / len(x), hidden.T @ error / len(x)


class LightDFA:
    def __init__(self, weights: tuple[Tensor, Tensor], credit: Credit,
                 eta_w: float, anchor_rate: float | None = None):
        if not math.isfinite(eta_w) or eta_w < 0 or (anchor_rate is not None and not 0 <= anchor_rate <= 1):
            raise ValueError("Invalid DFA rate")
        self.weights = tuple(w.detach().clone() for w in weights)
        self.credit = credit
        self.eta_w = eta_w
        self.anchor_rate = anchor_rate
        self.anchors = None if anchor_rate is None else tuple(w.clone() for w in self.weights)

    @torch.no_grad()
    def train_step(self, x: Tensor, y: Tensor) -> None:
        drives = local_drives(x, y, self.weights, self.credit)
        weights = tuple(w + self.eta_w * d for w, d in zip(self.weights, drives))
        anchors = self.anchors
        if anchors is not None:
            anchors = tuple(a + self.anchor_rate * (w - a) for a, w in zip(anchors, weights))
        if not all(t.isfinite().all() for t in (*weights, *(anchors or ()))):
            raise FloatingPointError("Nonfinite DFA update")
        self.weights, self.anchors = weights, anchors

    @torch.no_grad()
    def logits(self, x: Tensor) -> Tensor:
        return forward(x, *self.weights)[2]

    @torch.no_grad()
    def deployed_logits(self, x: Tensor) -> Tensor:
        return forward(x, *(self.anchors or self.weights))[2]
