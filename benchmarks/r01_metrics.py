"""Read-only diagnostics for the C01 direction grammar."""

from collections.abc import Callable
import hashlib
import json

import torch
from torch import Tensor
import torch.nn.functional as F

from ssenn.data import CLAUSE_LENGTH, Examples
from ssenn.local import LocalNetwork
from ssenn.network import forward


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def positions(n_docs: int, clauses: int, context: int) -> Tensor:
    return torch.arange(context, clauses * CLAUSE_LENGTH).repeat(n_docs) % CLAUSE_LENGTH


def score(logits: Tensor, y: Tensor) -> dict:
    ce = F.cross_entropy(logits, y).item()
    if not torch.isfinite(torch.tensor(ce)):
        raise FloatingPointError("Nonfinite cross-entropy")
    return {"n": len(y), "ce": ce, "accuracy": (logits.argmax(1) == y).float().mean().item()}


@torch.no_grad()
def breakdown(logits: Tensor, y: Tensor, phase: Tensor) -> dict:
    return {"overall": score(logits, y),
            "positions": [score(logits[phase == p], y[phase == p]) for p in range(CLAUSE_LENGTH)]}


@torch.no_grad()
def flip_examples(x: Tensor, y: Tensor, vocab: dict[str, int]) -> Examples:
    """For position 2, hold the starting symbol and earlier history fixed."""
    width = len(vocab) + 1
    ids = x.reshape(len(x), -1, width).argmax(-1)
    direction = ids[:, -2]
    up, down = vocab["up"], vocab["down"]
    if not ((direction == up) | (direction == down)).all():
        raise ValueError("Expected a direction at context offset -2")
    ring = torch.tensor([vocab[c] for c in "abcdef"])
    start = (ids[:, -1, None] == ring).long().argmax(1)
    sign = torch.where(direction == up, 1, -1)
    if not torch.equal(ring[(start + sign) % len(ring)], y):
        raise ValueError("Invalid direction-dependent target")
    ids[:, -2] = torch.where(direction == up, down, up)
    return F.one_hot(ids, width).flatten(1).float(), ring[(start - sign) % len(ring)]


@torch.no_grad()
def flip_accuracy(predict: Callable[[Tensor], Tensor], original: Examples, flipped: Examples) -> float:
    x, y = original
    changed_x, changed_y = flipped
    return ((predict(x).argmax(1) == y) &
            (predict(changed_x).argmax(1) == changed_y)).float().mean().item()


@torch.no_grad()
def credit_cosine(model: LocalNetwork, batch: Examples) -> float:
    """Diagnostic at the fast state. Neither matrix below is used in training."""
    x, y = batch
    inputs, h, logits = forward(x, model.first.w, model.second.w)
    residual = F.one_hot(y, logits.shape[1]) - logits.softmax(1)
    slope = 1 - h[:, :-1].square()
    dfa = inputs.T @ (model.credit.project(residual) * slope)
    exact = inputs.T @ ((residual @ model.second.w[:-1].T) * slope)
    return F.cosine_similarity(dfa.flatten(), exact.flatten(), dim=0).item()
