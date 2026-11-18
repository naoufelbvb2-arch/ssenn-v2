"""Shared two-affine-layer predictor. Last row of each matrix holds its bias."""

import math
import torch
from torch import Tensor


def initial_weights(inputs: int, hidden: int, outputs: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def sample(n: int, m: int) -> Tensor:
        bound = 1 / math.sqrt(n)
        return torch.empty(n + 1, m).uniform_(-bound, bound, generator=generator)

    return sample(inputs, hidden), sample(hidden, outputs)


def with_bias(x: Tensor) -> Tensor:
    return torch.cat((x, x.new_ones((len(x), 1))), dim=1)


def forward(x: Tensor, first: Tensor, second: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    inputs = with_bias(x)
    hidden = with_bias((inputs @ first).tanh())
    return inputs, hidden, hidden @ second
