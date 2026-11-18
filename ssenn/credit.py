"""Direct projection from output residuals into hidden-unit space."""

from dataclasses import dataclass
from typing import Protocol
import math

import torch
from torch import Tensor


class Credit(Protocol):
    def project(self, residual: Tensor) -> Tensor: ...
    def state_elements(self) -> int: ...


@dataclass(frozen=True)
class DirectFeedback:
    matrix: Tensor

    @classmethod
    def random(cls, outputs: int, hidden: int, seed: int) -> "DirectFeedback":
        generator = torch.Generator().manual_seed(seed)
        bound = 1 / math.sqrt(outputs)
        return cls(torch.empty(outputs, hidden).uniform_(-bound, bound, generator=generator))

    @torch.no_grad()
    def project(self, residual: Tensor) -> Tensor:
        return residual @ self.matrix

    def state_elements(self) -> int:
        return self.matrix.numel()
