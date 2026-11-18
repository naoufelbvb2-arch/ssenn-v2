"""Three-state synapses. Both slow laws see the same pre-consolidation stretch."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class Rates:
    eta_w: float
    eta_0: float
    gamma: float
    beta_k: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and x >= 0 for x in
                   (self.eta_w, self.eta_0, self.gamma, self.beta_k)):
            raise ValueError("Rates must be finite and nonnegative")
        if self.eta_0 > 1 or self.gamma > 1:
            raise ValueError("C01 supports anchor and restoring fractions up to one")


@dataclass(frozen=True)
class Spring:
    w: Tensor
    w0: Tensor
    k: Tensor

    @classmethod
    @torch.no_grad()
    def initialize(cls, weight: Tensor) -> "Spring":
        if weight.ndim != 2 or weight.dtype != torch.float32 or weight.device.type != "cpu":
            raise ValueError("Expected a CPU float32 weight matrix")
        if not weight.isfinite().all():
            raise ValueError("Initial weights must be finite")
        return cls(weight.detach().clone(), weight.detach().clone(), torch.zeros_like(weight))

    @torch.no_grad()
    def advance(self, drive: Tensor, rates: Rates) -> "Spring":
        if drive.shape != self.w.shape or drive.dtype != self.w.dtype or drive.device != self.w.device:
            raise ValueError("Drive must match the synapse matrix")
        w = self.w + (rates.eta_w * drive - rates.gamma * self.k * (self.w - self.w0))
        stretch = w - self.w0
        mobile = 1 - self.k
        w0 = self.w0 + rates.eta_0 * mobile * stretch
        stiffness = self.k + rates.beta_k * stretch.abs() * mobile
        if not (w.isfinite().all() and w0.isfinite().all() and stiffness.isfinite().all()):
            raise FloatingPointError("Nonfinite spring update")
        return Spring(w, w0, stiffness.clamp(0, 1))
