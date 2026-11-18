"""DFA with a manual tanh derivative and batch-mean local outer products."""

from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from .credit import Credit
from .dynamics import Rates, Spring
from .network import forward


@dataclass(frozen=True)
class FrozenNetwork:
    first: Tensor
    second: Tensor

    @torch.no_grad()
    def logits(self, x: Tensor) -> Tensor:
        return forward(x, self.first, self.second)[2]


class LocalNetwork:
    def __init__(self, weights: tuple[Tensor, Tensor], credit: Credit,
                 rates: Rates, ablated: bool = False):
        self.first, self.second = (Spring.initialize(w) for w in weights)
        self.credit = credit
        self.ablated = ablated
        self.rates = Rates(rates.eta_w, 0, rates.gamma, 0) if ablated else rates

    @torch.no_grad()
    def train_step(self, x: Tensor, y: Tensor) -> None:
        inputs, hidden, logits = forward(x, self.first.w, self.second.w)
        residual = one_hot(y, logits.shape[1]) - logits.softmax(dim=1)
        local_delta = self.credit.project(residual) * (1 - hidden[:, :-1].square())
        self.first, self.second = (
            self.first.advance(inputs.T @ local_delta / len(x), self.rates),
            self.second.advance(hidden.T @ residual / len(x), self.rates),
        )

    @torch.no_grad()
    def logits(self, x: Tensor) -> Tensor:
        return forward(x, self.first.w, self.second.w)[2]

    def deployment_weights(self) -> tuple[Tensor, Tensor]:
        if self.ablated:
            return self.first.w, self.second.w
        return self.first.w0, self.second.w0

    @torch.no_grad()
    def deployed_logits(self, x: Tensor) -> Tensor:
        return forward(x, *self.deployment_weights())[2]

    @torch.no_grad()
    def freeze(self) -> FrozenNetwork:
        a, b = self.deployment_weights()
        return FrozenNetwork(a.clone(), b.clone())

    def counts(self) -> tuple[int, int]:
        deployed = self.first.w.numel() + self.second.w.numel()
        return deployed, 2 * deployed + self.credit.state_elements()
