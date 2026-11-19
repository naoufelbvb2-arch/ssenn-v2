"""Conventional comparison for the shared C02-A decoder."""

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from .decoder import Decoder, DecoderConfig


class AdamDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, seed: int, rate: float, decay: float):
        super().__init__()
        self.net = Decoder(cfg, seed, trainable_by_autograd=True)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=rate, weight_decay=decay)

    def forward(self, ids, self_only=False):
        return self.net(ids, self_only)

    def train_step(self, x, y):
        loss = cross_entropy(self(x).flatten(0, 1), y.flatten())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
