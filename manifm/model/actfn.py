"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, t, x):
        return torch.sin(x)


class Softplus(nn.Module):
    def __init__(self, dim=None, beta=100):
        super().__init__()
        self.beta = beta

    def forward(self, t, x):
        return F.softplus(x, beta=self.beta)
