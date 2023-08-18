"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
from geoopt import Euclidean
import torch


class FlatTorus(Euclidean):
    """Represents a flat torus on the [0, 2pi]^D subspace.

    Isometric to the product of 1-D spheres."""

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (x + u) % (2 * math.pi)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x % (2 * math.pi)

    def random_uniform(self, *size, dtype=None, device=None) -> torch.Tensor:
        z = torch.rand(*size, dtype=dtype, device=device)
        return z * 2 * math.pi

    def uniform_logprob(self, x):
        dim = x.shape[-1]
        return torch.full_like(x[..., 0], -dim * math.log(2 * math.pi))

    def random_base(self, *args, **kwargs):
        return self.random_uniform(*args, **kwargs)

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)
