"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
from geoopt.manifolds import PoincareBall as geoopt_PoincareBall


class PoincareBall(geoopt_PoincareBall):

    def metric_normalized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Normalizes a vector U on the tangent space of X according to G^{-1/2}U."""
        return u / self.lambda_x(x, keepdim=True)

    def random_base(self, *args, **kwargs):
        raise NotImplementedError

    def base_logprob(self, *args, **kwargs):
        raise NotImplementedError
