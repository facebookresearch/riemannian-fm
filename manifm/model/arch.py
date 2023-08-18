"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import torch
import torch.nn as nn

import manifm.model.diffeq_layers as diffeq_layers
from manifm.model.actfn import Sine, Softplus
from manifm.manifolds.mesh import Mesh, closest_point, face_normal
from manifm.manifolds import SPD


ACTFNS = {
    "swish": diffeq_layers.TimeDependentSwish,
    "sine": Sine,
    "srelu": Softplus,
}


def tMLP(d_in, d_out=None, d_model=256, num_layers=6, actfn="swish", fourier=None):
    assert num_layers > 1, "No weak linear nets here"
    d_out = d_in if d_out is None else d_out
    actfn = ACTFNS[actfn]
    if fourier:
        layers = [
            diffeq_layers.diffeq_wrapper(
                PositionalEncoding(n_fourier_features=fourier)
            ),
            diffeq_layers.ConcatLinear_v2(d_in * fourier * 2, d_model),
        ]
    else:
        layers = [diffeq_layers.ConcatLinear_v2(d_in, d_model)]

    for _ in range(num_layers - 2):
        layers.append(actfn(d_model))
        layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_model))
    layers.append(actfn(d_model))
    layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_out))
    return diffeq_layers.SequentialDiffEq(*layers)


class PositionalEncoding(nn.Module):
    """Assumes input is in [0, 2pi]."""

    def __init__(self, n_fourier_features):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x):
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)


class Unbatch(nn.Module):
    def __init__(self, vecfield):
        super().__init__()
        self.vecfield = vecfield

    def forward(self, t, x):
        has_batch = x.ndim > 1
        if not has_batch:
            x = x.reshape(1, -1)
            t = t.reshape(-1)
        v = self.vecfield(t, x)
        if not has_batch:
            v = v[0]
        return v


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield, manifold, metric_normalize):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold
        self.metric_normalize = metric_normalize

    def forward(self, t, x):
        if isinstance(self.manifold, Mesh):
            # Memory-efficient implementation for meshes.
            with torch.no_grad():
                _, f_idx = closest_point(x, self.manifold.v, self.manifold.f)
                vs = self.manifold.v[self.manifold.f[f_idx]]
                n = face_normal(a=vs[:, 0], b=vs[:, 1], c=vs[:, 2])
            x = x + (n * (vs[:, 0] - x)).sum(-1, keepdim=True) * n
            v = self.vecfield(t, x)
            v = v - (n * v).sum(-1, keepdim=True) * n
        if isinstance(self.manifold, SPD):
            # projx is expensive and we can just skip it since it doesn't affect divergence.
            v = self.vecfield(t, x)
            v = self.manifold.proju(x, v)
        else:
            x = self.manifold.projx(x)
            v = self.vecfield(t, x)
            v = self.manifold.proju(x, v)

        if self.metric_normalize and hasattr(self.manifold, "metric_normalized"):
            v = self.manifold.metric_normalized(x, v)

        return v


if __name__ == "__main__":
    print(diffeq_layers.ConcatLinear_v2(3, 64))

    import torch

    model = tMLP(d_in=3, d_model=64, num_layers=3)
    t = torch.randn(2, 1)
    x = torch.randn(2, 3)

    print(model(t, x))
