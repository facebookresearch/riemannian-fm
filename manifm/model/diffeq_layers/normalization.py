"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import torch
import torch.nn as nn

__all__ = ["BatchNorm"]


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        basis = Legendre(basis_size=33, start=0.0, end=1.0, memoize_basis=False)
        self.affine = ContinuousAffine(
            basis=basis, shape=(None, num_features, None, None), dims=(1,)
        )

        # TODO: add running means and variances.

        self.rm_nonself_grads = False

    def set_rm_nonself_grads_(self, bool):
        self.rm_nonself_grads = bool

    def forward(self, t, x):
        c = x.size(1)  # hard-coded to always normalize all but the 2nd dim.

        # compute batch statistics
        x_t = x.transpose(0, 1).contiguous().view(c, -1)
        batch_mean = torch.mean(x_t, dim=1)
        batch_var = torch.var(x_t, dim=1)

        # for numerical issues
        batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

        bias = -batch_mean
        weight = -0.5 * torch.log(batch_var)

        shape = [1] * len(x.shape)
        shape[1] = -1

        bias = bias.view(*shape).expand_as(x)
        weight = weight.view(*shape).expand_as(x)

        if self.rm_nonself_grads:
            y = DetachedNormalize.apply(x, bias, torch.exp(weight))
        else:
            y = (x + bias) * torch.exp(weight)

        # Apply time-dependent affine transformation.
        return self.affine(t, y)

    def __repr__(self):
        return "{name}({num_features})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class DetachedNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, weight):
        y = (x + bias) * weight
        ctx.save_for_backward(x, bias, weight)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, bias, weight = ctx.saved_tensors

        # "Gradients" are calculated as if bias and weight don't depend on x.
        grad_x = weight
        grad_bias = weight
        grad_weight = x + bias

        return grad_x, grad_bias, grad_weight


class AffineBase(torch.nn.Module):
    def __init__(self, shape, dims, extra_linear_shape):
        super(AffineBase, self).__init__()

        check_shape_and_dims(shape, dims)
        for i, s in enumerate(shape):
            if i in dims and not isinstance(s, int):
                raise ValueError(
                    "The {}-th dimension is a dimension to compute an affine transformation over, so an "
                    "integer value for its size must be provided.".format(i)
                )

        self.shape = shape
        self.dims = dims
        linear_shape = [
            s if i in dims else 1 for i, s in enumerate(shape)
        ] + extra_linear_shape
        self.weight = torch.nn.Parameter(torch.ones(linear_shape))
        self.bias = torch.nn.Parameter(torch.zeros(linear_shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.reset_weight()
        torch.nn.init.zeros_(self.bias)

    def reset_weight(self):
        raise NotImplementedError

    def _affine(self, x, weight, bias):
        check_input_compat(x.shape, self.shape)
        out = x * weight
        out += bias
        return out


class Affine(AffineBase):
    def __init__(self, **kwargs):
        super(Affine, self).__init__(**kwargs, extra_linear_shape=[])

    def reset_weight(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x):
        return self._affine(x, self.weight, self.bias)


class ContinuousAffine(AffineBase):
    def __init__(self, basis, **kwargs):
        super(ContinuousAffine, self).__init__(
            **kwargs, extra_linear_shape=[basis.basis_size]
        )
        self.basis = basis

    def reset_weight(self):
        with torch.no_grad():
            self.weight[..., 1:].zero_()
            self.weight[..., 0].fill_(1.0)

    def forward(self, t, x):
        basis = self.basis(t)
        weight = self.weight @ basis
        bias = self.bias @ basis
        return self._affine(x, weight, bias)


def check_shape_and_dims(shape, dims):
    for s in shape:
        if not (s is None or isinstance(s, int)):
            raise ValueError("shape must be an iterable of just Nones and integers.")

    for d in dims:
        if not isinstance(d, int) or d < 0:
            raise ValueError("dims must be an iterable of non-negative integers.")


def check_input_compat(input_shape, expected_shape):
    if len(input_shape) != len(expected_shape):
        raise ValueError(
            "Expected shape compatible with {}, instead given {}.".format(
                expected_shape, input_shape
            )
        )
    for given_dim, expected_dim in zip(input_shape, expected_shape):
        if isinstance(expected_dim, int) and given_dim != expected_dim:
            raise ValueError(
                "Expected shape compatible with {}, instead given {}.".format(
                    expected_shape, input_shape
                )
            )


class _BufferDict(torch.nn.Module):
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except torch.nn.modules.module.ModuleAttributeError as e:
            raise KeyError from e

    def __setitem__(self, key, value):
        self.register_buffer(key, value)


class Basis(torch.nn.Module):
    def __init__(self, memoize_basis):
        super(Basis, self).__init__()
        self.memoize_basis = memoize_basis
        if memoize_basis:
            self._memo = _BufferDict()

    def forward(self, t):
        if self.memoize_basis:
            return self._forward_memoize(t)
        else:
            return self._forward(t)

    def _forward(self, t):
        raise NotImplementedError

    def _forward_memoize(self, t):
        if t.numel() == 1:
            t_item = str(t.item()).replace(".", "_")
            try:
                result = self._memo[t_item]
            except KeyError:
                result = self._forward(t)
                self._memo[t_item] = result
            return result
        else:
            return self._forward(t)


class _UncheckedAssign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scratch, value, index):
        ctx.index = index
        scratch.data[index] = value  # sneak past the version checker
        return scratch

    @staticmethod
    def backward(ctx, grad_scratch):
        return grad_scratch, grad_scratch[ctx.index], None


class Legendre(Basis):
    def __init__(self, basis_size, start, end, **kwargs):
        super(Legendre, self).__init__(**kwargs)

        self.basis_size = basis_size
        self.start = start
        self.end = end

        self._scale = 2 / (self.end - self.start)
        self._offset = -(self.start + self.end) / (self.end - self.start)

        self.register_buffer(
            "arange", torch.arange(1, 2 * basis_size, 2, dtype=torch.float32).sqrt()
        )

    def _forward(self, t):
        scaled_t = self._scale * t + self._offset

        # With help from https://github.com/goroda/PyTorchPoly/blob/master/poly.py
        retvar = torch.empty(*t.shape, self.basis_size, dtype=t.dtype, device=t.device)
        retvar[..., 0] = 1
        if self.basis_size > 1:
            retvar = _UncheckedAssign.apply(retvar, scaled_t, (..., 1))
            for i in range(1, self.basis_size - 1):
                value = (
                    (2 * i + 1) * scaled_t * retvar[..., i] - i * retvar[..., i - 1]
                ) / (i + 1)
                retvar = _UncheckedAssign.apply(retvar, value, (..., i + 1))

        return self.arange * retvar


class Fourier(Basis):
    def __init__(self, basis_size, start, end, **kwargs):
        super(Fourier, self).__init__(**kwargs)
        if (basis_size % 2) == 0:
            raise ValueError("Fourier bases must be have an odd `basis_size`.")

        self.basis_size = basis_size
        self.start = start
        self.end = end

        self.register_buffer(
            "arange",
            torch.arange(1, (basis_size + 1) // 2) * 2 * math.pi / (end - start),
        )

    def _forward(self, t):
        # t can have arbitrary shape
        scaled_t = (t - self.start).unsqueeze(-1) * self.arange
        return torch.cat(
            [
                torch.ones(*t.shape, 1, dtype=t.dtype, device=t.device),
                scaled_t.cos(),
                scaled_t.sin(),
            ],
            dim=-1,
        )
