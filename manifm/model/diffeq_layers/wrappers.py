"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from inspect import signature
import torch.nn as nn

__all__ = ["diffeq_wrapper", "reshape_wrapper"]


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y):
        if len(signature(self.module.forward).parameters) == 1:
            return self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            return self.module(t, y)
        else:
            raise ValueError(
                "Differential equation needs to either take (t, y) or (y,) as input."
            )

    def __repr__(self):
        return self.module.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class ReshapeDiffEq(nn.Module):
    def __init__(self, input_shape, net):
        super(ReshapeDiffEq, self).__init__()
        assert (
            len(signature(net.forward).parameters) == 2
        ), "use diffeq_wrapper before reshape_wrapper."
        self.input_shape = input_shape
        self.net = net

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


def reshape_wrapper(input_shape, layer):
    return ReshapeDiffEq(input_shape, layer)
