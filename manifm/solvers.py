"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
from tqdm import tqdm
from manifm.manifolds import Euclidean


@torch.no_grad()
def projx_integrator(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }[method]

    xts = [x0]
    vts = []

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    xt = x0
    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
        vts.append(vt)
        xts.append(xt)
    vts.append(odefunc(t1, xt))
    return torch.stack(xts), torch.stack(vts)


@torch.no_grad()
def projx_integrator_return_last(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
    return xt


def euler_step(odefunc, xt, vt, t0, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


def midpoint_step(odefunc, xt, vt, t0, dt, manifold=None):
    half_dt = 0.5 * dt
    if manifold is not None:
        x_mid = xt + half_dt * vt
        v_mid = odefunc(t0 + half_dt, x_mid)
        v_mid = manifold.transp(x_mid, xt, v_mid)
        return manifold.expmap(xt, dt * v_mid)
    else:
        x_mid = xt + half_dt * vt
        return xt + dt * odefunc(t0 + half_dt, x_mid)


def rk4_step(odefunc, xt, vt, t0, dt, manifold=None):
    k1 = vt
    if manifold is not None:
        raise NotImplementedError
    else:
        k2 = odefunc(t0 + dt / 3, xt + dt * k1 / 3)
        k3 = odefunc(t0 + dt * 2 / 3, xt + dt * (k2 - k1 / 3))
        k4 = odefunc(t0 + dt, xt + dt * (k1 - k2 + k3))
        return xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125
