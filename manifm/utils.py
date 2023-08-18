"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import torch


def lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    return A.expand(tuple(dimensions) + A.shape)


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,) * len(dimensions)).expand(A.shape + tuple(dimensions))


def cartesian_from_latlon(x):
    """Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.

    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    assert x.shape[-1] == 2
    lat = x.select(-1, 0)
    lon = x.select(-1, 1)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def lonlat_from_cartesian(x):
    r = x.pow(2).sum(-1).sqrt()
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    lat = torch.asin(z / r)
    lon = torch.atan2(y, x)
    return torch.cat([lon.unsqueeze(-1), lat.unsqueeze(-1)], dim=-1)


if __name__ == "__main__":

    torch.manual_seed(0)

    x = torch.randn(2, 3)
    x = x / torch.linalg.norm(x, dim=1, keepdim=True)

    lonlat = lonlat_from_cartesian(x)
    latlon = torch.stack([lonlat[:, 1], lonlat[:, 0]], dim=1)
    x_recon = cartesian_from_latlon(latlon)
    lonlat_recon = lonlat_from_cartesian(x_recon)

    print(x)
    print(lonlat)
    print(x_recon)
    print(lonlat_recon)
