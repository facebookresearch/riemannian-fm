"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from evtk import hl, vtk
import numpy as np
import torch


def trimesh_to_vtk(filename, v, f, *, cell_data={}, point_data={}):
    v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    f = f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f

    for key, value in cell_data.items():
        cell_data[key] = (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        )

    for key, value in point_data.items():
        point_data[key] = (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        )

    n_cells = f.shape[0]
    hl.unstructuredGridToVTK(
        path=filename,
        x=v[:, 0].copy(order="F"),
        y=v[:, 1].copy(order="F"),
        z=v[:, 2].copy(order="F"),
        connectivity=f.reshape(-1),
        offsets=np.arange(start=3, stop=3 * (n_cells + 1), step=3, dtype="uint32"),
        cell_types=np.ones(n_cells, dtype="uint8") * vtk.VtkTriangle.tid,
        cellData=cell_data,
        pointData=point_data,
    )


def points_to_vtk(filename, pts):
    pts = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else pts
    hl.pointsToVTK(
        filename,
        x=pts[:, 0].copy(order="F"),
        y=pts[:, 1].copy(order="F"),
        z=pts[:, 2].copy(order="F"),
    )
