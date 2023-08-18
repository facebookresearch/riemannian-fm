"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from csv import reader
import random
import numpy as np
import pandas as pd
import igl
import torch
from torch.utils.data import Dataset, DataLoader

from manifm.manifolds import Sphere, FlatTorus, Mesh, SPD, PoincareBall
from manifm.manifolds.mesh import Metric
from manifm.utils import cartesian_from_latlon


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = np.array(list(lines)[1:]).astype(np.float64)
    return dataset


class EarthData(Dataset):
    manifold = Sphere()
    dim = 3

    def __init__(self, dirname, filename):
        filename = os.path.join(dirname, filename)
        dataset = load_csv(filename)
        dataset = torch.Tensor(dataset)
        self.latlon = dataset
        self.data = cartesian_from_latlon(dataset / 180 * np.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Volcano(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "volcano.csv")


class Earthquake(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "earthquake.csv")


class Fire(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "fire.csv")


class Flood(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "flood.csv")


class Top500(Dataset):
    manifold = FlatTorus()
    dim = 2

    def __init__(self, root="data/top500", amino="General"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNA(Dataset):
    manifold = FlatTorus()
    dim = 7

    def __init__(self, root="data/rna"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MeshDataset(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale=1 / 250):
        with open(os.path.join(root, data_file), "rb") as f:
            data = np.load(f)

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()
        self.data = torch.tensor(data).float() * scale

    def manifold(self, *args, **kwargs):
        return Mesh(self.v, self.f, *args, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleBunny(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_simple.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny10(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn009.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny50(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn049.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny100(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn099.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Spot10(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn009.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class Spot50(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn049.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class Spot100(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn099.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class HyperbolicDatasetPair(Dataset):
    manifold = PoincareBall()
    dim = 2

    def __init__(self, distance=0.6, std=0.7):
        self.distance = distance
        self.std = std

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        sign0 = (torch.rand(1) > 0.5).float() * 2 - 1
        sign1 = (torch.rand(1) > 0.5).float() * 2 - 1

        mean0 = torch.tensor([self.distance, self.distance]) * sign0
        mean1 = torch.tensor([-self.distance, self.distance]) * sign1

        x0 = PoincareBall().wrapped_normal(2, mean=mean0, std=self.std)
        x1 = PoincareBall().wrapped_normal(2, mean=mean1, std=self.std)

        return {"x0": x0, "x1": x1}


class MeshDatasetPair(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale: float):
        data = np.load(os.path.join(root, data_file), "rb")
        x0 = data["x0"]
        x1 = data["x1"]

        self.Z0 = float(data["Z0"])
        self.Z1 = float(data["Z1"])

        if "std" in data:
            self.std = float(data["std"])
        else:
            # previous default
            self.std = 1 / 9.5

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()

        self.x0 = torch.tensor(x0).float() * scale
        self.x1 = torch.tensor(x1).float() * scale

    def manifold(self, *args, **kwargs):
        def base_logprob(x):
            x = (x[..., :2] - 0.5) / self.std
            logZ = -0.5 * np.log(2 * np.pi)
            logprob = logZ - x.pow(2) / 2
            logprob = logprob - np.log(self.std)
            return logprob.sum(-1) - np.log(self.Z0)

        mesh = Mesh(self.v, self.f, *args, **kwargs)
        mesh.base_logprob = base_logprob
        return mesh

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        idx0 = int(len(self.x0) * random.random())
        return {"x0": self.x0[idx0], "x1": self.x1[idx]}


class Maze3v2(MeshDatasetPair):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root, data_file="maze_3x3v2.npz", obj_file="maze_3x3.obj", scale=1 / 3
        )


class Maze4v2(MeshDatasetPair):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root, data_file="maze_4x4v2.npz", obj_file="maze_4x4.obj", scale=1 / 4
        )


class Wrapped(Dataset):
    def __init__(
        self,
        manifold,
        dim,
        n_mixtures=1,
        scale=0.2,
        centers=None,
        dataset_size=200000,
    ):
        self.manifold = manifold
        self.dim = dim
        self.n_mixtures = n_mixtures
        if centers is None:
            self.centers = self.manifold.random_uniform(n_mixtures, dim)
        else:
            self.centers = centers
        self.scale = scale
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        del idx

        idx = torch.randint(self.n_mixtures, (1,)).to(self.centers.device)
        mean = self.centers[idx].squeeze(0)

        tangent_vec = torch.randn(self.dim).to(self.centers)
        tangent_vec = self.manifold.proju(mean, tangent_vec)
        tangent_vec = self.scale * tangent_vec
        sample = self.manifold.expmap(mean, tangent_vec)
        return sample


class ExpandDataset(Dataset):
    def __init__(self, dset, expand_factor=1):
        self.dset = dset
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.dset) * self.expand_factor

    def __getitem__(self, idx):
        return self.dset[idx % len(self.dset)]


def _get_dataset(cfg):
    expand_factor = 1
    if cfg.data == "volcano":
        dataset = Volcano(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 1550
    elif cfg.data == "earthquake":
        dataset = Earthquake(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 210
    elif cfg.data == "fire":
        dataset = Fire(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 100
    elif cfg.data == "flood":
        dataset = Flood(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 260
    elif cfg.data == "general":
        dataset = Top500(cfg.top500_datadir, amino="General")
        expand_factor = 1
    elif cfg.data == "glycine":
        dataset = Top500(cfg.top500_datadir, amino="Glycine")
        expand_factor = 10
    elif cfg.data == "proline":
        dataset = Top500(cfg.top500_datadir, amino="Proline")
        expand_factor = 18
    elif cfg.data == "prepro":
        dataset = Top500(cfg.top500_datadir, amino="Pre-Pro")
        expand_factor = 20
    elif cfg.data == "rna":
        dataset = RNA(cfg.rna_datadir)
        expand_factor = 14
    elif cfg.data == "simple_bunny":
        dataset = SimpleBunny(cfg.mesh_datadir)
    elif cfg.data == "bunny10":
        dataset = Bunny10(cfg.mesh_datadir)
    elif cfg.data == "bunny50":
        dataset = Bunny50(cfg.mesh_datadir)
    elif cfg.data == "bunny100":
        dataset = Bunny100(cfg.mesh_datadir)
    elif cfg.data == "spot10":
        dataset = Spot10(cfg.mesh_datadir)
    elif cfg.data == "spot50":
        dataset = Spot50(cfg.mesh_datadir)
    elif cfg.data == "spot100":
        dataset = Spot100(cfg.mesh_datadir)
    elif cfg.data == "maze3v2":
        dataset = Maze3v2(cfg.mesh_datadir)
    elif cfg.data == "maze4v2":
        dataset = Maze4v2(cfg.mesh_datadir)
    elif cfg.data == "wrapped_torus":
        manifold = FlatTorus()
        dataset = Wrapped(
            manifold,
            cfg.wrapped.dim,
            cfg.wrapped.n_mixtures,
            cfg.wrapped.scale,
            dataset_size=200000,
        )
    elif cfg.data == "wrapped_spd":
        manifold = SPD()
        d = cfg.wrapped.dim
        n = manifold.matdim(d)
        manifold = SPD(scale_std=0.5, scale_Id=3.0, base_expmap=False)
        centers = manifold.vectorize(torch.eye(n) * 2.0).reshape(1, -1)

        dataset = Wrapped(
            manifold,
            cfg.wrapped.dim,
            cfg.wrapped.n_mixtures,
            cfg.wrapped.scale,
            centers=centers,
            dataset_size=10000,
        )
    elif cfg.data == "eeg_1":
        dataset = EEG(cfg.eeg_datadir, set="1", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "eeg_2a":
        dataset = EEG(cfg.eeg_datadir, set="2a", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "eeg_2b":
        dataset = EEG(cfg.eeg_datadir, set="2b", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "hyperbolic":
        dataset = HyperbolicDatasetPair()
    else:
        raise ValueError("Unknown dataset option '{name}'")
    return dataset, expand_factor


def get_loaders(cfg):
    dataset, expand_factor = _get_dataset(cfg)

    N = len(dataset)
    N_val = N_test = N // 10
    N_train = N - N_val - N_test

    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    if data_seed is None:
        raise ValueError("seed for data generation must be provided")
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(data_seed),
    )

    # Expand the training set (we optimize based on number of iterations anyway).
    train_set = ExpandDataset(train_set, expand_factor=expand_factor)

    train_loader = DataLoader(
        train_set, cfg.optim.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, cfg.optim.val_batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, cfg.optim.val_batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_manifold(cfg):
    dataset, _ = _get_dataset(cfg)

    if isinstance(dataset, MeshDataset) or isinstance(dataset, MeshDatasetPair):
        manifold = dataset.manifold(
            numeigs=cfg.mesh.numeigs, metric=Metric(cfg.mesh.metric), temp=cfg.mesh.temp
        )
        return manifold, dataset.dim
    else:
        return dataset.manifold, dataset.dim
