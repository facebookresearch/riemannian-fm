"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from enum import Enum
import scipy as sp
import numpy as np
import igl
from geoopt import Manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import scipy.sparse

import torch
from torch.func import vjp, vmap
from torchdiffeq import odeint

from manifm.solvers import projx_integrator


class Metric(Enum):
    DIFFUSION = "diffusion"
    BIHARMONIC = "biharmonic"
    COMMUTETIME = "commutetime"
    HEAT = "heat"


class Mesh(Manifold):
    ndim = 1
    name = "Mesh"
    reversible = False

    def __init__(
        self,
        v,
        f,
        numeigs=100,
        metric=Metric.COMMUTETIME,
        temp=1.0,
        dirichlet_bc: bool = False,
        upsample: int = 0,
    ):
        super().__init__()

        if upsample > 0:
            v_np, f_np = v.cpu().numpy(), f.cpu().numpy()
            v_np, f_np = igl.upsample(v_np, f_np, upsample)
            v, f = torch.tensor(v_np).to(v), torch.tensor(f_np).to(f)

        v_np, f_np = v.cpu().numpy(), f.cpu().numpy()
        self.register_buffer(
            "areas", torch.tensor(igl.doublearea(v_np, f_np)).reshape(-1) / 2
        )

        self.register_buffer("v", v)
        self.register_buffer("f", f)

        print("#vertices: ", v.shape[0], "#faces: ", f.shape[0])

        self.numeigs = numeigs
        self.metric = metric
        self.temp = temp
        self.dirichlet_bc = dirichlet_bc

        self._preprocess_eigenfunctions()

    def _preprocess_eigenfunctions(self):
        assert (
            self.numeigs <= self.v.shape[0]
        ), "Cannot compute more eigenvalues than the number of vertices."

        ## ---- in numpy ----  ##
        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        M = igl.massmatrix(v_np, f_np, igl.MASSMATRIX_TYPE_VORONOI)
        L = -igl.cotmatrix(v_np, f_np)

        if self.dirichlet_bc:
            b = igl.boundary_facets(f_np)
            b = np.unique(b.flatten())
            L = scipy.sparse.csr_matrix(L)
            csr_rows_set_nz_to_val(L, b, 0)
            for i in b:
                L[i, i] = 1.0

        eigvals, eigfns = sp.sparse.linalg.eigsh(
            L, self.numeigs + 1, M, sigma=0, which="LM", maxiter=100000
        )
        # Remove the zero eigenvalue.
        eigvals = eigvals[..., 1:]
        eigfns = eigfns[..., 1:]
        ## ---- end in numpy ----  ##

        self.register_buffer("eigvals", torch.tensor(eigvals).to(self.v))
        self.register_buffer("eigfns", torch.tensor(eigfns).to(self.v))

        print(
            "largest eigval: ",
            self.eigvals.max().item(),
            ", smallest eigval: ",
            self.eigvals.min().item(),
        )

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = closest_point(x, self.v, self.f)
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # determine which face the point is on
        _, f_idx = closest_point(x, self.v, self.f)
        vs = self.v[self.f[f_idx]]

        # compute normal for each face
        n = face_normal(a=vs[:, 0], b=vs[:, 1], c=vs[:, 2])

        # project by removing the normal direction
        u = u - (n * u).sum(-1, keepdim=True) * n
        return u

    def dist(self, x: torch.Tensor, y: torch.Tensor, squared: bool = False):
        eigfns_x = self.get_eigfns(x)
        eigfns_y = self.get_eigfns(y)

        fn = (lambda x: x) if squared else torch.sqrt

        if self.metric == Metric.BIHARMONIC:
            return fn(torch.sum(((eigfns_x - eigfns_y) / self.eigvals) ** 2, axis=1))
        elif self.metric == Metric.DIFFUSION:
            return fn(
                torch.sum(
                    torch.exp(-2 * self.temp * self.eigvals)
                    * (eigfns_x - eigfns_y) ** 2,
                    axis=1,
                )
            )
        elif self.metric == Metric.COMMUTETIME:
            return fn(torch.sum((eigfns_x - eigfns_y) ** 2 / self.eigvals, axis=1))
        elif self.metric == Metric.HEAT:
            k = torch.sum(
                eigfns_x * eigfns_y * torch.exp(-self.temp * self.eigvals), axis=1
            )
            dist = fn(-4 * self.temp * torch.log(k))
            return dist
        else:
            return ValueError(f"Unknown distance type option, metric={self.metric}.")

    def get_eigfns(self, x: torch.Tensor):
        """x, y : (N, 3) torch.Tensor representing points on the mesh."""

        N = x.shape[0]

        _, ix = closest_point(x, self.v, self.f)

        # compute barycentric coordinates
        vfx = self.v[self.f[ix]]
        vfx_a, vfx_b, vfx_c = vfx[..., 0, :], vfx[..., 1, :], vfx[..., 2, :]
        bc_x = barycenter_coordinates(x, vfx_a, vfx_b, vfx_c)[..., None]  # (N, 3, 1)

        # compute interpolated eigenfunction
        eigfns = torch.sum(bc_x * self.eigfns[self.f[ix]], dim=-2)

        return eigfns

    @torch.no_grad()
    def solve_path(
        self, x0, x1, t, projx=True, squared=False, method="euler", **kwargs
    ):
        """
        Inputs:
            x0 : (N, 3) Tensors on the mesh. The starting point.
            x1 : (N, 3) Tensors on the mesh. The end point.
            t: (T,) Tensor of time values between 0 and 1 (inclusive).
            projx: Bool. If true, projects x onto the mesh after every step.
        Outputs:
            xt : (T, N, 3) Tensors on the path.
            ut : (T, N, 3) Tensors on the tangent plane of xt. The vector field at xt that transports from x0 to x1.
        """

        orig_dist = self.dist(x0, x1, squared=squared)

        nfe = [0]

        def odefunc(t, x):
            del t
            nfe[0] += 1

            d, vjp_fn = vjp(lambda x: self.dist(x, x1, squared=squared), x)
            dgradx = vjp_fn(torch.ones_like(d))[0]

            dx = (
                -orig_dist[..., None]
                * dgradx
                / torch.linalg.norm(dgradx, dim=-1, keepdim=True)
                .pow(2)
                .clamp(min=1e-20)
            )
            return dx

        if method not in ["euler", "midpoint", "rk4"]:
            assert not projx, "projection not compatible with odeint, set projx=False"
            xt = odeint(odefunc, x0, t, method=method, **kwargs)
            shape = xt.shape
            vt = vmap(lambda x: odefunc(None, x))(xt)
            return xt.reshape(*shape), vt.reshape(*shape)
        else:
            return projx_integrator(self, odefunc, x0, t, method=method, local_coords=False, projx=projx)

    def random_uniform(self, n_samples, dim=3):
        assert dim == 3

        f_idx = torch.multinomial(self.areas, n_samples, replacement=True)
        barycoords = sample_simplex_uniform(
            2, (n_samples,), dtype=self.v.dtype, device=self.v.device
        )
        return torch.sum(self.v[self.f[f_idx]] * barycoords[..., None], axis=1)

    def uniform_logprob(self, x):
        tot_area = torch.sum(self.areas)
        return torch.full_like(x[..., 0], -torch.log(tot_area))

    def random_base(self, *args, **kwargs):
        return self.random_uniform(*args, **kwargs)

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.proju(x, u)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
        return True, None

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        return torch.sum(u * v, dim=-1, keepdim=keepdim)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def plot_trimesh(
    v, f, c=None, *, cmap=plt.cm.viridis, ax=None, elev=90, azim=-90, roll=0, **kwargs
):
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    ax.view_init(elev=elev, azim=azim, roll=roll)

    v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    f = f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f

    nv = v.shape[0]
    if c is not None:
        c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c
        if c.shape[0] == nv:
            # Turn values defined on vertices into values defined on faces.
            c = np.mean(c[f], axis=1)
        norm = plt.Normalize(c.min(), c.max())
        colors = cmap(norm(c))
    else:
        colors = None

    # Default values
    edgecolors = kwargs.pop("edgecolors", "black")
    linewidths = kwargs.pop("linewidths", 0.1)

    pc = art3d.Poly3DCollection(
        v[f], facecolors=colors, edgecolors=edgecolors, linewidths=linewidths, **kwargs
    )
    ax.add_collection(pc)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    ax.set_xlim([v.min(), v.max()])
    ax.set_ylim([v.min(), v.max()])
    ax.set_zlim([v.min(), v.max()])

    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def plot_data(data, *, ax=None, **kwargs):
    data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    ax.scatter(x, y, z, **kwargs)
    return ax


def plot_hist(data, v, f, **kwargs):
    data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    f = f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f

    numfaces = f.shape[0]
    face_idx = igl.signed_distance(data, v, f)[1]
    nsamples = np.bincount(face_idx, minlength=numfaces)
    ax = plot_trimesh(v, f, nsamples, **kwargs)
    return ax


def push_outward(points, v, f, eps=0.1):
    """Moves the points to be just a bit outside of the manifold.
    For better plotting.
    """
    _, f_idx, _ = igl.signed_distance(points, v, f)
    n = igl.per_face_normals(v, f, np.zeros(3))
    points = points + eps * n[f_idx]
    return points


def face_normals(v, f):
    """Computes normal vectors for every face on the mesh."""
    vs = v[f]
    return face_normal(a=vs[:, 0, :], b=vs[:, 1, :], c=vs[:, 2, :])


def face_normal(a, b, c):
    """Computes face normal based on three vertices. Ordering matters.

    Inputs:
        a, b, c: (N, 3)
    """
    u = b - a
    v = c - a
    n = torch.linalg.cross(u, v)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    return n


def project_edge(p, a, b):
    x = p - a
    v = b - a
    r = torch.sum(x * v, dim=-1, keepdim=True) / torch.sum(v * v, dim=-1, keepdim=True)
    r = r.clamp(max=1.0, min=0.0)
    projx = v * r
    return projx + a


def closest_point(p, v, f):
    """Returns the point on the mesh closest to the query point p.
    Algorithm follows https://www.youtube.com/watch?v=9MPr_XcLQuw&t=204s.

    Inputs:
        p : (#query, 3)
        v : (#vertices, 3)
        f : (#faces, 3)

    Return:
        A projected tensor of size (#query, 3) and an index (#query,) indicating the closest triangle.
    """

    orig_p = p

    nq = p.shape[0]
    nf = f.shape[0]

    vs = v[f]
    a, b, c = vs[:, 0], vs[:, 1], vs[:, 2]

    # calculate normal of each triangle
    n = face_normal(a, b, c)

    n = n.reshape(1, nf, 3)
    p = p.reshape(nq, 1, 3)

    a = a.reshape(1, nf, 3)
    b = b.reshape(1, nf, 3)
    c = c.reshape(1, nf, 3)

    # project onto the plane of each triangle
    p = p + (n * (a - p)).sum(-1, keepdim=True) * n

    # if barycenter coordinate is negative,
    # then point is outside of the edge on the opposite side of the vertex.
    bc = barycenter_coordinates(p, a, b, c)

    # for each outside edge, project point onto edge.
    p = torch.where((bc[..., 0] < 0)[..., None], project_edge(p, b, c), p)
    p = torch.where((bc[..., 1] < 0)[..., None], project_edge(p, c, a), p)
    p = torch.where((bc[..., 2] < 0)[..., None], project_edge(p, a, b), p)

    # compute distance to all points and take the closest one
    idx = torch.argmin(torch.linalg.norm(orig_p[:, None] - p, dim=-1), dim=-1)
    p_idx = vmap(lambda p_, idx_: torch.index_select(p_, 0, idx_))(
        p, idx.reshape(-1, 1)
    ).reshape(nq, 3)
    return p_idx, idx


def barycenter_coordinates(p, a, b, c):
    """Assumes inputs are (N, D).
    Follows https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row] : csr.indptr[row + 1]] = value


def csr_rows_set_nz_to_val(csr, rows, value=0):
    for row in rows:
        csr_row_set_nz_to_val(csr, row)
    if value == 0:
        csr.eliminate_zeros()


def sample_simplex_uniform(K, shape=(), dtype=torch.float32, device="cpu"):
    x = torch.sort(torch.rand(shape + (K,), dtype=dtype, device=device))[0]
    x = torch.cat(
        [
            torch.zeros(*shape, 1, dtype=dtype, device=device),
            x,
            torch.ones(*shape, 1, dtype=dtype, device=device),
        ],
        dim=-1,
    )
    diffs = x[..., 1:] - x[..., :-1]
    return diffs


if __name__ == "__main__":
    elev_azim_roll = {
        "zebra": (0, 180, 0),
        "bunny": (90, -90, 0),
        "buddha2": (0, -90, 0),
    }

    def test_closest_point():
        v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 2]])
        f = np.array([[0, 1, 2], [1, 3, 2]])

        p = np.random.randn(100, 3)

        # closest point computed by libigl
        _, _, x_np = igl.signed_distance(p, v, f)

        ax = plot_trimesh(v, f, alpha=0.4)
        plot_data(p, ax=ax, alpha=0.7, marker="x")
        ax.set_aspect("equal")

        v = torch.tensor(v)
        f = torch.tensor(f)
        p = torch.tensor(p)

        x, _ = closest_point(p, v, f)
        print(
            "closest_point implementation coincides with igl:",
            sp.linalg.norm(x.numpy() - x_np),
        )

        plot_data(x.numpy(), ax=ax, alpha=0.7)
        plt.savefig("closest_point.png")
        plt.close()

    def test_solve_path():
        M = 8
        T = 201
        device = torch.device("cuda:0")

        v, _, _, f, _, _ = igl.read_obj("../../data/mesh/buddha2.obj")
        print("original", v.shape, f.shape)
        # _, v, f, _, _ = igl.decimate(v, f, 2000)
        # print("decimated", v.shape, f.shape)

        v_, f_ = v, f
        v = torch.tensor(v).to(device)
        f = torch.tensor(f).to(device)

        fig = plt.figure(figsize=(20, 5))
        axs = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]

        kwargs = {"alpha": 0.02, "linewidths": 0.0, "antialiased": False}

        # bunny
        plot_trimesh(v_, f_, ax=axs[0], elev=90, azim=-90, roll=0, **kwargs)
        plot_trimesh(v_, f_, ax=axs[1], elev=0, azim=-90, roll=0, **kwargs)
        plot_trimesh(v_, f_, ax=axs[2], elev=180, azim=-90, roll=0, **kwargs)
        plot_trimesh(v_, f_, ax=axs[3], elev=180, azim=-90, roll=0, **kwargs)

        # buddha2
        plot_trimesh(v_, f_, ax=axs[0], elev=0, azim=-90, roll=0, **kwargs)
        plot_trimesh(v_, f_, ax=axs[1], elev=0, azim=0, roll=0, **kwargs)
        plot_trimesh(v_, f_, ax=axs[2], elev=0, azim=90, roll=0, **kwargs)
        plot_trimesh(v_, f_, ax=axs[3], elev=0, azim=180, roll=0, **kwargs)

        mesh = Mesh(v, f, metric=Metric.BIHARMONIC)

        x0 = mesh.random_uniform(M).reshape(M, 3).to(device)
        x1 = mesh.random_uniform(M).reshape(M, 3).to(device)

        with torch.no_grad():
            xt, vt = mesh.solve_path(x0, x1, t=torch.linspace(0, 1, T))

        # test that vt is on the tangent plane
        vt = vt.reshape(-1, 3)
        vt_ = mesh.proju(xt.reshape(-1, 3), vt)
        print("vt is on tangent plane:", torch.linalg.norm(vt_ - vt))

        # plot xt
        xt = xt.cpu()
        for i in range(M):
            for ax in axs:
                plot_data(
                    xt[:, i].detach().cpu().numpy(), ax=ax, s=3, c=np.linspace(0, 1, T)
                )

        plt.tight_layout()
        plt.savefig("solve_path.png")
        plt.close()

    test_closest_point()
    test_solve_path()
