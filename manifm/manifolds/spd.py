"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import numpy as np
from geoopt import Manifold
import torch
import scipy.linalg
from torch.func import jacrev, vmap
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class SPD(Manifold):
    """Symmetric Positive Definite matrices.
    Uses the Riemmanian metric from https://indico.ictp.it/event/a08167/session/124/contribution/85/material/0/0.pdf.
    """

    name = "SPD"
    ndim = 0

    def __init__(self, scale_std=0.2, scale_Id=1.0, base_expmap=True, Riem_geodesic=True, Riem_norm=True):
        """
        Riem_geodesic and Riem_norm only affects training and not likelihood evaluation.
        """
        super().__init__()
        self.scale_std = scale_std
        self.scale_Id = scale_Id
        self.base_expmap = base_expmap
        self.Riem_geodesic = Riem_geodesic
        self.Riem_norm = Riem_norm

    def vecdim(self, n):
        return n * (n + 1) // 2

    def matdim(self, d):
        return int((np.sqrt(8 * d + 1) - 1) / 2)

    def vectorize(self, A):
        """Vectorizes a symmetric matrix to a n(n+1)/2 vector."""
        n = A.shape[-1]
        mask = torch.triu(torch.ones(n, n)) == 1
        mask = mask.broadcast_to(A.shape).to(A.device)
        vec = A[mask].reshape(*A.shape[:-2], -1)
        return vec

    def devectorize(self, x):
        size = x.shape
        d = x.shape[-1]
        n = self.matdim(d)
        x = x.reshape(-1, d)

        def create_symm(x):
            A = torch.zeros(n, n).to(x)
            triu_indices = torch.triu_indices(row=n, col=n, offset=0).to(A.device)
            A = torch.index_put(A, (triu_indices[0], triu_indices[1]), x.reshape(-1))
            A = torch.index_put(
                A.mT, (triu_indices[0], triu_indices[1]), x.reshape(-1)
            ).mT
            return A

        A = vmap(create_symm)(x)
        A = A.reshape(*size[:-1], n, n)
        return A

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.proju(x, u)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
        return True, None

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metric_normalized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Normalizes a vector U on the tangent space of X according to G^{-1/2}U."""
        if self.Riem_norm:
            X, U = self.devectorize(x), self.devectorize(u)
            dtype = X.dtype
            X, U = X.double(), U.double()
            X_sqrt = sqrtmh(X)
            U = X_sqrt @ U @ X_sqrt
            return self.vectorize(U).to(dtype)
        else:
            return u

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if self.Riem_norm:
            P = self.devectorize(x)
            A = self.devectorize(u)
            B = self.devectorize(v)
            dtype = P.dtype
            P, A, B = P.double(), A.double(), B.double()
            Pinv_A = torch.linalg.solve(P, A)
            Pinv_B = torch.linalg.solve(P, B)
            return torch.diagonal(torch.matmul(Pinv_A, Pinv_B), dim1=-2, dim2=-1).to(dtype)
        else:
            return torch.sum(u * v, dim=-1, keepdim=keepdim)

    def geodesic(self, x, y, t):
        """Computes the Riemannian geodesic A exp(t log(A^{-1}B)).
        x: (..., D)
        y: (..., D)
        t: (...)
        """
        if self.Riem_geodesic:
            A, B = self.devectorize(x), self.devectorize(y)
            dtype = A.dtype
            A, B = A.double(), B.double()
            Ainv_B = torch.linalg.solve(A, B)
            U = t[..., None, None] * matrix_logarithm(Ainv_B)
            G_t = torch.matmul(A, torch.matrix_exp(U))
            return self.vectorize(G_t).to(dtype)
        else:
            return x + t[..., None] * (y - x)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        A, U = self.devectorize(x), self.devectorize(u)
        dtype = A.dtype
        A, U = A.double(), U.double()

        A_sqrt = sqrtmh(A) + 1e-6 * torch.eye(A.shape[-1]).to(A)

        B = torch.linalg.solve(A_sqrt, torch.linalg.solve(A_sqrt, U), left=False)
        B = torch.linalg.matrix_exp(B)
        B = A_sqrt @ B @ A_sqrt

        b = self.vectorize(B).to(dtype)
        return b

    @torch.no_grad()
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A, B = self.devectorize(x), self.devectorize(y)
        dtype = A.dtype
        A, B = A.double(), B.double()

        A_sqrt = sqrtmh(A) + 1e-6 * torch.eye(A.shape[-1]).to(A)

        U = torch.linalg.solve(A_sqrt, torch.linalg.solve(A_sqrt, B), left=False)
        U = matrix_logarithm(U)
        U = A_sqrt @ U @ A_sqrt

        return self.vectorize(U).to(dtype)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """We are already representing symmetric matrices."""
        return u

    def projx(self, x: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
        """Clamps the eigenvalues to be non-negative."""
        A = self.devectorize(x)
        dtype = A.dtype
        A = A.double()
        L, Q = torch.linalg.eigh(A)
        L = torch.clamp(L, min=threshold)
        P = (Q * L.unsqueeze(-2)) @ Q.mH
        return self.vectorize(P).to(dtype)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor):
        A, B, V = self.devectorize(x), self.devectorize(y), self.devectorize(v)
        dtype = A.dtype
        A, B, V = A.double(), B.double(), V.double()
        A_sqrt = sqrtmh(A)
        B_sqrt = sqrtmh(B)
        E = torch.linalg.solve(B_sqrt, A_sqrt, left=False)
        U = E @ V @ E.mT
        return self.vectorize(U).to(dtype)

    def random_base(self, *size, dtype=None, device=None) -> torch.Tensor:
        bsz = int(np.prod(size[:-1]))
        d = size[-1]
        n = self.matdim(d)

        # Wrap a Gaussian centered at the identity matrix.
        Id = torch.eye(n, dtype=dtype, device=device) * self.scale_Id
        c = self.vectorize(Id).reshape(1, -1).expand(bsz, d)

        # Construct symmetric matrix where elements are iid Normal.
        u = torch.randn(bsz, d).mul_(self.scale_std).to(dtype=dtype, device=device)

        if self.base_expmap:
            # Exponential map to the manifold.
            x = self.expmap(c, u)
        else:
            # Beware this can sample a non-SPD matrix unless scale is small enough.
            x = c + u
        return x.reshape(*size)

    def assert_spd(self, x):
        eigvals = torch.linalg.eigvals(self.devectorize(x)).real
        if eigvals.min() <= 0:
            raise ValueError(f"Matrix not SPD. Smallest eigval is {eigvals.min()}")

    def logdetG(self, x):
        """Log determinant of the metric tensor.

        logdetG = n(n-1)/2 * log(2) + (n+1) * log det A
        """
        A = self.devectorize(x)
        n = A.shape[-1]
        return (n * (n - 1) / 2 * np.log(2.0) + (n + 1) * torch.slogdet(A)[1]).to(x)

    def base_logprob(self, x):
        size = x.shape
        d = x.shape[-1]
        n = self.matdim(d)
        x = x.reshape(-1, d)
        Id = torch.eye(n, dtype=x.dtype, device=x.device) * self.scale_Id
        c = self.vectorize(Id).reshape(1, -1).expand_as(x)

        if self.base_expmap:
            # original N(0, 1) samples
            # print("x finite", torch.isfinite(x).all())
            u = self.logmap(c, x)
            # print("u finite", torch.isfinite(u).all())
            logpu = normal_logprob(u, 0.0, np.log(self.scale_std)).sum(-1)

            # print(u)
            # print("logpu", logpu.shape, logpu.mean())

            # Warning: For some reason, functorch doesn't play well with the sqrtmh implementation.
            with torch.inference_mode(mode=False):

                def logdetjac(f):
                    def _logdetjac(*args):
                        jac = jacrev(f, chunk_size=256)(*args)
                        return torch.linalg.slogdet(jac)[1]

                    return _logdetjac

                # Change of variables in Euclidean space
                ldjs = vmap(logdetjac(self.expmap))(c, u)
                logpu = logpu - ldjs

                # print("ldjs", ldjs.shape, ldjs.mean())
        else:
            u = x - c
            logpu = normal_logprob(u, 0.0, np.log(self.scale_std)).sum(-1)

            # print("logpu", logpu.shape, logpu.mean())

        # Change of metric from Euclidean to Riemannian
        ldgs = self.logdetG(x)

        # print("ldG", ldgs.shape, ldgs.mean())

        logpx = logpu - 0.5 * ldgs
        return logpx.reshape(*size[:-1])


def sqrtmh(A):
    """Compute the square root of a symmetric positive definite matrix."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def matrix_logarithm(A):
    L, V = torch.linalg.eig(A)
    return (V @ torch.diag_embed(torch.log(L + 1e-20)) @ torch.linalg.inv(V)).real


def matrix_logarithm_scipy(A):
    d = A.shape[-1]
    A = A.detach().cpu().numpy().reshape(-1, d, d)
    out = []
    for i in range(A.shape[0]):
        L_i = scipy.linalg.logm(A[i])
        out.append(L_i)
    L = torch.tensor(np.stack(out)).to(A).reshape_as(A)
    return L


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def plot_cone():
    def f(x, y):
        return np.sqrt(x**2 + y**2)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # Can manipulate with 100j and 80j values to make your cone looks different
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 80j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = f(x, y)

    ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.2)

    # Can set your view from different angles.
    ax.view_init(azim=90, elev=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_axis_off()
    plt.show()

    return ax
