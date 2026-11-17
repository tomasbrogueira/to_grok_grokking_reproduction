"""Setting A: minibatch SGD on the ridge objective of Eq (1), identity features.

Update (EXTENSION_EXPERIMENTS.md):
  theta <- (1 - eta lam) theta - (eta / b) sum_{i in B} x_i (x_i^T theta - y_i)

Three implementations of the same dynamics:
  DenseSGD     theta in R^m, nothing assumed. The reference; the only place Fact 1 is tested.
  ReducedSGD   theta = V c + theta_perp with V an orthonormal basis of span{x_i}. Only c in R^n
               is simulated, theta_perp follows Fact 1 in closed form. O(b n) per step.
  ClosedFormGD full batch, from grok.spectral.SpectralGD. The exact GD baseline.
All seeds of a problem set are simulated in one batch.
"""
import math
from dataclasses import dataclass

import torch

from ..config import BASE
from ..data import DTYPE, identity_moments
from ..spectral import SpectralGD, signed_power
from .seeds import BatchStream


@dataclass
class ProblemSet:
    seeds: list
    n: int
    m: int
    moms: list  # grok.data.Moments per seed, with Phi, theta0, theta_star
    Z: torch.Tensor  # [S, n, n], Z = X V = U diag(s): training inputs in row-space coordinates
    y: torch.Tensor  # [S, n] training labels, with noise if any
    c0: torch.Tensor  # [S, n] V^T theta_0
    c_star: torch.Tensor  # [S, n] V^T theta*
    perp: torch.Tensor  # [S, 3]: |p0|^2, p0.p*, |p*|^2 with p = P_perp theta
    kappa_min: torch.Tensor  # [S] extreme eigenvalues of K = X X^T
    kappa_max: torch.Tensor
    max_row_norm2: torch.Tensor  # [S] max_i |x_i|^2
    mean_row_norm2: torch.Tensor

    @property
    def S(self):
        return len(self.seeds)

    @property
    def device(self):
        return self.Z.device


def problem_params(n, m, nu2=1.0, noise_std=0.0):
    return {**BASE, "model": "identity", "teacher": "unit_linear", "n": n, "m": m, "nu2": nu2, "noise_std": noise_std}


def build_problems(n, m, nu2, noise_std, seeds, device="cpu"):
    """Draw setting A problems exactly as the replication does (grok.data.identity_moments)."""
    if n >= m:
        raise ValueError("setting A needs n < m")
    p = problem_params(n, m, nu2, noise_std)
    moms, Z, y, c0, cs, perp, kmin, kmax, rmax, rmean = ([] for _ in range(10))
    for s in seeds:
        mom = identity_moments(p, torch.Generator().manual_seed(s), device)
        gd = SpectralGD(mom, 1.0, 0.0)
        if len(gd.ev) != n:
            raise ValueError("training inputs are rank deficient")
        moms.append(mom)
        Z.append(gd.U * gd.s)
        y.append(mom.y)
        c0.append(gd.c0)
        cs.append(gd.c_teacher)
        perp.append(torch.stack(gd.perp))
        kmin.append(gd.ev.min())
        kmax.append(gd.ev.max())
        rn = mom.K.diagonal()
        rmax.append(rn.max())
        rmean.append(rn.mean())
    st = torch.stack
    return ProblemSet(list(seeds), n, m, moms, st(Z), st(y), st(c0), st(cs), st(perp), st(kmin), st(kmax),
                      st(rmax), st(rmean))


def _per_seed(value, S, device):
    """Scalar or per-seed value as a [S, 1] tensor."""
    if torch.is_tensor(value) and value.dtype != DTYPE:
        raise TypeError("per-seed values must be float64 tensors (a float32 step size loses 8 digits)")
    t = torch.as_tensor(value, dtype=DTYPE, device=device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(S)
    if t.numel() != S:
        raise ValueError("per-seed value has the wrong length")
    return t.clone()[:, None]


def null_loss(ps, eta, lam, times):
    """Null-space loss of Fact 1: (1/m) |(1 - eta lam)^t P_perp theta_0 - P_perp theta*|^2. [T, S]."""
    eta, lam = _per_seed(eta, ps.S, ps.device), _per_seed(lam, ps.S, ps.device)
    t = torch.as_tensor(times, dtype=DTYPE, device=ps.device).reshape(-1)
    rho = signed_power((1 - eta * lam)[:, 0], t)  # [T, S]
    a, b, c = ps.perp[:, 0], ps.perp[:, 1], ps.perp[:, 2]
    return (rho**2 * a - 2 * rho * b + c) / ps.m


class ReducedSGD:
    """SGD in the n row-space coordinates. sampling: 'replace', 'reshuffle' or 'full'."""

    def __init__(self, ps, eta, lam, b, sampling, batch_seeds=None):
        self.ps, self.b, self.sampling = ps, b, sampling
        self.eta, self.lam = _per_seed(eta, ps.S, ps.device), _per_seed(lam, ps.S, ps.device)
        self.rho = 1 - self.eta * self.lam
        self.c, self.t = ps.c0.clone(), 0
        if sampling == "full":
            if b != ps.n:
                raise ValueError("full-batch sampling needs b = n")
            self.stream = None
        else:
            self.stream = BatchStream(batch_seeds, ps.n, b, sampling)
        self._rows = torch.arange(ps.S, device=ps.device)[:, None]

    def step(self, idx):
        """One update. idx: [S, b] sample indices, or None for the full batch."""
        ps, c = self.ps, self.c
        if idx is None:
            z, yb, b = ps.Z, ps.y, ps.n
        else:
            z, yb, b = ps.Z[self._rows, idx], ps.y.gather(1, idx), idx.shape[1]
        r = (z @ c[:, :, None])[:, :, 0] - yb
        self.c = self.rho * c - (self.eta / b) * (z.transpose(1, 2) @ r[:, :, None])[:, :, 0]
        self.t += 1

    def losses(self, C, times):
        """Losses of states C [T, S, n] held at the given times."""
        ps = self.ps
        pred = C.transpose(0, 1) @ ps.Z.transpose(1, 2)  # [S, T, n]
        train = ((pred - ps.y[:, None, :]) ** 2).mean(dim=2).T
        row = ((C - ps.c_star[None]) ** 2).sum(dim=2) / ps.m
        return dict(times=times, train_mse=train, row_loss=row, null_loss=null_loss(ps, self.eta, self.lam, times))

    def advance(self, steps):
        """Run `steps` updates. Returns the losses of the states before each update: times t, ..., t+steps-1."""
        ps, t0 = self.ps, self.t
        idx = None if self.stream is None else self.stream.block(t0, t0 + steps).to(ps.device)
        C = torch.empty((steps, ps.S, ps.n), dtype=DTYPE, device=ps.device)
        for k in range(steps):
            C[k] = self.c
            self.step(None if idx is None else idx[k])
        return self.losses(C, list(range(t0, t0 + steps)))

    def state(self):
        return dict(c=self.c, t=self.t)

    def load(self, state):
        self.c, self.t = state["c"].to(self.ps.device), state["t"]


class ClosedFormGD:
    """Exact full-batch GD with the same interface as ReducedSGD.advance. Scalar eta and lam."""

    def __init__(self, ps, eta, lam):
        self.ps, self.eta, self.lam, self.t = ps, float(eta), float(lam), 0
        self.gds = [SpectralGD(mom, self.eta, self.lam) for mom in ps.moms]
        self.stable = [gd.stable for gd in self.gds]

    def advance(self, steps):
        ps, times = self.ps, list(range(self.t, self.t + steps))
        train = torch.stack([gd.train_mse(times) for gd in self.gds], dim=1)
        row = torch.stack([((gd.c_at(times) - gd.c_teacher) ** 2).sum(dim=1) for gd in self.gds], dim=1) / ps.m
        self.t += steps
        return dict(times=times, train_mse=train, row_loss=row, null_loss=null_loss(ps, self.eta, self.lam, times))

    def state(self):
        return dict(t=self.t)

    def load(self, state):
        self.t = state["t"]


def gd_limits(ps, eta, lam):
    """Per-seed t -> infinity limits of exact GD: train MSE, row-space loss, null-space loss, test loss."""
    train, row, null = [], [], []
    for i, mom in enumerate(ps.moms):
        gd = SpectralGD(mom, eta, lam)
        train.append((((gd.cstar * gd.s) @ gd.U.T - mom.y) ** 2).mean())
        row.append(((gd.cstar - gd.c_teacher) ** 2).sum() / ps.m)
        a, b, c = ps.perp[i]
        null.append((c if lam > 0 else a - 2 * b + c) / ps.m)
    train, row, null = torch.stack(train), torch.stack(row), torch.stack(null)
    return dict(train_mse=train, row_loss=row, null_loss=null, test=row + null)


def ridge_residual_norm2(ps, lam):
    """|r|^2 for r = n lam (K + n lam I)^-1 y, the training residual at the ridge solution. [S]."""
    out = []
    for mom in ps.moms:
        n = mom.K.shape[0]
        r = n * lam * torch.linalg.solve(mom.K + n * lam * torch.eye(n, dtype=DTYPE, device=mom.K.device), mom.y)
        out.append(r @ r)
    return torch.stack(out)


class DenseSGD:
    """Reference SGD on theta in R^m. Uses the raw inputs only, no eigen-decomposition."""

    def __init__(self, ps, eta, lam, b, sampling, batch_seeds=None):
        self.ps, self.b, self.sampling = ps, b, sampling
        self.eta, self.lam = _per_seed(eta, ps.S, ps.device), _per_seed(lam, ps.S, ps.device)
        self.X = torch.stack([mom.Phi for mom in ps.moms])  # [S, n, m]
        self.theta = torch.stack([mom.theta0 for mom in ps.moms])
        self.theta0 = self.theta.clone()
        self.theta_star = torch.stack([mom.theta_star for mom in ps.moms])
        self.Q = torch.linalg.qr(self.X.transpose(1, 2)).Q  # [S, m, n] orthonormal basis of span{x_i}
        self.t = 0
        self.stream = None if sampling == "full" else BatchStream(batch_seeds, ps.n, b, sampling)
        self._rows = torch.arange(ps.S, device=ps.device)[:, None]

    def step(self, idx=None):
        if idx is None and self.stream is not None:
            idx = self.stream.block(self.t, self.t + 1)[0].to(self.ps.device)
        if idx is None:
            xb, yb, b = self.X, self.ps.y, self.ps.n
        else:
            xb, yb, b = self.X[self._rows, idx], self.ps.y.gather(1, idx), idx.shape[1]
        r = (xb @ self.theta[:, :, None])[:, :, 0] - yb
        self.theta = (1 - self.eta * self.lam) * self.theta - (self.eta / b) * (xb.transpose(1, 2) @ r[:, :, None])[:, :, 0]
        self.t += 1

    def project_perp(self, v):
        return v - (self.Q @ (self.Q.transpose(1, 2) @ v[:, :, None]))[:, :, 0]

    def train_mse(self):
        return (((self.X @ self.theta[:, :, None])[:, :, 0] - self.ps.y) ** 2).mean(dim=1)

    def test_loss(self):
        return ((self.theta - self.theta_star) ** 2).sum(dim=1) / self.ps.m

    def split_loss(self):
        """Row-space and null-space parts of the test loss, by explicit projection."""
        e = self.theta - self.theta_star
        perp = self.project_perp(e)
        return ((e - perp) ** 2).sum(dim=1) / self.ps.m, (perp**2).sum(dim=1) / self.ps.m

    def fact1_error(self):
        """|P_perp theta_t - (1 - eta lam)^t P_perp theta_0| / |P_perp theta_0| per seed."""
        p0 = self.project_perp(self.theta0)
        rho_t = torch.exp(self.t * torch.log1p(-self.eta * self.lam))
        return (self.project_perp(self.theta) - rho_t * p0).norm(dim=1) / p0.norm(dim=1)


def predicted_t2(n, m, nu2, c, eta, lam):
    """The weight-decay clock: first t with nu^2 (m-n)/m (1 - eta lam)^(2t) <= c."""
    return math.log(nu2 * (m - n) / (m * c)) / (-2 * math.log1p(-eta * lam))
