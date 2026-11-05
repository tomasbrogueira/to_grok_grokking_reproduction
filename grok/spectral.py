"""Exact gradient descent on Eq (1), without iterating Eq (2).

With Phi = U S V^T, GD on the 1/2-scaled ridge objective acts mode by mode:
  row space:  c(t) = c* + (c0 - c*) r^t,  r = 1 - eta (s^2/n + lam),
              c* = (s/n) (U^T y) / (s^2/n + lam)
  null space: theta_perp(t) = (1 - eta lam)^t theta_perp(0).
tests/test_spectral.py proves this equals explicit iteration of Eq (2).
"""
import torch

from .data import Moments


def signed_power(base, t):
    """base^t for integer t >= 0 tensors, accurate for base close to 1 or negative."""
    t = t.to(base.dtype)
    out = torch.exp(t[:, None] * torch.log1p(torch.abs(base)[None, :] - 1))
    negative = (base < 0)[None, :] & (torch.remainder(t, 2)[:, None] == 1)
    return torch.where(negative, -out, out)


class SpectralGD:
    def __init__(self, mom: Moments, eta, lam, rank_tol=1e-11):
        self.mom, self.eta, self.lam = mom, eta, lam
        self.n = mom.K.shape[0]
        evals, U = torch.linalg.eigh(mom.K)
        keep = evals > rank_tol * evals.max().clamp_min(1e-300)
        self.evals_all = evals
        self.ev, self.U = evals[keep], U[:, keep]
        self.s = torch.sqrt(self.ev)
        h = self.ev / self.n + lam
        self.c0 = (self.U.T @ mom.f0) / self.s
        self.cstar = (self.s / self.n) * (self.U.T @ mom.y) / h
        self.r = 1 - eta * h
        self.rho = torch.as_tensor(1 - eta * lam, dtype=self.ev.dtype, device=self.ev.device)
        self.lam_plus = float(self.ev.min())  # lambda^+_min(Phi^T Phi)
        self.lam_max = float(self.ev.max())
        self.stable = bool((self.r.abs() < 1).all())
        self.B = None
        if mom.C is not None:
            self.B = (mom.C @ self.U) / self.s
        if mom.theta0 is not None:
            V = (mom.Phi.T @ self.U) / self.s
            self.c_teacher = V.T @ mom.theta_star
            p0 = mom.theta0 - V @ self.c0
            ps = mom.theta_star - V @ self.c_teacher
            self.perp = (p0 @ p0, p0 @ ps, ps @ ps)

    def _t(self, t):
        return torch.as_tensor(t, dtype=self.ev.dtype, device=self.ev.device).reshape(-1)

    def c_at(self, t):
        return self.cstar + (self.c0 - self.cstar) * signed_power(self.r, self._t(t))

    def rho_at(self, t):
        return signed_power(self.rho.reshape(1), self._t(t))[:, 0]

    def train_pred(self, t):
        return (self.c_at(t) * self.s) @ self.U.T

    def train_mse(self, t):
        """Unhalved training MSE, as plotted."""
        return ((self.train_pred(t) - self.mom.y) ** 2).mean(dim=1)

    def Ln(self, t):
        """L_n of Section 2, with its factor 1/2."""
        return 0.5 * self.train_mse(t)

    def test_pred(self, t):
        rho = self.rho_at(t)[:, None]
        perp_part = self.mom.f0_test - self.B @ self.c0
        return rho * perp_part[None, :] + self.c_at(t) @ self.B.T

    def test_mse(self, t, chunk=200):
        t = self._t(t)
        out = [((self.test_pred(t[i:i + chunk]) - self.mom.y_test) ** 2).mean(dim=1) for i in range(0, len(t), chunk)]
        return torch.cat(out)

    def population_loss(self, t):
        """Exact L(theta_t) for phi ~ N(0, I/m): ||theta_t - theta*||^2 / m."""
        m = self.mom.theta0.shape[0]
        rho = self.rho_at(t)
        par = ((self.c_at(t) - self.c_teacher) ** 2).sum(dim=1)
        a, b, c = self.perp
        return (par + rho**2 * a - 2 * rho * b + c) / m

    def generalization_loss(self, t):
        return self.population_loss(t) if self.mom.theta0 is not None else self.test_mse(t)

    def accuracy(self, t, eps):
        """Surrogate accuracy of Appendix B on the training and test samples."""
        train = ((self.train_pred(t) - self.mom.y) ** 2 <= eps).to(self.ev.dtype).mean(dim=1)
        test = ((self.test_pred(t) - self.mom.y_test) ** 2 <= eps).to(self.ev.dtype).mean(dim=1)
        return train, test

    def theta_at(self, t):
        """Full parameter vector, identity features only. Used by tests."""
        V = (self.mom.Phi.T @ self.U) / self.s
        par = self.c_at(t) @ V.T
        return par + self.rho_at(t)[:, None] * (self.mom.theta0 - V @ self.c0)[None, :]
