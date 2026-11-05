"""Features, teachers and inputs, as defined in the paper's Sections 5.1, 5.2 and Appendix B.

Linear problems are reduced to moments, so m=1e5 needs no m-by-n matrix in memory:
K = Phi Phi^T, C = Phi_test Phi^T, and the initial predictions.
"""
import math
from dataclasses import dataclass

import torch

DTYPE = torch.float64
BLOCK = 2000


def randn(shape, gen, device):
    return torch.randn(shape, generator=gen, dtype=DTYPE).to(device)


def rand(shape, gen, device):
    return torch.rand(shape, generator=gen, dtype=DTYPE).to(device)


def sample_x(n, d, dist, gen, device):
    x = randn((n, d), gen, device)
    if dist == "gaussian":
        return x
    if dist == "identity_gaussian":  # phi(x)=x ~ N(0, I_m/m), Section 5.1
        return x / math.sqrt(d)
    unit = x / x.norm(dim=1, keepdim=True)
    if dist == "unit_sphere":
        return unit
    if dist == "sphere_sqrt_d":
        return unit * math.sqrt(d)
    raise ValueError(dist)


def unit_vector(d, gen, device):
    v = randn((d,), gen, device)
    return v / v.norm()


def rf_weights(b, d, nu2, m, w_scale, gen, device):
    """w_j ~ N(0, nu^2/(dm) I_d), Section 5.2. w_scale='m' is diagnostic X4 only."""
    var = {"dm": nu2 / (d * m), "m": nu2 / m}[w_scale]
    return randn((b, d), gen, device) * math.sqrt(var)


@dataclass
class Moments:
    K: torch.Tensor
    y: torch.Tensor  # training labels, including any noise
    f0: torch.Tensor  # Phi theta_0
    C: torch.Tensor | None = None  # Phi_test Phi^T
    y_test: torch.Tensor | None = None
    f0_test: torch.Tensor | None = None
    # exact population terms for phi(x)=x with Sigma = I/m (A.14)
    Phi: torch.Tensor | None = None
    theta0: torch.Tensor | None = None
    theta_star: torch.Tensor | None = None


def _add_noise(y, p, gen, device):
    if p["noise_std"] > 0:
        y = y + p["noise_std"] * randn(y.shape, gen, device)
    return y


def identity_moments(p, gen, device):
    n, m = p["n"], p["m"]
    X = sample_x(n, m, "identity_gaussian", gen, device)
    theta0 = randn((m,), gen, device) * math.sqrt(p["nu2"])
    if p["teacher"] == "unit_linear":
        theta_star = unit_vector(m, gen, device)
    elif p["teacher"] == "zero":
        theta_star = torch.zeros(m, dtype=DTYPE, device=device)
    else:
        raise ValueError(p["teacher"])
    y = _add_noise(X @ theta_star, p, gen, device)
    return Moments(K=X @ X.T, y=y, f0=X @ theta0, Phi=X, theta0=theta0, theta_star=theta_star)


def random_feature_moments(p, gen, device):
    n, m, d, nt = p["n"], p["m"], p["d"], p["n_test"]
    Xtr = sample_x(n, d, p["x_dist"], gen, device)
    Xte = sample_x(nt, d, p["x_dist"], gen, device)
    y = torch.zeros(n, dtype=DTYPE, device=device)
    yt = torch.zeros(nt, dtype=DTYPE, device=device)
    if p["teacher"] == "relu":
        w_star = unit_vector(d, gen, device)
        y, yt = torch.relu(Xtr @ w_star), torch.relu(Xte @ w_star)
    elif p["teacher"] == "fourier_independent":
        w_star = randn((d,), gen, device) * math.sqrt(p["sigma2"])
        b_star = rand((1,), gen, device) * 2 * math.pi
        scale = math.sqrt(2 / m)
        y, yt = scale * torch.cos(Xtr @ w_star + b_star), scale * torch.cos(Xte @ w_star + b_star)
    elif p["teacher"] not in ("zero", "feature"):
        raise ValueError(p["teacher"])

    K = torch.zeros((n, n), dtype=DTYPE, device=device)
    C = torch.zeros((nt, n), dtype=DTYPE, device=device)
    f0 = torch.zeros(n, dtype=DTYPE, device=device)
    f0t = torch.zeros(nt, dtype=DTYPE, device=device)
    for start in range(0, m, BLOCK):
        b = min(BLOCK, m - start)
        if p["model"] == "rf_relu":
            W = rf_weights(b, d, p["nu2"], m, p["w_scale"], gen, device)
            Ptr, Pte = torch.relu(Xtr @ W.T), torch.relu(Xte @ W.T)
            theta0 = randn((b,), gen, device)  # a_j ~ N(0,1), Section 5.2
        elif p["model"] == "rff":
            W = randn((b, d), gen, device) * math.sqrt(p["sigma2"])
            phase = rand((b,), gen, device) * 2 * math.pi
            scale = math.sqrt(2 / m)
            Ptr = scale * torch.cos(Xtr @ W.T + phase)
            Pte = scale * torch.cos(Xte @ W.T + phase)
            theta0 = randn((b,), gen, device) * math.sqrt(p["nu2"])
            if p["teacher"] == "feature" and start == 0:  # theta* = e_1 (U5)
                y, yt = Ptr[:, 0].clone(), Pte[:, 0].clone()
        else:
            raise ValueError(p["model"])
        K += Ptr @ Ptr.T
        C += Pte @ Ptr.T
        f0 += Ptr @ theta0
        f0t += Pte @ theta0
    y = _add_noise(y, p, gen, device)
    return Moments(K=K, y=y, f0=f0, C=C, y_test=yt, f0_test=f0t)


def moments(p, gen, device):
    return identity_moments(p, gen, device) if p["model"] == "identity" else random_feature_moments(p, gen, device)
