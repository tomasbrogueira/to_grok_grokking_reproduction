"""Exact solver against explicit iteration of Eq (2) and the paper's proofs."""
import math

import pytest
import torch

from grok.config import BASE
from grok.data import DTYPE, Moments, identity_moments, random_feature_moments
from grok.spectral import SpectralGD

DEV = torch.device("cpu")


def params(**kw):
    return {**BASE, **kw}


def gd_iterate(Phi, y, theta0, eta, lam, steps):
    n = Phi.shape[0]
    th = theta0.clone()
    out = [th.clone()]
    for _ in range(steps):
        th = th - eta * (Phi.T @ (Phi @ th - y) / n + lam * th)  # Eq (2) for general labels
        out.append(th.clone())
    return torch.stack(out)


def dense_problem(n, m, seed=0, rank=None, noise=0.0):
    g = torch.Generator().manual_seed(seed)
    Phi = torch.randn((n, m), generator=g, dtype=DTYPE) / math.sqrt(m)
    if rank is not None:
        Phi = Phi[:, :rank] @ torch.randn((rank, m), generator=g, dtype=DTYPE) / math.sqrt(rank)
    star = torch.randn((m,), generator=g, dtype=DTYPE)
    star /= star.norm()
    th0 = torch.randn((m,), generator=g, dtype=DTYPE)
    y = Phi @ star + noise * torch.randn((n,), generator=g, dtype=DTYPE)
    Xt = torch.randn((50, m), generator=g, dtype=DTYPE) / math.sqrt(m)
    return Phi, y, th0, star, Xt


@pytest.mark.parametrize("n,m,rank,noise,eta,lam", [
    (10, 40, None, 0.0, 1.0, 1e-2),    # realizable, over-parameterized
    (10, 40, None, 0.3, 1.0, 1e-3),    # noisy labels
    (12, 40, 5, 0.1, 1.0, 1e-3),       # rank-deficient Phi
    (10, 40, None, 0.0, 30.0, 1e-3),   # step size with negative modes
])
def test_equals_explicit_iteration(n, m, rank, noise, eta, lam):
    Phi, y, th0, star, Xt = dense_problem(n, m, rank=rank, noise=noise)
    mom = Moments(K=Phi @ Phi.T, y=y, f0=Phi @ th0, C=Xt @ Phi.T, y_test=Xt @ star, f0_test=Xt @ th0,
                  Phi=Phi, theta0=th0, theta_star=star)
    gd = SpectralGD(mom, eta, lam)
    ts = [0, 1, 2, 3, 7, 50, 400]
    traj = gd_iterate(Phi, y, th0, eta, lam, max(ts))
    assert torch.allclose(gd.theta_at(ts), traj[ts], atol=1e-9, rtol=1e-9)
    mse = ((Phi @ traj[ts].T).T - y).pow(2).mean(1)
    assert torch.allclose(gd.train_mse(ts), mse, atol=1e-9, rtol=1e-9)
    assert torch.allclose(gd.Ln(ts), mse / 2)
    test = ((Xt @ traj[ts].T).T - Xt @ star).pow(2).mean(1)
    assert torch.allclose(gd.test_mse(ts), test, atol=1e-9, rtol=1e-9)
    pop = (traj[ts] - star).pow(2).sum(1) / m
    assert torch.allclose(gd.population_loss(ts), pop, atol=1e-9, rtol=1e-9)


def test_non_realizable_labels_rf_moments_match_dense_features():
    p = params(model="rf_relu", teacher="relu", n=8, m=30, d=5, n_test=20, lam=1e-3)
    mom = random_feature_moments(p, torch.Generator().manual_seed(1), DEV)
    # rebuild the same draws with a dense feature matrix
    g = torch.Generator().manual_seed(1)
    from grok.data import rf_weights, sample_x, unit_vector
    Xtr, Xte = sample_x(8, 5, "gaussian", g, DEV), sample_x(20, 5, "gaussian", g, DEV)
    ws = unit_vector(5, g, DEV)
    W = rf_weights(30, 5, 1.0, 30, "dm", g, DEV)
    a = torch.randn((30,), generator=g, dtype=DTYPE)
    Ptr, Pte = torch.relu(Xtr @ W.T), torch.relu(Xte @ W.T)
    assert torch.allclose(mom.K, Ptr @ Ptr.T)
    assert torch.allclose(mom.C, Pte @ Ptr.T)
    assert torch.allclose(mom.f0, Ptr @ a)
    assert torch.allclose(mom.y, torch.relu(Xtr @ ws))
    gd = SpectralGD(mom, 1.0, 1e-3)
    traj = gd_iterate(Ptr, mom.y, a, 1.0, 1e-3, 300)
    test = ((Pte @ traj[[1, 300]].T).T - mom.y_test).pow(2).mean(1)
    assert torch.allclose(gd.test_mse([1, 300]), test, atol=1e-9, rtol=1e-9)


def test_null_space_decay_and_eq11():
    """p6, p13: perpendicular part decays as (1-eta lam)^t; Eq (11) bounds the parallel part."""
    p = params(teacher="zero", n=10, m=60, lam=1e-3)
    mom = identity_moments(p, torch.Generator().manual_seed(2), DEV)
    gd = SpectralGD(mom, 1.0, 1e-3)
    ts = [0, 5, 50, 500]
    th = gd.theta_at(ts)
    Phi = mom.Phi
    P = Phi.T @ torch.linalg.solve(Phi @ Phi.T, Phi)
    par, perp = th @ P, th - th @ P
    for i, t in enumerate(ts):
        assert torch.allclose(perp[i], (1 - 1e-3) ** t * perp[0], atol=1e-9)
        rate = 1 - gd.lam_plus / 10 - 1e-3
        assert par[i].norm() ** 2 <= rate ** (2 * t) * par[0].norm() ** 2 * (1 + 1e-9)


def test_converges_to_ridge_solution():
    """Theta*_lambda = argmin L_n(theta; lambda), first-order condition on p16."""
    Phi, y, th0, star, _ = dense_problem(10, 40, noise=0.2)
    lam = 1e-2
    mom = Moments(K=Phi @ Phi.T, y=y, f0=Phi @ th0, Phi=Phi, theta0=th0, theta_star=star)
    th = SpectralGD(mom, 1.0, lam).theta_at([10**6])[0]
    closed = torch.linalg.solve(Phi.T @ Phi / 10 + lam * torch.eye(40, dtype=DTYPE), Phi.T @ y / 10)
    assert torch.allclose(th, closed, atol=1e-8)


def test_stability_flag():
    Phi, y, th0, star, _ = dense_problem(10, 40)
    mom = Moments(K=Phi @ Phi.T, y=y, f0=Phi @ th0, Phi=Phi, theta0=th0, theta_star=star)
    assert SpectralGD(mom, 1.0, 1e-3).stable
    assert not SpectralGD(mom, 1e5, 1e-3).stable
