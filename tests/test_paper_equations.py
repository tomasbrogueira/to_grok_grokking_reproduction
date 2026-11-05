"""Theorems, bounds and gradients of the paper."""
import math

import numpy as np
import pytest
import torch

from grok import theory
from grok.config import BASE
from grok.data import DTYPE, identity_moments, random_feature_moments, rf_weights, sample_x
from grok.spectral import SpectralGD

DEV = torch.device("cpu")


def params(**kw):
    return {**BASE, **kw}


def test_gradient_matches_autograd_on_eq1():
    g = torch.Generator().manual_seed(0)
    n, m, lam = 8, 20, 0.03
    Phi = torch.randn((n, m), generator=g, dtype=DTYPE)
    star = torch.randn((m,), generator=g, dtype=DTYPE)
    th = torch.randn((m,), generator=g, dtype=DTYPE).requires_grad_()
    obj = ((Phi @ th - Phi @ star) ** 2).sum() / (2 * n) + lam / 2 * th @ th  # Eq (1)
    obj.backward()
    analytic = (Phi.T @ (Phi @ (th.detach() - star))) / n + lam * th.detach()  # Eq (2)
    assert torch.allclose(th.grad, analytic)


def test_theorem_4_1_zero_teacher():
    n, m, lam, eta = 20, 200, 1e-3, 1.0
    p = params(teacher="zero", n=n, m=m, lam=lam, eta=eta)
    mom = identity_moments(p, torch.Generator().manual_seed(3), DEV)
    gd = SpectralGD(mom, eta, lam)
    L = (mom.Phi**2).sum() / n  # L = ||Phi||_F^2 / n, p4
    assert eta <= 2 / (L + 2 * lam)
    th02 = float(mom.theta0 @ mom.theta0)
    ts = [0, 1, 10, 100, 1000, 10000]
    for t, ln, pop, th in zip(ts, gd.Ln(ts), gd.population_loss(ts), gd.theta_at(ts)):
        assert ln <= theory.thm41_train_bound(float(L), n, eta, lam, gd.lam_plus, th02, t) * (1 + 1e-9)
        assert pop >= theory.thm41_test_bound(1 / m, eta, lam, m, n, 1.0, t)
        assert th @ th <= theory.thm41_norm_bound(eta, lam, th02, t) * (1 + 1e-9)


def test_population_loss_matches_monte_carlo():
    p = params(n=20, m=100, lam=1e-3)
    mom = identity_moments(p, torch.Generator().manual_seed(4), DEV)
    gd = SpectralGD(mom, 1.0, 1e-3)
    th = gd.theta_at([500])[0]
    g = torch.Generator().manual_seed(5)
    x = sample_x(400000, 100, "identity_gaussian", g, DEV)
    mc = ((x @ (th - mom.theta_star)) ** 2).mean()
    assert float(mc) == pytest.approx(float(gd.population_loss([500])[0]), rel=0.02)


def test_remark_a14_arithmetic():
    """p22: the steps from Eq (6), (7) to Eq (8)."""
    for m, nu2 in [(1000, 1.0), (3000, 1e6)]:
        assert 6 * 1.5 * (3 * m * nu2 / 2) <= 14 * m * nu2
    for m in (1000, 3000):
        eps, n = 0.01, 100
        inner = (math.sqrt(eps * m) + 1) ** -2  # c/lmin(Sigma) = eps m, ||theta*|| = 1
        assert (m - n) / 2 * inner >= (m - n) / (8 * m * eps)
    for x in np.linspace(1e-9, 0.01, 200):
        assert math.log(1 - x) >= -1.01 * x


def test_eq8_values_at_defaults():
    """Hand values in the plan: 11984 and the MP estimate of about 1513."""
    assert theory.eq8_t2_bound(100, 1000, 1.0, 0.01, 1.0, 1e-4) == pytest.approx(11984, abs=2)
    assert theory.eq8_t1_bound(100, 1000, 1.0, 0.01, 1.0, (1 - math.sqrt(0.1)) ** 2) == pytest.approx(1514, abs=3)


def test_eq8_t1_is_weaker_than_eq6_under_a14():
    assert theory.eq6_t1_bound(100, 1.5, 1.5 * 1000, 0.01, 1.0, 0.5) <= theory.eq8_t1_bound(100, 1000, 1.0, 0.01, 1.0, 0.5)


def test_eq7_vs_eq8():
    """Eq (8) relaxes the log term and tightens 4 to 2.02, per A.14."""
    t7 = theory.eq7_t2_bound(100, 1000, 1.0, 0.01, 1 / 1000, 1.0, 1.0, 1e-4)
    t8 = theory.eq8_t2_bound(100, 1000, 1.0, 0.01, 1.0, 1e-4)
    assert t7 * 4 / 2.02 >= t8


def test_theorem_a9_lower_bound_on_population_loss():
    """p19: L(theta_t) >= lmin(Sigma) ((1-eta lam)^t sqrt((m-n)nu^2/2) - ||theta*||)^2."""
    m, n, lam = 400, 40, 1e-3
    p = params(n=n, m=m, lam=lam)
    gd = SpectralGD(identity_moments(p, torch.Generator().manual_seed(6), DEV), 1.0, lam)
    ts = [0, 10, 100, 1000]
    for t, pop in zip(ts, gd.population_loss(ts)):
        lb = (1 - lam) ** t * math.sqrt((m - n) / 2) - 1
        if lb > 0:
            assert float(pop) >= lb**2 / m


def test_feature_statistics():
    g = torch.Generator().manual_seed(7)
    d, m, nu2 = 50, 5000, 4.0
    X = sample_x(200, d, "gaussian", g, DEV)
    W = rf_weights(m, d, nu2, m, "dm", g, DEV)
    phi = torch.relu(X @ W.T)
    assert float((phi**2).sum(1).mean()) == pytest.approx(nu2 / 2, rel=0.06)  # E||phi||^2 = nu^2/2
    p = params(model="rff", teacher="zero", n=200, m=5000, d=10, n_test=10)
    mom = random_feature_moments(p, torch.Generator().manual_seed(8), DEV)
    assert float(mom.K.diagonal().mean()) == pytest.approx(1.0, rel=0.03)  # RFF E||phi||^2 = 1


def test_nu2_scales_rf_only_through_weights():
    x = params(model="rf_relu", teacher="zero", n=5, m=30, d=5, n_test=5, nu2=1.0)
    a = random_feature_moments(x, torch.Generator().manual_seed(9), DEV)
    b = random_feature_moments({**x, "nu2": 4.0}, torch.Generator().manual_seed(9), DEV)
    assert torch.allclose(b.K, 4 * a.K)  # phi scales with nu, K with nu^2
    assert torch.allclose(b.f0, 2 * a.f0)  # a ~ N(0,1) does not scale with nu


def test_identity_x_scale():
    x = sample_x(5000, 100, "identity_gaussian", torch.Generator().manual_seed(1), DEV)
    assert float((x**2).sum(1).mean()) == pytest.approx(1.0, rel=0.02)  # N(0, I/m)
