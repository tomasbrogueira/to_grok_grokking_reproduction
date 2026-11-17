"""Setting A simulators: closed form, dense reference, reduced coordinates, Fact 1."""
import math

import pytest
import torch

from grok.data import DTYPE
from grok.runner import linear_seed
from grok.sgd import linear
from grok.sgd.crossing import Tracker, crossing_times
from grok.sgd.seeds import BatchStream
from grok.spectral import SpectralGD

SEEDS = [101, 202, 303]


def problems(n=12, m=60, noise=0.0, nu2=1.0):
    return linear.build_problems(n, m, nu2, noise, SEEDS)


def rel(a, b):
    return ((a - b).abs() / b.abs()).max().item()


def closed_form_losses(ps, eta, lam, times):
    train = torch.stack([SpectralGD(mom, eta, lam).train_mse(times) for mom in ps.moms], 1)
    test = torch.stack([SpectralGD(mom, eta, lam).population_loss(times) for mom in ps.moms], 1)
    return train, test


@pytest.mark.parametrize("noise,eta,lam,steps", [(0.0, 1.0, 1e-3, 400), (0.2, 1.0, 1e-2, 400), (0.0, 3.0, 1e-3, 400),
                                                  (0.0, 3.0, 0.0, 60)])
def test_full_batch_reduced_sgd_equals_exact_gd(noise, eta, lam, steps):
    """With lam = 0 the training loss reaches machine zero, where a relative error means nothing: 60 steps only."""
    ps = problems(noise=noise)
    out = linear.ReducedSGD(ps, eta, lam, ps.n, "full").advance(steps)
    train, test = closed_form_losses(ps, eta, lam, out["times"])
    assert rel(out["train_mse"], train) < 1e-9
    assert rel(out["row_loss"] + out["null_loss"], test) < 1e-12


def test_closed_form_arm_equals_spectral_gd_exactly():
    ps = problems()
    out = linear.ClosedFormGD(ps, 1.0, 1e-3).advance(50)
    train, test = closed_form_losses(ps, 1.0, 1e-3, out["times"])
    assert torch.equal(out["train_mse"], train)
    assert rel(out["row_loss"] + out["null_loss"], test) < 1e-13


def test_sampling_n_points_with_replacement_is_not_gd():
    """The comparison above can fail: b = n drawn with replacement is a different algorithm."""
    ps = problems()
    out = linear.ReducedSGD(ps, 1.0, 1e-3, ps.n, "replace", SEEDS).advance(400)
    train, _ = closed_form_losses(ps, 1.0, 1e-3, out["times"])
    assert rel(out["train_mse"][1:], train[1:]) > 1e-2


@pytest.mark.parametrize("b,sampling", [(1, "replace"), (5, "replace"), (12, "replace"), (4, "reshuffle"), (12, "full")])
def test_reduced_equals_dense_reference(b, sampling):
    ps = problems(noise=0.1)
    red = linear.ReducedSGD(ps, 0.7, 5e-3, b, sampling, SEEDS)
    dense = linear.DenseSGD(ps, 0.7, 5e-3, b, sampling, SEEDS)
    steps = 300
    out = red.advance(steps)
    for t in range(steps):
        if t in (0, 1, 2, 17, 150, 299):
            row, null = dense.split_loss()
            assert rel(out["train_mse"][t], dense.train_mse()) < 1e-9
            assert rel(out["row_loss"][t], row) < 1e-9
            assert rel(out["null_loss"][t], null) < 1e-10
            assert rel(out["row_loss"][t] + out["null_loss"][t], dense.test_loss()) < 1e-10
        dense.step()
    # the parameters themselves: theta = V c + rho^t P_perp theta_0
    for i, mom in enumerate(ps.moms):
        gd = SpectralGD(mom, 1.0, 0.0)
        V = (mom.Phi.T @ gd.U) / gd.s
        theta = V @ red.c[i] + (1 - 0.7 * 5e-3) ** steps * (mom.theta0 - V @ gd.c0)
        assert torch.allclose(theta, dense.theta[i], atol=1e-11)


def test_reduced_and_dense_differ_when_batches_differ():
    """The equality above depends on the batch sequence, so it is not vacuous."""
    ps = problems(noise=0.1)
    a = linear.ReducedSGD(ps, 0.7, 5e-3, 1, "replace", SEEDS).advance(50)
    b = linear.ReducedSGD(ps, 0.7, 5e-3, 1, "replace", [s + 1 for s in SEEDS]).advance(50)
    assert rel(a["train_mse"][1:], b["train_mse"][1:]) > 1e-3


def test_dense_step_is_a_gradient_step_on_the_minibatch_objective():
    ps = problems(noise=0.1)
    eta, lam = 0.3, 0.02
    dense = linear.DenseSGD(ps, eta, lam, 4, "replace", SEEDS)
    idx = torch.tensor([[0, 3, 3, 7], [1, 1, 1, 2], [5, 6, 7, 8]])  # duplicates count twice
    th = dense.theta.clone().requires_grad_()
    xb = dense.X[torch.arange(3)[:, None], idx]
    obj = (((xb @ th[:, :, None])[:, :, 0] - ps.y.gather(1, idx)) ** 2).sum() / (2 * 4) + lam / 2 * (th**2).sum()
    obj.backward()
    dense.step(idx)
    assert torch.allclose(dense.theta, th.detach() - eta * th.grad, atol=1e-13)


def test_mean_of_single_sample_steps_is_the_gd_step():
    ps = problems(noise=0.1)
    eta, lam, n = 0.5, 1e-2, ps.n
    acc = torch.zeros_like(ps.c0)
    for i in range(n):
        sim = linear.ReducedSGD(ps, eta, lam, 1, "replace", SEEDS)
        sim.step(torch.full((ps.S, 1), i))
        acc += sim.c / n
    gd = linear.ReducedSGD(ps, eta, lam, n, "full")
    gd.step(None)
    assert torch.allclose(acc, gd.c, atol=1e-13)


@pytest.mark.parametrize("b,sampling,lam", [(1, "replace", 1e-2), (5, "replace", 1e-2), (4, "reshuffle", 1e-2), (1, "replace", 0.0)])
def test_fact_1_on_the_dense_reference(b, sampling, lam):
    ps = problems(noise=0.1)
    dense = linear.DenseSGD(ps, 1.0, lam, b, sampling, SEEDS)
    worst = 0.0
    for _ in range(500):
        dense.step()
        worst = max(worst, dense.fact1_error().max().item())
    assert worst < 1e-12
    assert dense.project_perp(dense.theta).norm(dim=1).min() > 1e-3  # the identity is not about a zero vector


def test_fact_1_check_fails_when_a_gradient_leaves_the_span():
    ps = problems()
    dense = linear.DenseSGD(ps, 1.0, 1e-2, 1, "replace", SEEDS)
    g = torch.Generator().manual_seed(0)
    dense.X = dense.X + 1e-6 * torch.randn(dense.X.shape, generator=g, dtype=DTYPE)  # Q still spans the old rows
    for _ in range(50):
        dense.step()
    assert dense.fact1_error().max().item() > 1e-8


def test_fact_1_check_fails_for_a_wrong_decay_factor():
    ps = problems()
    dense = linear.DenseSGD(ps, 1.0, 1e-2, 1, "replace", SEEDS)
    for _ in range(50):
        dense.step()
    dense.lam = dense.lam * 1.001  # the reference decay (1 - eta lam)^t now uses a slightly wrong lam
    assert dense.fact1_error().max().item() > 1e-6


def test_null_loss_at_zero_weight_decay_is_constant():
    ps = problems()
    nl = linear.null_loss(ps, 1.0, 0.0, [0, 10, 10**6])
    assert torch.equal(nl[0], nl[1]) and torch.equal(nl[0], nl[2])
    assert nl[0].mean().item() == pytest.approx((ps.m - ps.n) / ps.m, rel=0.25)  # about nu^2 (m-n)/m


def test_gd_limits_and_ridge_residual():
    ps = problems(noise=0.2)
    eta, lam = 1.0, 1e-2
    lim = linear.gd_limits(ps, eta, lam)
    far = linear.ClosedFormGD(ps, eta, lam)
    far.t = 10**6
    out = far.advance(1)
    assert rel(out["train_mse"][0], lim["train_mse"]) < 1e-9
    assert rel(out["row_loss"][0] + out["null_loss"][0], lim["test"]) < 1e-9
    r2 = linear.ridge_residual_norm2(ps, lam)
    assert rel(r2 / ps.n, lim["train_mse"]) < 1e-9  # the limit's training residual is r = n lam (K + n lam I)^-1 y
    for i, mom in enumerate(ps.moms):  # and it is the residual of the ridge minimiser, computed in R^m
        A = mom.Phi.T @ mom.Phi / ps.n + lam * torch.eye(ps.m, dtype=DTYPE)
        theta_hat = torch.linalg.solve(A, mom.Phi.T @ mom.y / ps.n)
        assert ((mom.y - mom.Phi @ theta_hat) ** 2).sum().item() == pytest.approx(r2[i].item(), rel=1e-8)
    zero = linear.gd_limits(ps, eta, 0.0)
    assert rel(zero["null_loss"], linear.null_loss(ps, eta, 0.0, [0])[0]) < 1e-13


def test_closed_form_tracker_reproduces_the_replication_times():
    """Per-step tracking of exact GD gives the t1 and t2 that grok.runner computes by bisection."""
    n, m, lam = 20, 100, 1e-2
    ps = linear.build_problems(n, m, 1.0, 0.0, SEEDS)
    gd = linear.ClosedFormGD(ps, 1.0, lam)
    train, test = Tracker(ps.S, [0.01, 0.02]), Tracker(ps.S, [0.01])
    for _ in range(4):
        out = gd.advance(500)
        train.update(out["times"], out["train_mse"])
        test.update(out["times"], out["row_loss"] + out["null_loss"])
    p = {**linear.problem_params(n, m), "lam": lam, "eta": 1.0}
    for i, s in enumerate(SEEDS):
        ref = linear_seed(p, s, "timing", torch.device("cpu"))
        assert crossing_times(train.export(), 0.02)["t1"][i] == ref["t1"]  # L_n >= eps is mse >= 2 eps
        assert crossing_times(train.export(), 0.01)["t1"][i] == ref["t1_unhalved"]
        assert crossing_times(test.export(), 0.01)["first"][i] == ref["t2"]


def test_predicted_t2_is_the_weight_decay_clock():
    """Exact GD's t2 sits within a few percent of the clock when 1/m is far below c and n lam is small."""
    n, m, lam = 20, 2000, 1e-3
    p = {**linear.problem_params(n, m), "lam": lam, "eta": 1.0}
    pred = linear.predicted_t2(n, m, 1.0, 0.01, 1.0, lam)
    assert pred == pytest.approx(math.log(0.99 / 0.01) / 2e-3, rel=1e-3)
    for s in SEEDS:
        assert linear_seed(p, s, "timing", torch.device("cpu"))["t2"] == pytest.approx(pred, rel=0.05)


def test_per_seed_step_sizes_match_separate_runs():
    ps = problems(noise=0.1)
    etas = [0.3, 0.9, 1.4]
    both = linear.ReducedSGD(ps, torch.tensor(etas, dtype=DTYPE), 1e-2, 2, "replace", SEEDS).advance(100)
    for i, eta in enumerate(etas):
        one = linear.ReducedSGD(ps, eta, 1e-2, 2, "replace", SEEDS).advance(100)
        assert rel(both["train_mse"][:, i], one["train_mse"][:, i]) < 1e-12
        assert rel(both["null_loss"][:, i], one["null_loss"][:, i]) < 1e-12


def test_batched_seeds_equal_single_seed_runs():
    ps = problems(noise=0.1)
    both = linear.ReducedSGD(ps, 1.0, 1e-2, 3, "replace", SEEDS).advance(200)
    solo_ps = linear.build_problems(12, 60, 1.0, 0.1, SEEDS[1:2])
    solo = linear.ReducedSGD(solo_ps, 1.0, 1e-2, 3, "replace", SEEDS[1:2]).advance(200)
    assert rel(both["train_mse"][:, 1], solo["train_mse"][:, 0]) < 1e-10


def test_stream_used_by_the_simulator_is_the_documented_one():
    ps = problems()
    sim = linear.ReducedSGD(ps, 1.0, 1e-2, 2, "replace", SEEDS)
    by_hand = linear.ReducedSGD(ps, 1.0, 1e-2, 2, "replace", SEEDS)
    idx = BatchStream(SEEDS, ps.n, 2, "replace").block(0, 30)
    sim.advance(30)
    for k in range(30):
        by_hand.step(idx[k])
    assert torch.equal(sim.c, by_hand.c)
    assert math.isfinite(sim.c.sum().item())
