"""The eta_max probe of A2 and A0."""
import pytest
import torch

from grok.data import DTYPE
from grok.sgd import linear
from grok.sgd.runner import eta_max_probe

SEEDS = [5, 6, 7]
PROBE = dict(steps=3000, grid_per_decade=200, eta_lo=1e-2, eta_hi=1e5)


def test_gd_edge_is_the_exact_bound_within_one_grid_step():
    ps = linear.build_problems(12, 60, 1.0, 0.0, SEEDS)
    lam = 1e-2
    res = eta_max_probe(ps, 12, "full", None, lam, **PROBE)
    exact = 2 / (ps.kappa_max / 12 + lam)
    assert torch.allclose(torch.tensor(res["formula_gd"], dtype=DTYPE), exact)
    ratio = torch.tensor(res["eta_max_per_seed"], dtype=DTYPE) / exact
    assert (ratio <= 1 + 1e-12).all() and (ratio > 1 / res["grid_ratio"] ** 1.5).all()
    assert res["eta_max"] == min(res["eta_max_per_seed"])


def test_probe_result_is_the_edge_of_a_direct_simulation():
    """Just below the reported value the loss falls, one grid step above it does not."""
    ps = linear.build_problems(12, 60, 1.0, 0.0, SEEDS)
    lam, bs = 1e-2, [11, 12, 13]
    res = eta_max_probe(ps, 1, "replace", bs, lam, **PROBE)
    eta = torch.tensor(res["eta_max_per_seed"], dtype=DTYPE)

    def final_over_initial(e):
        sim = linear.ReducedSGD(ps, e, lam, 1, "replace", bs)
        first = sim.advance(PROBE["steps"])["train_mse"][0]
        return sim.losses(sim.c[None], [0])["train_mse"][0] / first

    assert (final_over_initial(eta) < 1).all()
    above = final_over_initial(eta * res["grid_ratio"])
    assert (~(above < 1)).all()  # not below: larger, infinite or NaN
    # 2 / max |x_i|^2 is a sufficient bound for b = 1, so the measured edge cannot be below it (up to the grid)
    assert (eta * res["grid_ratio"] >= torch.tensor(res["formula_b1"], dtype=DTYPE) * 0.99).all()


def test_probe_refuses_a_bracket_without_the_edge():
    ps = linear.build_problems(12, 60, 1.0, 0.0, SEEDS)
    with pytest.raises(RuntimeError):
        eta_max_probe(ps, 12, "full", None, 1e-2, steps=200, grid_per_decade=50, eta_lo=1e-3, eta_hi=1e-2)
