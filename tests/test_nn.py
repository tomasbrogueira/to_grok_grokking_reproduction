import math

import pytest
import torch

from grok import nn
from grok.config import BASE

DEV = torch.device("cpu")
P = {**BASE, "model": "nn", "teacher": "zero", "n": 12, "m": 30, "d": 6, "lam": 0.05, "nu2": 1.0,
     "eta": 1e-2, "n_test": 40, "max_steps": 200, "checkpoint_every": 50}


def test_gradient_matches_autograd():
    b = nn.Batch(P, [1, 2, 3], DEV)
    mask, h, out = b.forward_train()
    gw, ga = b.grads(mask, h, out)
    W, a = b.W.clone().requires_grad_(), b.a.clone().requires_grad_()
    pred = (torch.relu(b.X @ W.transpose(1, 2)) * a[:, None, :]).sum(2)
    obj = (pred**2).sum() / (2 * P["n"]) + P["lam"] / 2 * ((W**2).sum() + (a**2).sum())  # Eq (1), all seeds
    obj.backward()
    assert torch.allclose(W.grad, gw, atol=1e-10)
    assert torch.allclose(a.grad, ga, atol=1e-10)


def test_batch_equals_single_seed_runs(tmp_path):
    a = nn.run_condition(P, [1, 2], DEV, tmp_path / "a.pt", lambda s: None, lambda r: None)
    b = nn.run_condition(P, [2], DEV, tmp_path / "b.pt", lambda s: None, lambda r: None)
    assert a["test_mse"][-1][1] == pytest.approx(b["test_mse"][-1][0], rel=1e-10)
    assert a["t1"][1] == b["t1"][0]


def test_init_statistics():
    p = {**P, "m": 4000, "d": 50, "nu2": 4.0, "n": 5, "n_test": 5}
    b = nn.Batch(p, [11], DEV)
    assert float(b.a.var()) == pytest.approx(1 / 4000, rel=0.1)  # a ~ N(0, 1/m)
    assert float(b.W.var()) == pytest.approx(4.0 / 50, rel=0.05)  # w ~ N(0, nu^2/d I_d)


def test_zero_output_layer_gives_pure_weight_decay():
    """With a = 0 the residual vanishes, so only the penalty acts on W, and a stays at 0."""
    b = nn.Batch(P, [5], DEV)
    b.a = torch.zeros_like(b.a)
    mask, h, out = b.forward_train()
    gw, ga = b.grads(mask, h, out)
    assert torch.allclose(gw, P["lam"] * b.W)
    assert torch.count_nonzero(ga) == 0


def test_resume_matches_uninterrupted(tmp_path):
    full = nn.run_condition(P, [1, 2], DEV, tmp_path / "full.pt", lambda s: None, lambda r: None)
    with pytest.raises(nn.Interrupted):
        nn.run_condition(P, [1, 2], DEV, tmp_path / "cut.pt", lambda s: None, lambda r: None, interrupt_at=100)
    res = nn.run_condition(P, [1, 2], DEV, tmp_path / "cut.pt", lambda s: None, lambda r: None)
    assert res["t1"] == full["t1"] and res["t2"] == full["t2"]
    assert res["times"] == full["times"]
    assert res["test_mse"] == full["test_mse"]


def test_chunking_gives_same_test_loss():
    b = nn.Batch(P, [1, 2], DEV)
    whole = b.test_mse()
    b.chunk = 7
    assert torch.allclose(whole, b.test_mse())


def test_next_eval_spacing():
    assert nn.next_eval(0, False) == 1
    assert nn.next_eval(1000, False) == 1020
    assert nn.next_eval(1000, True) == 1005
    assert math.isclose((nn.next_eval(10**5, True) - 10**5) / 10**5, 0.005)
