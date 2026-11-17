"""Setting B: minibatch gradients, Fact 2, the ablation, and agreement with the replication's GD."""
import pytest
import torch

from grok import nn
from grok.config import BASE
from grok.data import DTYPE
from grok.sgd import network
from grok.sgd.crossing import crossing_times

DEV = torch.device("cpu")
P = {**BASE, "model": "nn", "teacher": "zero", "n": 12, "m": 30, "d": 6, "lam": 0.05, "nu2": 1.0, "eta": 1e-2,
     "n_test": 40, "max_steps": 200, "checkpoint_every": 50, "b": 12, "sampling": "full"}
NULL = {**P, "n": 4}  # n < d: the first layer has a part the training inputs cannot see
SEEDS, BSEEDS = [1, 2, 3], [7, 8, 9]
TRAIN_THR, TEST_THR = [0.01, 0.02], [0.005, 0.01]
quiet = dict(log=lambda s: None, on_metrics=lambda r: None)


def run(p, path, **kw):
    return network.run_arm(p, SEEDS, BSEEDS, TRAIN_THR, TEST_THR, DEV, path, **quiet, **kw)


def test_minibatch_gradient_matches_autograd_with_repeats():
    net = network.SGDNet(P, SEEDS, DEV)
    mask, h, out = net.forward_train()
    idx = torch.tensor([[0, 5, 5, 11], [3, 3, 3, 3], [1, 2, 4, 8]])
    gw, ga = net.grads_minibatch(mask, h, out, idx)
    W, a = net.W.clone().requires_grad_(), net.a.clone().requires_grad_()
    xb = net.X.gather(1, idx[:, :, None].expand(-1, -1, P["d"]))
    pred = (torch.relu(xb @ W.transpose(1, 2)) * a[:, None, :]).sum(2)
    obj = (pred**2).sum() / (2 * 4) + P["lam"] / 2 * ((W**2).sum() + (a**2).sum())
    obj.backward()
    assert torch.allclose(W.grad, gw, atol=1e-12)
    assert torch.allclose(a.grad, ga, atol=1e-12)
    assert not torch.allclose(gw, net.grads(mask, h, out)[0], atol=1e-6)  # and it is not the full-batch gradient


def test_minibatch_of_all_points_is_the_full_gradient():
    net = network.SGDNet(P, SEEDS, DEV)
    mask, h, out = net.forward_train()
    idx = torch.arange(P["n"]).repeat(3, 1)
    for mine, ref in zip(net.grads_minibatch(mask, h, out, idx), net.grads(mask, h, out)):
        assert torch.allclose(mine, ref, atol=1e-14)


def test_full_batch_arm_follows_the_replication_trajectory(tmp_path):
    p = {**P, "eta": 0.05, "lam": 0.5}  # both crossings happen within about 30 steps
    res, curves = run(p, tmp_path / "a.pt")
    ref = nn.run_condition(p, SEEDS, DEV, tmp_path / "ref.pt", lambda s: None, lambda r: None)
    times = curves["times"].tolist()
    shared = [t for t in times if t in ref["times"]]
    assert len(shared) >= 20 and res["steps"] > ref["steps"]  # this run goes on to c_min / 2
    for t in shared:
        mine = curves["test_mse"][times.index(t)]
        assert torch.equal(mine, torch.tensor(ref["test_mse"][ref["times"].index(t)], dtype=DTYPE))  # bit for bit
        assert torch.equal(curves["train_mse"][times.index(t)], torch.tensor(ref["train_mse"][ref["times"].index(t)], dtype=DTYPE))
    t1 = crossing_times(res["train_tracker"], 0.02)["t1"]  # L_n >= eps is mse >= 2 eps
    t1u = crossing_times(res["train_tracker"], 0.01)["t1"]
    for i in range(3):
        assert 0 < ref["t1"][i] < ref["t1_unhalved"][i] < ref["steps"]
        assert (t1[i], t1u[i]) == (ref["t1"][i], ref["t1_unhalved"][i])
        first = next(t for k, t in enumerate(times) if curves["test_mse"][k, i] <= 0.01)
        assert first == ref["t2"][i]


@pytest.mark.parametrize("b,sampling", [(1, "replace"), (2, "reshuffle"), (4, "full")])
def test_fact_2_w_perp_decays_exactly(tmp_path, b, sampling):
    p = {**NULL, "b": b, "sampling": sampling, "eta": 0.05, "lam": 0.1, "max_steps": 300}
    res, curves = run(p, tmp_path / "a.pt")
    assert res["has_null"]
    assert curves["w_perp_dev"].max().item() < 1e-12
    rho = (1 - 0.05 * 0.1) ** curves["times"].double()
    assert torch.allclose(curves["w_perp_norm"], rho[:, None] * curves["w_perp_norm"][0], rtol=1e-11)
    assert curves["w_perp_norm"][0].min().item() > 0.5  # a real, non-zero component
    assert curves["w_par_norm"][-1].min().item() > 0.1


def test_fact_2_check_fails_without_weight_decay_on_w(tmp_path):
    """Sensitivity: if the penalty on W is dropped from the update, W_perp stops decaying and the check trips."""
    p = {**NULL, "b": 1, "sampling": "replace", "eta": 0.05, "lam": 0.1}
    net = network.SGDNet(p, SEEDS, DEV)
    for _ in range(20):
        mask, h, out = net.forward_train()
        gw, ga = net.grads_minibatch(mask, h, out, torch.zeros((3, 1), dtype=torch.long))
        net.W.add_(gw - p["lam"] * net.W, alpha=-p["eta"])
        net.a.add_(ga, alpha=-p["eta"])
    mask, h, out = net.forward_train()
    assert net.evaluate(20, mask, (out**2).mean(1))["w_perp_dev"].min().item() > 1e-3


def test_training_outputs_depend_only_on_w_par():
    net = network.SGDNet(NULL, SEEDS, DEV)
    _, _, out = net.forward_train()
    net.W = net.W - net.w_perp()
    _, _, out_par = net.forward_train()
    assert torch.allclose(out, out_par, atol=1e-13)


def test_ablated_test_loss_is_the_network_with_w_par():
    net = network.SGDNet(NULL, SEEDS, DEV)
    mask, h, out = net.forward_train()
    ev = net.evaluate(0, mask, (out**2).mean(1))
    full = net.test_mse()
    assert torch.allclose(ev["test_mse"], full)
    W_par = net.W @ net.P
    f = (torch.relu(net.Xt @ net.W.transpose(1, 2)) * net.a[:, None, :]).sum(2)
    f_par = (torch.relu(net.Xt @ W_par.transpose(1, 2)) * net.a[:, None, :]).sum(2)
    assert torch.allclose(ev["test_mse_par"], (f_par**2).mean(1), atol=1e-13)
    assert torch.allclose(ev["null_out_mse"], ((f - f_par) ** 2).mean(1), atol=1e-13)
    flips = ((net.Xt @ net.W.transpose(1, 2) > 0) != (net.Xt @ W_par.transpose(1, 2) > 0)).double().mean((1, 2))
    assert torch.allclose(ev["flip_frac_test"], flips)
    assert (ev["test_mse_par"] - ev["test_mse"]).abs().min().item() > 1e-3  # the ablation changes the loss when n < d
    assert torch.allclose(net.P @ net.P, net.P, atol=1e-12) and torch.allclose(net.X @ net.P, net.X, atol=1e-12)
    assert torch.allclose(ev["w_perp_norm"] ** 2 + ev["w_par_norm"] ** 2, (net.W**2).sum((1, 2)))
    net.chunk = 7
    again = net.evaluate(0, mask, (out**2).mean(1))
    assert all(torch.allclose(ev[k], again[k]) for k in network.EVAL_KEYS)


def test_ablation_is_the_identity_when_inputs_span_the_space():
    net = network.SGDNet(P, SEEDS, DEV)
    mask, h, out = net.forward_train()
    ev = net.evaluate(0, mask, (out**2).mean(1))
    assert not net.has_null
    assert torch.equal(ev["test_mse_par"], ev["test_mse"]) and ev["w_perp_norm"].abs().max() == 0


@pytest.mark.parametrize("stop", [100, 137])
def test_resume_matches_uninterrupted(tmp_path, stop):
    p = {**NULL, "b": 2, "sampling": "replace", "checkpoint_every": 20}
    full_res, full_curves = run(p, tmp_path / "full.pt")
    with pytest.raises(nn.Interrupted):
        run(p, tmp_path / "cut.pt", interrupt_at=stop)
    res, curves = run(p, tmp_path / "cut.pt")
    assert res == full_res
    assert all(torch.equal(curves[k], full_curves[k]) for k in full_curves)


def test_resume_from_a_periodic_checkpoint(tmp_path):
    """Kill between checkpoints: restart from the last periodic one, not from a clean stop."""
    p = {**NULL, "b": 2, "sampling": "replace", "checkpoint_every": 20, "max_steps": 200}
    full_res, full_curves = run(p, tmp_path / "full.pt")
    calls = []

    def die_late(row):
        calls.append(row["t"])
        if row["t"] > 120:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        network.run_arm(p, SEEDS, BSEEDS, TRAIN_THR, TEST_THR, DEV, tmp_path / "cut.pt", lambda s: None, die_late)
    saved = torch.load(tmp_path / "cut.pt", weights_only=False)
    assert 0 < saved["t"] <= 120
    res, curves = run(p, tmp_path / "cut.pt")
    assert res == full_res
    assert all(torch.equal(curves[k], full_curves[k]) for k in full_curves)


def test_batched_seeds_equal_single_seed_runs(tmp_path):
    p = {**NULL, "b": 2, "sampling": "replace"}
    res, curves = run(p, tmp_path / "all.pt")
    one, one_curves = network.run_arm(p, SEEDS[1:2], BSEEDS[1:2], TRAIN_THR, TEST_THR, DEV, tmp_path / "one.pt", **quiet)
    k = min(len(curves["times"]), len(one_curves["times"]))
    assert torch.equal(curves["times"][:5], one_curves["times"][:5])
    assert torch.allclose(curves["test_mse"][:5, 1], one_curves["test_mse"][:5, 0], rtol=1e-10) and k > 5


def test_arms_share_data_and_differ_only_in_batches(tmp_path):
    a, ca = run({**NULL, "b": 1, "sampling": "replace"}, tmp_path / "a.pt")
    b, cb = run({**NULL, "b": 4, "sampling": "full"}, tmp_path / "b.pt")
    assert torch.equal(ca["test_mse"][0], cb["test_mse"][0])  # same data, test set and initialization at t = 0
    assert not torch.allclose(ca["test_mse"][-1], cb["test_mse"][-1], rtol=1e-6)
    assert a["batch_seeds"] == BSEEDS and a["seeds"] == SEEDS


def test_divergence_is_reported_not_hidden(tmp_path):
    res, curves = run({**P, "eta": 50.0}, tmp_path / "a.pt")
    assert res["diverged"] and not res["finished"]
    assert crossing_times(res["train_tracker"], 0.01)["t1"] == [None, None, None]  # censored, not converged
