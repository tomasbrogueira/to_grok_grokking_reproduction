import json

import pytest

from grok.config import expand, load_experiment
from grok.nn import Interrupted
from grok.runner import execute, smoke


def test_no_overwrite_and_resume_rules(tmp_path):
    execute("fig2", "r", tmp_path, smoke=True, device="cpu")
    with pytest.raises(FileExistsError):
        execute("fig2", "r", tmp_path, smoke=True, device="cpu")
    with pytest.raises(FileNotFoundError):
        execute("fig2", "other", tmp_path, smoke=True, resume=True, device="cpu")


def test_resume_refuses_changed_config(tmp_path):
    execute("fig2", "r", tmp_path, smoke=True, device="cpu")
    with pytest.raises(ValueError):
        execute("fig2", "r", tmp_path, smoke=True, seeds=1, resume=True, device="cpu")


def test_linear_resume_equals_uninterrupted(tmp_path):
    execute("fig2", "full", tmp_path, smoke=True, device="cpu")
    with pytest.raises(Interrupted):
        execute("fig2", "cut", tmp_path, smoke=True, device="cpu", stop_after_conditions=2)
    execute("fig2", "cut", tmp_path, smoke=True, resume=True, device="cpu")
    for a in (tmp_path / "full" / "results").glob("*.json"):
        b = tmp_path / "cut" / "results" / a.name
        assert json.loads(a.read_text()) == json.loads(b.read_text())


def test_smoke_writes_all_outputs(tmp_path):
    d = smoke("fig1r", "s", tmp_path)
    for name in ("config.yaml", "log.txt", "metrics.jsonl", "results.json"):
        assert (d / name).exists()
    assert json.loads((d / "results.json").read_text())["label"] == "CPU smoke test"
    assert list((d / "checkpoints").glob("*.pt"))


@pytest.mark.parametrize("exp,panel,param,values", [
    ("fig2", "a", "lam", [2e-6, 4e-6, 8e-6, 1.6e-5, 3.2e-5]),
    ("fig2", "b", "n", [10, 20, 40, 80, 160]),
    ("fig2", "c", "m", [1000, 1500, 2000, 2500, 3000]),
    ("fig2", "d", "nu2", [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]),
    ("fig3", "a", "lam", [1e-4, 1e-5, 1e-6]),
    ("fig3", "b", "n", [100, 200, 500]),
    ("fig3", "c", "m", [1000, 10000, 100000]),
    ("fig3", "d", "nu2", [1, 5, 10]),
    ("fig4", "a", "lam", [.05, .1, .2, .4, .8, 1.6]),
    ("fig4", "b", "n", [10, 20, 40, 80, 160]),
    ("fig4", "c", "m", [2000, 2500, 3000, 3500, 4000]),
    ("fig4", "d", "nu2", [.1, .2, .4, .8, 1.6, 3.2, 6.4]),
    ("fig6", "right", "noise_std", [.01, .05, .1]),
])
def test_configs_match_the_reproduction_table(exp, panel, param, values):
    c = load_experiment(exp)
    assert c["panels"][panel]["param"] == param
    assert c["panels"][panel]["values"] == pytest.approx(values)


@pytest.mark.parametrize("exp,seeds,expected", [
    ("fig1l", 50, dict(model="rf_relu", teacher="zero", n=100, m=10000, d=100, lam=1e-5, nu2=1, eta=1)),
    ("fig1r", 50, dict(model="nn", teacher="zero", n=50, m=1000, d=50, lam=0.1, nu2=1, eta=1e-4)),
    ("fig2", 50, dict(model="identity", n=100, m=1000, nu2=1, lam=1e-4, eta=1)),
    ("fig3", 10, dict(model="rf_relu", teacher="relu", d=100, eta=1, n=100, lam=1e-5, m=10000, nu2=1)),
    ("fig4", 50, dict(model="nn", teacher="zero", eta=1e-4, d=50, n=50, m=1000, nu2=1, lam=0.05)),
    ("fig5", 1, dict(m=10000, d=100, n=100, eta=1, lam=1e-5, eps=0.01)),
    ("fig6", 10, dict(m=5000, d=50, n=100, eta=1, lam=1e-5, noise_std=0.1)),
    ("fig7", 10, dict(model="rff", m=5000, d=10, n=100, eta=1, sigma2=1, lam=1e-4)),
    ("x1", 10, dict(model="rff", lam=1e-5)),
    ("x2", 50, dict(model="nn", lam=0.05)),
    ("x3", 50, dict(model="nn", lam=0.1)),
    ("x4", 10, dict(x_dist="unit_sphere", w_scale="m", teacher="relu")),
])
def test_config_defaults_match_the_table(exp, seeds, expected):
    c = load_experiment(exp)
    assert c["seeds"] == seeds
    for k, v in expected.items():
        assert c["defaults"][k] == pytest.approx(v) if not isinstance(v, str) else c["defaults"][k] == v


def test_every_primary_row_expands():
    n = {e: len(expand(load_experiment(e))) for e in ("fig2", "fig3", "fig4")}
    assert n == {"fig2": 22, "fig3": 12, "fig4": 23}
