"""Analysis of extension runs: statistics, crossing times, decision rules and resume."""
import json
import math

import numpy as np
import pytest

from grok.sgd import analysis, linear
from grok.sgd.experiments import predictions_path
from grok.sgd.runner import execute, smoke

BOOT = dict(resamples=2000, seed=0, level=0.95)


def test_paired_relative_counts_censored_seeds_and_brackets_the_mean():
    x = np.array([1.1, 0.9, np.nan, 1.2])
    base = np.array([1.0, 1.0, 1.0, 1.0])
    p = analysis.paired_relative(x, base, BOOT)
    assert (p["pairs"], p["censored"]) == (3, 1)
    assert p["mean"] == pytest.approx((0.1 - 0.1 + 0.2) / 3)
    assert p["max_abs"] == pytest.approx(0.2)
    assert p["ci_low"] <= p["mean"] <= p["ci_high"]


def test_log_linear_interpolation_is_exact_for_an_exponential():
    times = np.array([0.0, 10.0, 20.0, 30.0])
    series = np.exp(-0.1 * times)[:, None]
    cr = analysis.crossings_from_curve(times, series, math.exp(-1.2))
    assert cr["first"][0] == pytest.approx(12.0)
    assert cr["grid"][0] == 20.0
    assert cr["sustained"][0] == pytest.approx(12.0)


def test_first_and_sustained_crossings_differ_for_a_non_monotone_curve():
    times = np.arange(5.0)
    series = np.array([[1.0, 1.0], [0.1, 0.1], [1.0, 0.1], [0.1, 0.1], [0.1, 1.0]])
    cr = analysis.crossings_from_curve(times, series, 0.5)
    assert cr["grid"].tolist() == [1.0, 1.0]
    assert cr["sustained"][0] > 2.0  # recrossed at t = 2
    assert np.isnan(cr["sustained"][1])  # above at the end: censored


def test_null_clock_matches_a_brute_force_search():
    eta, lam, m, c = 1.0, 1e-2, 60, 0.05
    ps = linear.build_problems(12, m, 1.0, 0.0, [3, 4, 5])
    arm = dict(params=dict(eta=eta, lam=lam, m=m), perp=ps.perp.tolist())
    loss = linear.null_loss(ps, eta, lam, list(range(3000))).numpy()  # [T, S]
    brute = [int(np.argmax(loss[:, s] <= c)) for s in range(3)]
    assert (loss.min(axis=0) <= c).all()
    assert analysis.null_clock(arm, c).tolist() == brute


@pytest.fixture(scope="module")
def smoke_a1(tmp_path_factory):
    return smoke("a1", "s", tmp_path_factory.mktemp("runs"))


def test_smoke_analysis_is_stamped_and_checks_the_freeze(smoke_a1):
    s = json.loads((smoke_a1 / "analysis" / "summary.json").read_text())
    md = (smoke_a1 / "analysis" / "summary.md").read_text()
    assert s["smoke"] and s["warning"] == analysis.SMOKE_STAMP
    assert "NOT A RESULT" in md
    assert s["freeze_check"].startswith("the predictions file is the one frozen")
    for f in s["figures"]:
        assert (smoke_a1 / "analysis" / f).stat().st_size > 0


def test_a_tighter_rule_flips_the_verdict_and_a_changed_file_is_flagged(smoke_a1, tmp_path):
    verdicts = {p["id"]: p["verdict"] for p in analysis.analyze(smoke_a1)["predictions"]}
    assert verdicts["p3_matched_epochs"] == "holds"
    frozen = json.loads(predictions_path().read_text())
    frozen["a1"]["p3_matched_epochs"]["tolerance"] = 1e-9  # the smoke gaps are about 0.2%
    path = tmp_path / "tight.json"
    path.write_text(json.dumps(frozen))
    s = analysis.analyze(smoke_a1, predictions=path)
    assert {p["id"]: p["verdict"] for p in s["predictions"]}["p3_matched_epochs"] == "FAILS"
    assert s["freeze_check"].startswith("WARNING")


@pytest.mark.parametrize("experiment", ["a1", "b1"])
def test_interrupted_and_resumed_run_equals_an_uninterrupted_one(experiment, tmp_path):
    cut = smoke(experiment, "cut", tmp_path)
    full = execute(experiment, "full", tmp_path, smoke=True, device="cpu")
    names = sorted(p.name for p in (full / "results").iterdir())
    assert names == sorted(p.name for p in (cut / "results").iterdir())
    for name in names:
        a, b = full / "results" / name, cut / "results" / name
        if name.endswith(".json"):
            assert json.loads(a.read_text()) == json.loads(b.read_text()), name
        else:
            x, y = np.load(a), np.load(b)
            for k in x.files:
                assert np.array_equal(x[k], y[k], equal_nan=True), f"{name}:{k}"
