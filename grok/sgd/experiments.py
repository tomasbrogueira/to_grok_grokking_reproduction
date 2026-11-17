"""The arms of A1-A3 and B1-B3, built from configs/extension/<name>.yaml.

An arm is one training setup. Arms of the same group share data seeds (paired seeds); each
arm has its own batch seeds. `baseline` names the full-batch arm an arm is compared with.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import BASE, ROOT
from . import network
from .seeds import batch_seed, data_seed

CONFIG_DIR = ROOT / "configs" / "extension"
EXPERIMENTS = ("a0", "a1", "a2", "a3", "b1", "b2", "b3")


@dataclass
class Arm:
    key: str
    setting: str  # "A" ridge regression, "B" two-layer network
    group: str
    params: dict
    baseline: str | None = None


def load_config(name, smoke=False):
    """Read a config. With smoke, the keys under `smoke:` replace the top-level ones."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no extension experiment called {name}")
    cfg = yaml.safe_load(path.read_text())
    over = cfg.pop("smoke", {})
    if smoke:
        cfg.update(over)
    cfg["smoke"] = bool(smoke)
    return cfg


def train_thresholds(thresholds):
    """Training MSE levels to track: eps for the unhalved loss and 2 eps for L_n = MSE / 2 (U2)."""
    return sorted({float(x) for e in thresholds for x in (e, 2 * e)})


def kappa_bar_min(n, m):
    return (1 - math.sqrt(n / m)) ** 2


def kappa_bar_max(n, m):
    return (1 + math.sqrt(n / m)) ** 2


def _linear(cfg, prob, **kw):
    p = dict(n=prob["n"], m=prob["m"], nu2=prob["nu2"], noise_std=prob.get("noise_std", 0.0),
             horizon_factor=cfg["horizon_factor"], cap_factor=cfg["cap_factor"],
             checkpoint_every=cfg["checkpoint_every"], tail_window=bool(cfg.get("tail_window", False)),
             eta_fraction=None, lam_ref=None)
    p.update(kw)
    if p["lam_ref"] is None:
        p["lam_ref"] = p["lam"]
    return p


def arms_a1(cfg):
    prob, eta, lam, n = cfg["problem"], cfg["eta"], cfg["lam"], cfg["problem"]["n"]
    arms = [Arm("gd", "A", "all", _linear(cfg, prob, eta=eta, lam=lam, b=n, sampling="full"))]
    for b in cfg["batch_sizes"]:
        arms.append(Arm(f"steps_b{b}", "A", "all", _linear(cfg, prob, eta=eta, lam=lam, b=b, sampling="replace"), "gd"))
    for b in cfg["batch_sizes"]:  # matched epochs: eta = b / n times the GD step size
        arms.append(Arm(f"epochs_b{b}", "A", "all",
                        _linear(cfg, prob, eta=eta * b / n, lam=lam, b=b, sampling="replace"), "gd"))
    b = cfg["reshuffle_b"]
    arms.append(Arm(f"reshuffle_b{b}", "A", "all", _linear(cfg, prob, eta=eta, lam=lam, b=b, sampling="reshuffle"), "gd"))
    arms.append(Arm("nowd_gd", "A", "all", _linear(cfg, prob, eta=eta, lam=0.0, b=n, sampling="full", lam_ref=lam)))
    arms.append(Arm("nowd_b1", "A", "all", _linear(cfg, prob, eta=eta, lam=0.0, b=1, sampling="replace", lam_ref=lam),
                    "nowd_gd"))
    return arms


def a2_batch_sizes(cfg, n):
    """Batch sizes of A2 at one n: the listed ones up to n, then b = n, which is exact full-batch GD."""
    return sorted({b for b in cfg["batch_sizes"] if b < n}) + [n]


def arms_a2(cfg):
    """The step size of every arm is a fraction of eta_max(n, b), measured by the probe before the arms run."""
    arms = []
    for n in cfg["ns"]:
        prob = dict(n=n, m=cfg["m"], nu2=cfg["nu2"])
        for frac in cfg["eta_fractions"]:
            for b in a2_batch_sizes(cfg, n):
                p = _linear(cfg, prob, eta=None, lam=cfg["lam"], b=b, sampling="full" if b == n else "replace",
                            eta_fraction=frac)
                arms.append(Arm(f"n{n}_f{frac}_b{b}", "A", f"n={n}", p, None if b == n else f"n{n}_f{frac}_b{n}"))
    return arms


def arms_a3(cfg):
    prob, n = cfg["problem"], cfg["problem"]["n"]
    arms = []
    for ratio in cfg["ratios"]:
        lam = ratio * kappa_bar_min(n, prob["m"]) / n
        for sigma in cfg["sigmas"]:
            pr = dict(prob, noise_std=sigma)
            tag = f"r{ratio}_s{sigma}"
            arms.append(Arm(f"{tag}_gd", "A", "all", _linear(cfg, pr, eta=cfg["eta"], lam=lam, b=n, sampling="full",
                                                           ratio=ratio)))
            for b in cfg["batch_sizes"]:
                arms.append(Arm(f"{tag}_b{b}", "A", "all", _linear(cfg, pr, eta=cfg["eta"], lam=lam, b=b,
                                                                  sampling="replace", ratio=ratio), f"{tag}_gd"))
    return arms


def _network(cfg, model, b, sampling):
    p = {**BASE, "model": "nn", "teacher": "zero", **model, "b": b, "sampling": sampling,
         "checkpoint_every": cfg["checkpoint_every"]}
    p["max_steps"] = math.ceil(cfg["cap_factor"] * network.predicted_t2(p["nu2"], 0.01, p["eta"], p["lam"]))
    return p


def arms_b1(cfg):
    model, n = cfg["model"], cfg["model"]["n"]
    arms = [Arm("gd", "B", "all", _network(cfg, model, n, "full"))]
    for b in cfg["batch_sizes"]:
        arms.append(Arm(f"b{b}", "B", "all", _network(cfg, model, b, "replace"), "gd"))
    return arms


def arms_b2(cfg):
    arms = []
    for n in cfg["ns"]:
        model = dict(cfg["model"], n=n)
        arms.append(Arm(f"n{n}_gd", "B", f"n={n}", _network(cfg, model, n, "full")))
        for b in cfg["batch_sizes"]:
            arms.append(Arm(f"n{n}_b{b}", "B", f"n={n}", _network(cfg, model, b, "replace"), f"n{n}_gd"))
    return arms


def arms_b3(cfg):
    """Every arm is compared with full-batch GD at the first (reference) step size."""
    n, arms, ref = cfg["model"]["n"], [], f"eta{cfg['etas'][0]}_gd"
    for eta in cfg["etas"]:
        model = dict(cfg["model"], eta=eta)
        key = f"eta{eta}_gd"
        arms.append(Arm(key, "B", "all", _network(cfg, model, n, "full"), None if key == ref else ref))
        for b in cfg["batch_sizes"]:
            arms.append(Arm(f"eta{eta}_b{b}", "B", "all", _network(cfg, model, b, "replace"), ref))
    return arms


BUILDERS = dict(a1=arms_a1, a2=arms_a2, a3=arms_a3, b1=arms_b1, b2=arms_b2, b3=arms_b3)


def build_arms(name, cfg):
    arms = BUILDERS[name](cfg)
    keys = [a.key for a in arms]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name}: duplicate arm keys")
    for a in arms:
        if a.baseline is not None and a.baseline not in keys:
            raise ValueError(f"{name}: arm {a.key} names a missing baseline {a.baseline}")
        if a.baseline is not None and next(x for x in arms if x.key == a.baseline).group != a.group:
            raise ValueError(f"{name}: arm {a.key} and its baseline do not share data")
    return arms


def seeds_for(name, arm, count):
    """Data seeds (shared within the arm's group) and batch seeds (the arm's own)."""
    return ([data_seed(name, arm.group, i) for i in range(count)],
            [batch_seed(name, arm.key, i) for i in range(count)])


def predictions_path():
    return Path(ROOT) / "anchors" / "extension_predictions.json"
