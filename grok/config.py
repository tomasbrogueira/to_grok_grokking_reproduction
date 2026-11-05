"""Experiment configs: one YAML per experiment, expanded into conditions."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIRS = [ROOT / "configs", ROOT / "configs" / "diagnostics", ROOT / "configs" / "alternatives"]

BASE = dict(
    model="identity", teacher="unit_linear", n=100, m=1000, d=None, lam=1e-4, eta=1.0, nu2=1.0,
    sigma2=1.0, noise_std=0.0, n_test=10000, x_dist="gaussian", w_scale="dm", eps=0.01, c=0.01,
    max_steps=5_000_000, checkpoint_every=20000,
)
SMOKE_LIMITS = dict(n=20, m=100, d=10, n_test=200)
SMOKE_MAX_STEPS = 300
SMOKE_SEEDS = 2


@dataclass
class Condition:
    panel: str
    param: str | None
    value: float | int | None
    params: dict

    @property
    def key(self):
        return self.panel if self.param is None else f"{self.panel}__{self.param}={self.value}"


def find_config(name):
    for d in CONFIG_DIRS:
        p = d / f"{name}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(f"no config called {name}")


def load_experiment(name):
    cfg = yaml.safe_load(find_config(name).read_text())
    cfg["defaults"] = {**BASE, **cfg.get("defaults", {})}
    unknown = set(cfg["defaults"]) - set(BASE)
    if unknown:
        raise ValueError(f"{name}: unknown settings {sorted(unknown)}")
    return cfg


def expand(cfg, smoke=False):
    """List the conditions of an experiment. Smoke keeps two sweep values on small data."""
    conds = []
    for panel, spec in cfg["panels"].items():
        param, values = spec.get("param"), spec.get("values", [None])
        if smoke:
            values = values[:2]
        for v in values:
            p = {**cfg["defaults"], **spec.get("set", {})}
            if param is not None:
                p[param] = v
            if smoke:
                for k, lim in SMOKE_LIMITS.items():
                    if p.get(k) is not None:
                        p[k] = min(p[k], lim)
                p["max_steps"] = SMOKE_MAX_STEPS
                p["checkpoint_every"] = 100
            conds.append(Condition(panel, param, v, p))
    return conds


def seed_for(experiment, cond, index):
    """Stable 63-bit seed for a condition and seed index, independent of run order."""
    text = f"{experiment}|{cond.key}|{index}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)
