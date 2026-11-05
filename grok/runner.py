"""Run one experiment: linear conditions by exact solution, NN conditions by explicit GD."""
import json
import time

import numpy as np
import torch

from . import nn, theory
from .config import expand, load_experiment, seed_for
from .data import moments
from .metrics import curve_times, first_at_or_below, last_above, search_grid
from .runio import Run, pick_device, write_json
from .spectral import SpectralGD


def scalar(f):
    return lambda t: float(f([t])[0])


def linear_seed(p, seed, kind, device):
    gen = torch.Generator().manual_seed(seed)
    gd = SpectralGD(moments(p, gen, device), p["eta"], p["lam"])
    out = dict(seed=seed, stable=gd.stable, lam_plus=gd.lam_plus, lam_max=gd.lam_max)
    if kind == "timing":
        grid = search_grid()
        Ln, gen_loss = gd.Ln(grid).cpu().numpy(), gd.generalization_loss(grid).cpu().numpy()
        mse = 2 * Ln
        eps, c = p["eps"], p["c"]
        out["t1"] = last_above(Ln, grid, eps, scalar(gd.Ln))
        out["t1_unhalved"] = last_above(mse, grid, eps, scalar(gd.train_mse))
        out["t2"] = first_at_or_below(gen_loss, grid, c, scalar(gd.generalization_loss))
        out["t1_bound"] = theory.eq8_t1_bound(p["n"], p["m"], p["nu2"], eps, p["eta"], gd.lam_plus)
        out["t2_bound"] = theory.eq8_t2_bound(p["n"], p["m"], p["nu2"], eps, p["eta"], p["lam"])
    else:
        ts = curve_times()
        out["times"] = ts
        out["train_mse"] = gd.train_mse(ts).cpu().tolist()
        out["test_mse"] = gd.generalization_loss(ts).cpu().tolist()
        if gd.B is not None:
            tr, te = gd.accuracy(ts, p["eps"])
            out["train_acc"], out["test_acc"] = tr.cpu().tolist(), te.cpu().tolist()
    return out


def run_linear_condition(run, cond, name, seeds, kind, device):
    rows = []
    for i in range(seeds):
        t0 = time.time()
        row = linear_seed(cond.params, seed_for(name, cond, i), kind, device)
        rows.append(row)
        if not row["stable"]:
            run.log(f"WARNING {cond.key} seed {i}: GD unstable, lam_max/n too large")
        run.metric(condition=cond.key, seed_index=i, **{k: v for k, v in row.items() if not isinstance(v, list)})
        run.log(f"{cond.key} seed {i} done in {time.time() - t0:.1f}s")
    return dict(condition=cond.key, panel=cond.panel, param=cond.param, value=cond.value,
                params=cond.params, kind=kind, rows=rows)


def run_nn_condition(run, cond, name, seeds, kind, device, interrupt_at):
    seed_list = [seed_for(name, cond, i) for i in range(seeds)]
    t0 = time.time()
    res = nn.run_condition(cond.params, seed_list, device, run.ckpts / f"{cond.key}.pt",
                           run.log, lambda row: run.metric(condition=cond.key, **row), interrupt_at)
    run.log(f"{cond.key} finished={res['finished']} steps={res['steps']} in {time.time() - t0:.1f}s")
    return dict(condition=cond.key, panel=cond.panel, param=cond.param, value=cond.value,
                params=cond.params, kind=kind, nn=res)


def label_for(smoke, device):
    if smoke:
        return "CPU smoke test" if device.type == "cpu" else "smoke test"
    return f"full {device.type.upper().replace('CUDA', 'GPU')} run"


def execute(name, run_name, base="runs", smoke=False, resume=False, seeds=None, device="auto",
            stop_after_conditions=None, interrupt_at=None):
    cfg = load_experiment(name)
    device = pick_device(device)
    nseeds = seeds or (2 if smoke else cfg["seeds"])
    if smoke:
        nseeds = min(nseeds, 2)
    conds = expand(cfg, smoke)
    resolved = dict(experiment=name, kind=cfg["kind"], plot_as=cfg.get("plot_as", name), seeds=nseeds,
                    smoke=smoke, conditions={c.key: c.params for c in conds}, note=cfg.get("note", ""))
    run = Run(base, run_name, resolved, resume)
    torch.manual_seed(0)
    run.log(f"start {name} as {run_name} on {device}; {len(conds)} conditions, {nseeds} seeds; "
            f"label: {label_for(smoke, device)}; env {run.env()}")
    done = 0
    for cond in conds:
        if run.has_result(cond.key):
            run.log(f"{cond.key} already done")
            continue
        run.log(f"begin {cond.key}: {cond.params}")
        if cond.params["model"] == "nn":
            res = run_nn_condition(run, cond, name, nseeds, cfg["kind"], device, interrupt_at)
        else:
            res = run_linear_condition(run, cond, name, nseeds, cfg["kind"], device)
        run.save_result(cond.key, res)
        done += 1
        if stop_after_conditions is not None and done >= stop_after_conditions:
            raise nn.Interrupted(f"stopped after {done} conditions")
    summary = dict(experiment=name, run_name=run_name, label=label_for(smoke, device), device=str(device),
                   seeds=nseeds, env=run.env(), conditions=[c.key for c in conds])
    write_json(run.dir / "results.json", summary)
    run.log(f"finished: {summary['label']}")
    return run.dir


def smoke(name, run_name, base="runs", device="cpu"):
    """Small full path: stop part way, resume, finish."""
    cfg = load_experiment(name)
    try:
        if cfg["defaults"]["model"] == "nn":
            execute(name, run_name, base, smoke=True, device=device, interrupt_at=150)
        else:
            execute(name, run_name, base, smoke=True, device=device, stop_after_conditions=1)
    except nn.Interrupted as e:
        print(f"smoke: {e}; resuming", flush=True)
    return execute(name, run_name, base, smoke=True, resume=True, device=device)


def load_results(run_dir):
    from pathlib import Path

    run_dir = Path(run_dir)
    return {p.stem: json.loads(p.read_text()) for p in sorted((run_dir / "results").glob("*.json"))}


def aggregate_timing(res):
    """Per-condition arrays of t1, t1_unhalved, t2 and bounds across seeds."""
    if "nn" in res:
        r = res["nn"]
        return {k: np.array([np.nan if v is None else v for v in r[k]], float) for k in ("t1", "t1_unhalved", "t2")}
    keys = ("t1", "t1_unhalved", "t2", "t1_bound", "t2_bound")
    return {k: np.array([np.nan if row[k] is None else row[k] for row in res["rows"]], float) for k in keys}


def aggregate_curves(res):
    """times, train and test arrays of shape [seeds, times], using times all seeds share."""
    if "nn" in res:
        r = res["nn"]
        return dict(times=np.array(r["times"]), train_mse=np.array(r["train_mse"]).T,
                    test_mse=np.array(r["test_mse"]).T)
    rows = res["rows"]
    out = dict(times=np.array(rows[0]["times"]))
    for k in ("train_mse", "test_mse", "train_acc", "test_acc"):
        if k in rows[0]:
            out[k] = np.array([row[k] for row in rows])
    return out
