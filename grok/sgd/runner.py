"""Run one extension experiment: resolved config, log, metrics, checkpoints, per-arm results.

Layout of runs/<run-name>/ (grok.runio.Run): config.yaml with every seed, log.txt, metrics.jsonl,
checkpoints/<arm>.pt, results/<arm>.json (crossing times, settings) and results/<arm>.npz (curves),
results.json once everything is done. An existing run is never overwritten; --resume continues it.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import nn
from ..metrics import curve_times
from ..runio import Run, pick_device, write_json
from ..runner import label_for
from . import experiments as ex
from . import linear, network
from .crossing import Tracker
from .seeds import batch_seed, data_seed

BLOCK_SGD, BLOCK_GD = 256, 8192
C_REF = 0.01  # threshold used in the predicted t2 that sets run lengths


def log_grid(cap, points=400):
    """Times at which curves are stored: the replication's grid plus about `points` log-spaced steps."""
    geo = np.unique(np.round(np.geomspace(1, max(cap, 2), points)).astype(np.int64))
    return np.array(sorted({0, *curve_times(cap), *geo.tolist()}), dtype=np.int64)


def save_npz(path, arrays):
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    tmp.replace(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def run_linear_arm(ps, params, batch_seeds, thresholds, ckpt_path, log, on_metrics, interrupt_at=None):
    """One setting A arm, all seeds at once. Every step's losses go through the crossing trackers.

    Length: to horizon_factor x predicted t2, then on until every seed has stayed at or below the smallest
    test threshold for a quarter of the predicted t2, and never beyond cap_factor x predicted t2.
    """
    eta, lam, b, sampling = params["eta"], params["lam"], params["b"], params["sampling"]
    t2_pred = linear.predicted_t2(ps.n, ps.m, params["nu2"], C_REF, eta, params["lam_ref"])
    horizon, cap = math.ceil(params["horizon_factor"] * t2_pred), math.ceil(params["cap_factor"] * t2_pred)
    runway = math.ceil(0.25 * t2_pred)
    exact = sampling == "full"
    sim = linear.ClosedFormGD(ps, eta, lam) if exact else linear.ReducedSGD(ps, eta, lam, b, sampling, batch_seeds)
    if exact and not all(sim.stable):
        log("WARNING: exact GD is unstable at this step size for some seeds")
    block = BLOCK_GD if exact else BLOCK_SGD
    limits = linear.gd_limits(ps, eta, lam)
    dev, S = ps.device, ps.S
    trk = dict(train=Tracker(S, ex.train_thresholds(thresholds), dev), test=Tracker(S, thresholds, dev),
               train_excess=Tracker(S, ex.train_thresholds(thresholds), dev), test_excess=Tracker(S, thresholds, dev))
    grid = log_grid(cap)
    window = math.ceil(2 / (eta * lam)) if params["tail_window"] and lam > 0 else 0
    state = dict(curves=[], tail_sum=torch.zeros(S, dtype=torch.float64, device=dev), tail_count=0, last_ckpt=0)
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists() and not exact:
        saved = torch.load(ckpt_path, map_location=dev, weights_only=False)
        sim.load(saved["sim"])
        for k, t in trk.items():
            t.load(saved["trackers"][k])
        state = saved["state"]
        log(f"resumed at step {sim.t}")

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(dict(sim=sim.state(), trackers={k: t.state() for k, t in trk.items()}, state=state), tmp)
        tmp.replace(ckpt_path)

    c_min, last_log = min(thresholds), time.time()
    state.setdefault("finite", [True] * S)

    def stop_reason():
        settled = bool(((sim.t - 1 - trk["test"].last_gt[trk["test"].thresholds.index(c_min)]) >= runway).all())
        if sim.t >= cap:
            return "cap"
        if not any(state["finite"]):
            return "diverged"
        if sim.t >= horizon and settled:
            return "settled"
        return None

    reason = stop_reason()  # a resumed arm may already be complete
    while reason is None:
        t0 = sim.t
        out = sim.advance(min(block, cap - t0))
        times, test = out["times"], out["row_loss"] + out["null_loss"]
        trk["train"].update(times, out["train_mse"])
        trk["test"].update(times, test)
        trk["train_excess"].update(times, out["train_mse"] - limits["train_mse"])
        trk["test_excess"].update(times, test - limits["test"])
        if window:
            tt = torch.as_tensor(times, device=dev)
            inside = (tt >= horizon - window) & (tt < horizon)
            state["tail_sum"] = state["tail_sum"] + test[inside].sum(dim=0)
            state["tail_count"] += int(inside.sum())
        lo, hi = np.searchsorted(grid, [t0, sim.t])
        for g in grid[lo:hi]:
            k = int(g) - t0
            row = dict(t=int(g), train_mse=out["train_mse"][k].cpu(), row_loss=out["row_loss"][k].cpu(),
                       null_loss=out["null_loss"][k].cpu())
            state["curves"].append(row)
            on_metrics({"t": int(g), "train_mse_mean": row["train_mse"].mean().item(),
                        "test_mean": (row["row_loss"] + row["null_loss"]).mean().item()})
        state["finite"] = torch.isfinite(out["train_mse"][-1]).tolist()
        reason = stop_reason()
        if time.time() - last_log > 30:
            last_log = time.time()
            log(f"step {sim.t} of at most {cap}: train {out['train_mse'][-1].mean():.3e} test {test[-1].mean():.3e}")
        if reason is None and not exact:
            if interrupt_at is not None and sim.t >= interrupt_at:
                save()
                raise nn.Interrupted(f"stopped at step {sim.t}")
            if sim.t - state["last_ckpt"] >= params["checkpoint_every"]:
                state["last_ckpt"] = sim.t
                save()
    if not exact:
        save()
    cur = state["curves"]
    curves = dict(times=np.array([r["t"] for r in cur]),
                  **{k: torch.stack([r[k] for r in cur]).numpy() for k in ("train_mse", "row_loss", "null_loss")})
    result = dict(
        seeds=list(ps.seeds), batch_seeds=None if exact else list(batch_seeds), steps=sim.t, stop_reason=reason,
        t2_pred=t2_pred, horizon=horizon, cap=cap, runway=runway, finite=state["finite"],
        trackers={k: t.export() for k, t in trk.items()}, limits={k: v.tolist() for k, v in limits.items()},
        tail_window=window, tail_mean_test=(state["tail_sum"] / state["tail_count"]).tolist() if window else None,
        null_loss_0=linear.null_loss(ps, eta, lam, [0])[0].tolist(), perp=ps.perp.tolist(),
        kappa_min=ps.kappa_min.tolist(), kappa_max=ps.kappa_max.tolist(),
        max_row_norm2=ps.max_row_norm2.tolist(), mean_row_norm2=ps.mean_row_norm2.tolist(),
        ridge_residual_norm2=linear.ridge_residual_norm2(ps, lam).tolist(),
        label_norm2=(ps.y**2).sum(dim=1).tolist())
    return result, curves


def eta_max_probe(ps, b, sampling, batch_seeds, lam, steps, grid_per_decade, eta_lo, eta_hi):
    """Largest stable step size per seed, by bisection on the grid eta_k = 10^(k / grid_per_decade).

    A2's rule: eta is stable for a seed if the training loss after `steps` steps is below its initial value.
    The batch sequence of a seed is the same at every eta. Assumes stability is monotone in eta.
    """
    S, dev = ps.S, ps.device
    k_of = lambda x: round(grid_per_decade * math.log10(x))
    eta_of = lambda k: torch.pow(torch.tensor(10.0, dtype=torch.float64, device=dev), k.double() / grid_per_decade)

    def stable(k):
        sim = linear.ReducedSGD(ps, eta_of(k), lam, b, sampling, batch_seeds)
        first = sim.losses(sim.c[None], [0])["train_mse"][0]
        idx = None if sim.stream is None else sim.stream.block(0, steps).to(dev)
        for t in range(steps):
            sim.step(None if idx is None else idx[t])
        last = sim.losses(sim.c[None], [steps])["train_mse"][0]
        return torch.isfinite(last) & (last < first)

    lo = torch.full((S,), k_of(eta_lo), dtype=torch.long, device=dev)
    hi = torch.full((S,), k_of(eta_hi), dtype=torch.long, device=dev)
    if not bool(stable(lo).all()) or bool(stable(hi).any()):
        raise RuntimeError("eta_max probe: the bracket does not contain the stability edge")
    rounds = 0
    while bool((hi - lo > 1).any()):
        mid = torch.div(lo + hi, 2, rounding_mode="floor")
        ok = stable(mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
        rounds += 1
    eta = eta_of(lo)
    return dict(eta_max_per_seed=eta.tolist(), eta_max=float(eta.min()), grid_ratio=10 ** (1 / grid_per_decade),
                rounds=rounds, steps=steps,
                formula_gd=(2 / (ps.kappa_max / ps.n + lam)).tolist(), formula_b1=(2 / ps.max_row_norm2).tolist(),
                kappa_min=ps.kappa_min.tolist(), kappa_max=ps.kappa_max.tolist(),
                max_row_norm2=ps.max_row_norm2.tolist(), mean_row_norm2=ps.mean_row_norm2.tolist())


def resolve_device(setting, requested):
    """Setting A does tiny per-step work, so 'auto' means the CPU there; setting B takes the GPU if there is one."""
    if setting == "A" and requested == "auto":
        return torch.device("cpu")
    return pick_device(requested)


def execute(name, run_name, base="runs", smoke=False, resume=False, seeds=None, device="auto", interrupt_at=None):
    if name == "a0":
        from . import a0
        return a0.execute(run_name, base, smoke, resume, seeds, device, stop_after=interrupt_at)
    cfg = ex.load_config(name, smoke)
    nseeds = seeds or cfg["seeds"]
    arms = ex.build_arms(name, cfg)
    dev = resolve_device(cfg["setting"], device)
    frozen = file_sha256(ex.predictions_path())
    if frozen is None and not smoke:
        raise FileNotFoundError(f"{ex.predictions_path()} is missing: predictions must be frozen before a full run")
    thresholds = [float(c) for c in cfg["thresholds"]]
    arm_table = {}
    for a in arms:
        ds, bs = ex.seeds_for(name, a, nseeds)
        arm_table[a.key] = dict(setting=a.setting, group=a.group, baseline=a.baseline, params=a.params,
                                data_seeds=ds, batch_seeds=bs)
    resolved = dict(experiment=name, smoke=smoke, seeds=nseeds, thresholds=thresholds, config=cfg, arms=arm_table,
                    predictions_sha256=frozen)
    run = Run(base, run_name, resolved, resume)
    torch.manual_seed(0)
    label = label_for(smoke, dev)
    run.log(f"start {name} as {run_name} on {dev}; {len(arms)} arms, {nseeds} seeds; label: {label}; "
            f"frozen predictions sha256: {frozen}; env {run.env()}")
    problems, pending = {}, {"at": interrupt_at}

    def problem_set(group, n, m, nu2, noise_std, data_seeds):
        key = (group, n, m, nu2, noise_std, tuple(data_seeds))
        if key not in problems:
            problems.clear()  # one problem set in memory at a time
            problems[key] = linear.build_problems(n, m, nu2, noise_std, data_seeds, dev)
        return problems[key]

    eta_max = run_a2_probe(run, name, cfg, dev) if name == "a2" else {}
    for a in arms:
        if run.has_result(a.key):
            run.log(f"{a.key} already done")
            continue
        info, params = arm_table[a.key], dict(a.params)
        if params.get("eta_fraction") is not None:
            params["eta"] = params["eta_fraction"] * eta_max[(params["n"], params["b"])]
        run.log(f"begin {a.key}: {params}")
        t0 = time.time()
        stop = None if params["sampling"] == "full" and a.setting == "A" else pending["at"]
        if stop is not None:
            pending["at"] = None  # the test interruption hits one arm only
        metric = lambda row, key=a.key: run.metric(arm=key, **row)
        if a.setting == "A":
            ps = problem_set(a.group, params["n"], params["m"], params["nu2"], params["noise_std"], info["data_seeds"])
            res, curves = run_linear_arm(ps, params, info["batch_seeds"], thresholds, run.ckpts / f"{a.key}.pt",
                                         run.log, metric, stop)
        else:
            res, curves = network.run_arm(params, info["data_seeds"], info["batch_seeds"],
                                          ex.train_thresholds(thresholds), thresholds, dev, run.ckpts / f"{a.key}.pt",
                                          run.log, metric, stop)
            curves = {k: v.numpy() for k, v in curves.items()}
        save_npz(run.results / f"{a.key}.npz", curves)
        run.save_result(a.key, dict(arm=a.key, setting=a.setting, group=a.group, baseline=a.baseline, params=params,
                                    thresholds=thresholds, **res))
        run.log(f"{a.key} done: {res['steps']} steps in {time.time() - t0:.1f}s")
    summary = dict(experiment=name, run_name=run_name, label=label, smoke=smoke, device=str(dev), seeds=nseeds,
                   env=run.env(), arms=[a.key for a in arms], predictions_sha256=frozen)
    write_json(run.dir / "results.json", summary)
    run.log(f"finished: {label}")
    return run.dir


def run_a2_probe(run, name, cfg, dev):
    """A2 stage 1: eta_max(n, b) on probe seeds that are not among the run's paired seeds."""
    pr, out = cfg["probe"], {}
    for n in cfg["ns"]:
        ps = None
        for b in ex.a2_batch_sizes(cfg, n):
            key = f"probe_n{n}_b{b}"
            if not run.has_result(key):
                if ps is None:
                    ps = linear.build_problems(n, cfg["m"], cfg["nu2"], 0.0,
                                               [data_seed(name, f"probe|n={n}", i) for i in range(pr["seeds"])], dev)
                t0 = time.time()
                bs = [batch_seed(name, key, i) for i in range(pr["seeds"])]
                res = eta_max_probe(ps, b, "full" if b == n else "replace", bs, cfg["lam"], pr["steps"],
                                    pr["grid_per_decade"], pr["eta_lo"], pr["eta_hi"])
                run.save_result(key, dict(probe=True, n=n, b=b, lam=cfg["lam"], seeds=list(ps.seeds), batch_seeds=bs, **res))
                run.log(f"{key}: eta_max = {res['eta_max']:.5g} (min over {pr['seeds']} probe seeds) in {time.time() - t0:.1f}s")
            out[(n, b)] = json.loads(run.result_path(key).read_text())["eta_max"]
    return out


def smoke(name, run_name, base="runs", device="cpu"):
    """Small full path: stop part way, resume, finish, analyse."""
    cut = 1 if name == "a0" else (200 if ex.load_config(name, True)["setting"] == "A" else 60)
    try:
        execute(name, run_name, base, smoke=True, device=device, interrupt_at=cut)
        raise RuntimeError("smoke: the interruption did not happen")
    except nn.Interrupted as e:
        print(f"smoke: {e}; resuming", flush=True)
    d = execute(name, run_name, base, smoke=True, resume=True, device=device)
    if name != "a0":
        from . import analysis
        analysis.analyze(d)
    return d


def load_run(run_dir):
    """Summary, resolved config, and per-arm results with their curves."""
    import yaml

    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    summary = json.loads((run_dir / "results.json").read_text()) if (run_dir / "results.json").exists() else None
    arms, probes = {}, {}
    for p in sorted((run_dir / "results").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("probe"):
            probes[p.stem] = r
            continue
        npz = p.with_suffix(".npz")
        r["curves"] = dict(np.load(npz)) if npz.exists() else {}
        arms[p.stem] = r
    return dict(dir=run_dir, config=config, summary=summary, arms=arms, probes=probes)
