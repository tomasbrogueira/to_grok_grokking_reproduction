"""A0: correctness checks that gate everything else (EXTENSION_EXPERIMENTS.md, "A0").

Each check writes results/<check>.json with its numbers and a pass flag. Tolerances come from
configs/extension/a0.yaml, fixed before the run.
"""
import math
import time

import torch

from .. import nn
from ..config import BASE
from ..data import DTYPE
from ..runio import Run, write_json
from ..runner import label_for
from ..spectral import SpectralGD
from . import experiments as ex
from . import linear, network
from .runner import eta_max_probe, resolve_device
from .seeds import batch_seed, data_seed

CHECKS = ("gd_match", "fact1", "fact2", "approx_spectrum", "approx_eta_max", "approx_residual", "approx_noise")


def rel_max(a, b):
    return ((a - b).abs() / b.abs()).max().item()


def _stats(x):
    t = torch.as_tensor(x, dtype=DTYPE)
    return dict(min=t.min().item(), median=t.median().item(), max=t.max().item(), values=t.tolist())


def check_gd_match(cfg, seeds, dev, log):
    """SGD fed the whole training set every step, against the closed form in grok/spectral.py."""
    pr, eta, lam, steps = cfg["problem"], cfg["eta"], cfg["lam"], cfg["steps"]
    ps = linear.build_problems(pr["n"], pr["m"], pr["nu2"], pr["noise_std"], seeds, dev)
    gds = [SpectralGD(mom, eta, lam) for mom in ps.moms]
    red, dense = linear.ReducedSGD(ps, eta, lam, ps.n, "full"), linear.DenseSGD(ps, eta, lam, ps.n, "full")
    worst = dict(reduced_train=0.0, reduced_test=0.0, dense_train=0.0, dense_test=0.0)
    smallest_train, t = float("inf"), 0
    while t < steps:
        k = min(256, steps - t)
        out = red.advance(k)
        times = out["times"]
        train = torch.stack([gd.train_mse(times) for gd in gds], 1)
        test = torch.stack([gd.population_loss(times) for gd in gds], 1)
        worst["reduced_train"] = max(worst["reduced_train"], rel_max(out["train_mse"], train))
        worst["reduced_test"] = max(worst["reduced_test"], rel_max(out["row_loss"] + out["null_loss"], test))
        for j in range(k):
            worst["dense_train"] = max(worst["dense_train"], rel_max(dense.train_mse(), train[j]))
            worst["dense_test"] = max(worst["dense_test"], rel_max(dense.test_loss(), test[j]))
            dense.step()
        smallest_train = min(smallest_train, train.min().item())
        t += k
    return dict(passed=max(worst.values()) < cfg["tol_exact"], tolerance=cfg["tol_exact"], steps=steps,
                max_relative_difference=worst, smallest_training_loss=smallest_train,
                note="relative differences over every step and seed; the smallest training loss shows the "
                     "comparison never happens at machine zero")


def check_fact1(cfg, seeds, dev, log):
    """Fact 1 on the dense m-dimensional simulator, and the reduced simulator against it on the same batches."""
    pr, eta, lam, steps = cfg["problem"], cfg["eta"], cfg["lam"], cfg["steps"]
    ps = linear.build_problems(pr["n"], pr["m"], pr["nu2"], pr["noise_std"], seeds, dev)
    arms, ok = {}, True
    for b, sampling in cfg["fact1_arms"]:
        bs = [batch_seed("a0", f"fact1_b{b}_{sampling}", i) for i in range(len(seeds))]
        dense, red = linear.DenseSGD(ps, eta, lam, b, sampling, bs), linear.ReducedSGD(ps, eta, lam, b, sampling, bs)
        w = dict(fact1=0.0, train=0.0, test=0.0, row=0.0, null=0.0)
        t = 0
        while t < steps:
            k = min(256, steps - t)
            out = red.advance(k)
            for j in range(k):
                row, null = dense.split_loss()
                w["fact1"] = max(w["fact1"], dense.fact1_error().max().item())
                w["train"] = max(w["train"], rel_max(out["train_mse"][j], dense.train_mse()))
                w["test"] = max(w["test"], rel_max(out["row_loss"][j] + out["null_loss"][j], dense.test_loss()))
                w["row"] = max(w["row"], rel_max(out["row_loss"][j], row))
                w["null"] = max(w["null"], rel_max(out["null_loss"][j], null))
                dense.step()
            t += k
        w["fact1"] = max(w["fact1"], dense.fact1_error().max().item())
        p0 = dense.project_perp(dense.theta0).norm(dim=1)
        passed = (w["fact1"] < cfg["tol_exact"] and w["null"] < cfg["tol_exact"]
                  and max(w["train"], w["test"]) < cfg["tol_reduced_vs_dense"])
        arms[f"b{b}_{sampling}"] = dict(passed=passed, max_fact1_error=w["fact1"], reduced_vs_dense=w,
                                        perp_norm_start=p0.min().item(),
                                        perp_norm_end=dense.project_perp(dense.theta).norm(dim=1).min().item())
        ok &= passed
        log(f"fact1 b={b} {sampling}: fact1 error {w['fact1']:.2e}, reduced vs dense train {w['train']:.2e} test {w['test']:.2e}")
    return dict(passed=ok, tolerance_fact1=cfg["tol_exact"], tolerance_reduced_vs_dense=cfg["tol_reduced_vs_dense"],
                steps=steps, arms=arms)


def check_fact2(cfg, seeds, dev, log):
    """Fact 2: W_perp = W (I - P_X) decays as (1 - eta lam)^t in the network, for n < d, under any batching."""
    f2, arms, ok = cfg["fact2"], {}, True
    for eta in f2["etas"]:
        for b, sampling in f2["arms"]:
            p = {**BASE, "model": "nn", "teacher": "zero", **f2["model"], "eta": eta, "b": b, "sampling": sampling}
            net = network.SGDNet(p, seeds, dev)
            if not net.has_null:
                raise ValueError("the Fact 2 check needs n < d")
            bs = [batch_seed("a0", f"fact2_eta{eta}_b{b}_{sampling}", i) for i in range(len(seeds))]
            stream = None if sampling == "full" else network.BatchStream(bs, p["n"], b, sampling)
            idx = None if stream is None else stream.block(0, f2["steps"]).to(dev)
            w_par0, a0, worst, finite = (net.W - net.w_perp()).clone(), net.a.clone(), 0.0, True
            for t in range(f2["steps"] + 1):
                mask, h, out = net.forward_train()
                if t % f2["every"] == 0 or t == f2["steps"]:
                    wp = net.w_perp()
                    rho_t = math.exp(t * math.log1p(-eta * p["lam"]))
                    dev_t = (wp - rho_t * net.W0_perp).norm(dim=(1, 2)) / net.W0_perp.norm(dim=(1, 2))
                    worst = max(worst, dev_t.max().item())
                    finite &= bool(torch.isfinite(dev_t).all())
                if t == f2["steps"]:
                    break
                gw, ga = net.grads(mask, h, out) if idx is None else net.grads_minibatch(mask, h, out, idx[t])
                net.W.add_(gw, alpha=-eta)
                net.a.add_(ga, alpha=-eta)
            w_par = net.W - net.w_perp()
            moved = ((w_par - w_par0).norm(dim=(1, 2)) / w_par0.norm(dim=(1, 2))).min().item()
            passed = finite and worst < cfg["tol_exact"]
            arms[f"eta{eta}_b{b}_{sampling}"] = dict(
                passed=passed, max_w_perp_error=worst, w_perp_norm_start=net.W0_perp.norm(dim=(1, 2)).min().item(),
                expected_decay=math.exp(f2["steps"] * math.log1p(-eta * p["lam"])),
                w_par_relative_change=moved, a_relative_change=((net.a - a0).norm(dim=1) / a0.norm(dim=1)).min().item())
            ok &= passed
            log(f"fact2 eta={eta} b={b} {sampling}: W_perp error {worst:.2e}, W_par moved by {moved:.2e}")
    return dict(passed=ok, tolerance=cfg["tol_exact"], steps=f2["steps"], model=f2["model"], arms=arms)


def _approx_problems(cfg, n, seeds, dev, noise_std=0.0):
    return linear.build_problems(n, cfg["approx"]["m"], cfg["problem"]["nu2"], noise_std, seeds, dev)


def check_approx_spectrum(cfg, seeds, dev, log):
    """kappa_min and kappa_max of K = X X^T against the Marchenko-Pastur edges (1 -+ sqrt(n/m))^2."""
    ap, out, ok = cfg["approx"], {}, True
    for n in ap["ns"]:
        ps = _approx_problems(cfg, n, seeds, dev)
        lo, hi = ps.kappa_min / ex.kappa_bar_min(n, ap["m"]), ps.kappa_max / ex.kappa_bar_max(n, ap["m"])
        passed = bool(((lo - 1).abs().max() <= ap["tol_kappa"]) and ((hi - 1).abs().max() <= ap["tol_kappa"]))
        out[f"n{n}"] = dict(passed=passed, kappa_bar_min=ex.kappa_bar_min(n, ap["m"]), kappa_bar_max=ex.kappa_bar_max(n, ap["m"]),
                            kappa_min_over_edge=_stats(lo), kappa_max_over_edge=_stats(hi),
                            max_row_norm2=_stats(ps.max_row_norm2), mean_row_norm2=_stats(ps.mean_row_norm2),
                            predicted_e2_ratio=_stats(ps.kappa_max / ps.max_row_norm2))
        ok &= passed
    return dict(passed=ok, tolerance=ap["tol_kappa"], by_n=out)


def check_approx_eta_max(cfg, seeds, dev, log):
    """Measured eta_max for b = n and b = 1 (A2's rule) against 2/(kappa_max/n + lam) and 2/max|x_i|^2."""
    ap, lam, out, ok = cfg["approx"], cfg["lam"], {}, True
    lo_band, hi_band = ap["band_eta_b1"]
    for n in ap["ns"]:
        ps = _approx_problems(cfg, n, seeds, dev)
        res = {}
        for b, sampling in ((n, "full"), (1, "replace")):
            bs = [batch_seed("a0", f"etamax_n{n}_b{b}", i) for i in range(len(seeds))]
            res[b] = eta_max_probe(ps, b, sampling, bs, lam, **ap["probe"])
        gd = torch.tensor(res[n]["eta_max_per_seed"], dtype=DTYPE) / torch.tensor(res[n]["formula_gd"], dtype=DTYPE)
        b1 = torch.tensor(res[1]["eta_max_per_seed"], dtype=DTYPE) / torch.tensor(res[1]["formula_b1"], dtype=DTYPE)
        e2_measured = n * torch.tensor(res[1]["eta_max_per_seed"], dtype=DTYPE) / torch.tensor(res[n]["eta_max_per_seed"], dtype=DTYPE)
        e2_pred = ps.kappa_max.cpu() / ps.max_row_norm2.cpu()
        gd_ok = bool((gd - 1).abs().max() <= ap["tol_eta_gd"])
        b1_ok = bool(lo_band <= b1.median() <= hi_band)
        out[f"n{n}"] = dict(gd_passed=gd_ok, b1_passed=b1_ok, gd_measured_over_formula=_stats(gd),
                            b1_measured_over_formula=_stats(b1),
                            b1_over_mean_norm_formula=_stats(torch.tensor(res[1]["eta_max_per_seed"], dtype=DTYPE)
                                                             * ps.mean_row_norm2.cpu() / 2),
                            implied_e2_ratio_over_predicted=_stats(e2_measured / e2_pred),
                            eta_max_gd=res[n]["eta_max_per_seed"], eta_max_b1=res[1]["eta_max_per_seed"],
                            grid_ratio=res[n]["grid_ratio"])
        ok &= gd_ok and b1_ok
        log(f"eta_max n={n}: GD measured/formula median {gd.median():.4f}; b=1 measured/formula median {b1.median():.4f} "
            f"(range {b1.min():.3f} to {b1.max():.3f})")
    return dict(passed=ok, tolerance_gd=ap["tol_eta_gd"], band_b1=ap["band_eta_b1"], probe=ap["probe"], lam=lam, by_n=out)


def check_approx_residual(cfg, seeds, dev, log):
    """The ridge residual r = n lam (K + n lam I)^-1 y: checked against the bound |r| <= n lam / kappa_min |y|."""
    ap, noise, out, ok = cfg["approx"], cfg["approx"]["noise"], {}, True
    cases = [(n, cfg["lam"], 0.0, f"n{n}_default_lam") for n in ap["ns"]]
    for ratio in noise["ratios"]:
        for sigma in noise["sigmas"]:
            lam = ratio * ex.kappa_bar_min(noise["n"], ap["m"]) / noise["n"]
            cases.append((noise["n"], lam, sigma, f"n{noise['n']}_ratio{ratio}_sigma{sigma}"))
    for n, lam, sigma, tag in cases:
        ps = _approx_problems(cfg, n, seeds, dev, sigma)
        r2, y2 = linear.ridge_residual_norm2(ps, lam), (ps.y**2).sum(dim=1)
        rel, bound = torch.sqrt(r2 / y2), n * lam / ps.kappa_min
        passed = bool((rel <= bound * (1 + 1e-9)).all())
        out[tag] = dict(passed=passed, lam=lam, sigma=sigma, n_lam_over_kappa_min=_stats(bound),
                        residual_norm_over_label_norm=_stats(rel), train_mse_at_ridge_solution=_stats(r2 / n))
        ok &= passed
    return dict(passed=ok, note="passes if |r|/|y| <= n lam / kappa_min for every seed (an exact bound); "
                               "the size of the ratio is the information", cases=out)


def check_approx_noise(cfg, seeds, dev, log):
    """Stationary excess test loss of b = 1 SGD around the ridge solution, on A3's grid.

    Tests "SGD noise scales with the training residual": m * DeltaL / |r|^2 should be O(1) where n lam << kappa_min.
    Also gives DeltaL / c, which says whether A3 can see any effect at all (plan problem 1 in EXTENSION_NOTES.md).
    """
    ap = cfg["approx"]
    nz = ap["noise"]
    n, eta, out = nz["n"], nz["eta"], {}
    for ratio in nz["ratios"]:
        lam = ratio * ex.kappa_bar_min(n, ap["m"]) / n
        burn, window = math.ceil(3 / (eta * lam)) + nz["burn_extra"], math.ceil(2 / (eta * lam))
        for sigma in nz["sigmas"]:
            ps = _approx_problems(cfg, n, seeds, dev, sigma)
            bs = [batch_seed("a0", f"noise_r{ratio}_s{sigma}", i) for i in range(len(seeds))]
            sim = linear.ReducedSGD(ps, eta, lam, 1, "replace", bs)
            kappa = (ps.Z**2).sum(dim=1)  # column norms of Z = U diag(s) are the eigenvalues of K
            c_hat = (ps.Z.transpose(1, 2) @ ps.y[:, :, None])[:, :, 0] / (kappa + n * lam)  # ridge solution
            sim.c = c_hat.clone()
            row_hat = ((c_hat - ps.c_star) ** 2).sum(dim=1) / ps.m
            acc, t = torch.zeros(ps.S, dtype=DTYPE, device=dev), 0
            while t < burn + window:
                k = min(256, burn + window - t)
                o = sim.advance(k)
                tt = torch.as_tensor(o["times"], device=dev)
                acc += o["row_loss"][tt >= burn].sum(dim=0)
                t += k
            delta = acc / window - row_hat
            r2 = linear.ridge_residual_norm2(ps, lam)
            out[f"ratio{ratio}_sigma{sigma}"] = dict(
                lam=lam, sigma=sigma, burn_in=burn, window=window, delta_L=_stats(delta),
                delta_L_over_c=_stats(delta / nz["c"]), m_delta_L_over_residual_norm2=_stats(ps.m * delta / r2),
                gd_limit_test_loss=_stats(linear.gd_limits(ps, eta, lam)["test"]))
            log(f"noise ratio={ratio} sigma={sigma}: DeltaL mean {delta.mean():.2e} = {delta.mean() / nz['c']:.4f} c; "
                f"m DeltaL / |r|^2 median {(ps.m * delta / r2).median():.3f}")
    worst = max(v["delta_L_over_c"]["max"] for v in out.values())
    return dict(passed=True, informative=True, max_delta_L_over_c=worst, cases=out,
                note="no pass criterion: the plan gives none. Reported for the freeze of A3's predictions.")


FUNCS = dict(gd_match=check_gd_match, fact1=check_fact1, fact2=check_fact2, approx_spectrum=check_approx_spectrum,
             approx_eta_max=check_approx_eta_max, approx_residual=check_approx_residual, approx_noise=check_approx_noise)


def execute(run_name, base="runs", smoke=False, resume=False, seeds=None, device="auto", stop_after=None):
    cfg = ex.load_config("a0", smoke)
    nseeds = seeds or cfg["seeds"]
    dev = resolve_device("A", device)
    data_seeds = [data_seed("a0", "all", i) for i in range(nseeds)]
    resolved = dict(experiment="a0", smoke=smoke, seeds=nseeds, config=cfg, data_seeds=data_seeds, checks=list(CHECKS))
    run = Run(base, run_name, resolved, resume)
    torch.manual_seed(0)
    label = label_for(smoke, dev)
    run.log(f"start a0 as {run_name} on {dev}; {nseeds} seeds; label: {label}; env {run.env()}")
    done = 0
    for name in CHECKS:
        if run.has_result(name):
            run.log(f"{name} already done")
            continue
        t0 = time.time()
        res = FUNCS[name](cfg, data_seeds, dev, run.log)
        run.save_result(name, res)
        run.metric(check=name, passed=res["passed"], seconds=time.time() - t0)
        run.log(f"{name}: {'PASS' if res['passed'] else 'FAIL'} in {time.time() - t0:.1f}s")
        done += 1
        if stop_after is not None and done >= stop_after:
            raise nn.Interrupted(f"stopped after {done} checks")
    import json
    results = {name: json.loads(run.result_path(name).read_text()) for name in CHECKS}
    summary = dict(experiment="a0", run_name=run_name, label=label, smoke=smoke, device=str(dev), seeds=nseeds,
                   env=run.env(), passed={k: v["passed"] for k, v in results.items()},
                   all_passed=all(v["passed"] for v in results.values()))
    write_json(run.dir / "results.json", summary)
    run.log(f"finished: {label}; checks passed: {summary['passed']}")
    return run.dir
