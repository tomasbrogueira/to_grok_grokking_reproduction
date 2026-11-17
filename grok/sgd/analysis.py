"""Analysis of an extension run: paired statistics, the frozen decision rules, tables and figures.

Every tolerance comes from anchors/extension_predictions.json. Output goes to
<run>/analysis/: summary.json, summary.md and PNG figures. It is derived from the run's files
only and can be regenerated at any time. A smoke run is stamped as such everywhere.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import experiments as ex  # noqa: E402
from .crossing import crossing_times  # noqa: E402
from .runner import file_sha256, load_run  # noqa: E402

SMOKE_STAMP = "smoke test: not a result"
# Categorical colours in a fixed order (validated reference palette); line style repeats the identity.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"


# ---------------------------------------------------------------- statistics

def nan_array(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def summarize(x):
    """Unpaired spread of a per-seed quantity. Censored seeds (NaN) are counted, not dropped silently."""
    x = np.asarray(x, dtype=float)
    ok = x[np.isfinite(x)]
    out = dict(seeds=int(x.size), censored=int(x.size - ok.size), censored_fraction=float(1 - ok.size / x.size))
    if ok.size:
        out.update(mean=float(ok.mean()), std=float(ok.std()), median=float(np.median(ok)), min=float(ok.min()),
                   max=float(ok.max()), q25=float(np.quantile(ok, 0.25)), q75=float(np.quantile(ok, 0.75)))
    return out


def paired_relative(x, base, boot):
    """(x - base) / base on the seeds where both are finite, with a bootstrap interval for the mean."""
    x, base = np.asarray(x, dtype=float), np.asarray(base, dtype=float)
    ok = np.isfinite(x) & np.isfinite(base) & (base != 0)
    d = (x[ok] - base[ok]) / base[ok]
    out = dict(seeds=int(x.size), pairs=int(ok.sum()), censored=int(x.size - ok.sum()))
    if d.size:
        rng = np.random.default_rng(boot["seed"])
        means = d[rng.integers(0, d.size, (boot["resamples"], d.size))].mean(axis=1)
        alpha = (1 - boot["level"]) / 2
        out.update(mean=float(d.mean()), median=float(np.median(d)), max_abs=float(np.abs(d).max()),
                   ci_low=float(np.quantile(means, alpha)), ci_high=float(np.quantile(means, 1 - alpha)),
                   values=d.tolist())
    return out


def times_a(arm, eps, c, excess=False):
    """Per-seed t1 (halved and unhalved loss) and t2 (first, sustained) of a setting A arm. NaN = censored."""
    suffix = "_excess" if excess else ""
    halved = crossing_times(arm["trackers"]["train" + suffix], 2 * eps)["t1"]
    unhalved = crossing_times(arm["trackers"]["train" + suffix], eps)["t1"]
    test = crossing_times(arm["trackers"]["test" + suffix], c)
    return dict(t1=nan_array(halved), t1_unhalved=nan_array(unhalved), t2=nan_array(test["first"]),
                t2_sustained=nan_array(test["sustained"]))


def crossings_from_curve(times, series, c):
    """Setting B: first and sustained crossing of `c` per seed from evaluations, with log-linear interpolation."""
    times, series = np.asarray(times, dtype=float), np.asarray(series, dtype=float)
    series = np.where(np.isfinite(series), series, np.inf)
    E, S = series.shape
    grid, first, sustained = np.full(S, np.nan), np.full(S, np.nan), np.full(S, np.nan)

    def interp(a, b, s):  # series[a, s] > c >= series[b, s]
        la, lb = series[a, s], series[b, s]
        if not np.isfinite(la) or lb <= 0 or la <= lb:
            return times[b]
        return times[a] + (times[b] - times[a]) * math.log(la / c) / math.log(la / lb)

    for s in range(S):
        below = np.where(series[:, s] <= c)[0]
        if below.size:
            k = below[0]
            grid[s] = times[k]
            first[s] = times[0] if k == 0 else interp(k - 1, k, s)
        above = np.where(series[:, s] > c)[0]
        if above.size == 0:
            sustained[s] = times[0]
        elif above[-1] < E - 1:
            sustained[s] = interp(above[-1], above[-1] + 1, s)
    return dict(grid=grid, first=first, sustained=sustained)


def times_b(arm, eps, c, series="test_mse"):
    tr = arm["train_tracker"]
    cr = crossings_from_curve(arm["curves"]["times"], arm["curves"][series], c)
    return dict(t1=nan_array(crossing_times(tr, 2 * eps)["t1"]), t1_unhalved=nan_array(crossing_times(tr, eps)["t1"]),
                t2=cr["first"], t2_grid=cr["grid"], t2_sustained=cr["sustained"])


def null_clock(arm, c):
    """Setting A: first step at which the null-space loss alone is <= c (Fact 1, closed form). NaN if never."""
    eta, lam, m = arm["params"]["eta"], arm["params"]["lam"], arm["params"]["m"]
    out = []
    for a, b, cp in arm["perp"]:
        disc = b * b - a * (cp - m * c)
        if lam <= 0 or disc < 0:
            out.append(np.nan)
            continue
        u = (b + math.sqrt(disc)) / a  # (1 - eta lam)^t must be <= u
        out.append(0.0 if u >= 1 else float(math.ceil(math.log(u) / math.log1p(-eta * lam) - 1e-12)))
    return np.array(out)


def verdict(flag):
    return {True: "holds", False: "FAILS", None: "not exercised"}[flag]


# ---------------------------------------------------------------- figures

def _style(ax, xlabel, ylabel, title=None, logx=True, logy=False):
    ax.set_facecolor(SURFACE)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _finish(fig, path, smoke):
    fig.patch.set_facecolor(SURFACE)
    if smoke:
        fig.suptitle(SMOKE_STAMP, color="crimson", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _median_curve(ax, times, values, color, label, ls="-", scale=1.0):
    values = np.where(np.isfinite(values), values, np.nan)
    keep = np.asarray(times) >= 1
    t = np.asarray(times)[keep] * scale
    ax.plot(t, np.nanmedian(values[keep], axis=1), color=color, lw=2, ls=ls, label=label)


def _threshold_line(ax, c):
    ax.axhline(c, color=MUTED, lw=1, ls=":")
    ax.annotate(f"c = {c:g}", xy=(0.995, c), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=7, color=MUTED)


def _dot_panel(ax, groups, xlabel, band=None, title=None):
    """One row per group, one dot per seed, median marked. groups: [(label, values, color)]."""
    for i, (label, values, color) in enumerate(groups):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        y = np.full(v.size, len(groups) - 1 - i) + np.linspace(-0.18, 0.18, v.size) if v.size else []
        ax.scatter(v, y, s=14, color=color, alpha=0.7, linewidths=0)
        if v.size:
            ax.plot([np.median(v)] * 2, [len(groups) - 1 - i - 0.3, len(groups) - 1 - i + 0.3], color=INK, lw=2)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in reversed(groups)])
    _style(ax, xlabel, "", title, logx=False)
    ax.tick_params(axis="y", colors=INK)
    if band is not None:
        for edge in band:
            ax.axvline(edge, color=MUTED, lw=1, ls=":")
    ax.set_ylim(-0.6, len(groups) - 0.4)


# ---------------------------------------------------------------- shared tables

def sweep_table(arms, times_fn, sweep):
    """Medians and censored fractions of t1 and t2 for every swept threshold (eps = c = threshold)."""
    out = {}
    for key, arm in arms.items():
        out[key] = {}
        for thr in sweep:
            t = times_fn(arm, thr, thr)
            out[key][repr(thr)] = {k: summarize(v) for k, v in t.items()}
    return out


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}g}"


def pct(x):
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:+.2f}%"


def spread_rows(arms, times):
    rows = []
    for key in arms:
        for q in ("t1", "t2", "t2_sustained"):
            s = summarize(times[key][q])
            rows.append([key, q, fmt(s.get("median")), fmt(s.get("q25")) + " to " + fmt(s.get("q75")),
                         fmt(s.get("mean")) + " ± " + fmt(s.get("std"), 3), f"{s['censored']}/{s['seeds']}"])
    return md_table(["arm", "time", "median", "IQR", "mean ± std", "censored"], rows)


# ---------------------------------------------------------------- A1

def analyze_a1(run, frozen, ctx):
    arms, d, boot, out = run["arms"], ctx["defaults"], ctx["defaults"]["bootstrap"], ctx["out"]
    eps, c = d["eps"], d["c"]
    times = {k: times_a(a, eps, c) for k, a in arms.items()}
    gd = times["gd"]
    preds, gaps, md = [], {}, []
    preds.append(dict(id="p1_null_space", statement=frozen["p1_null_space"]["statement"], verdict="by construction",
                      detail=frozen["p1_null_space"]["status"]))

    def clock_check(pid, prefixes, in_epochs):
        spec, ok, detail = frozen[pid], True, {}
        for key, arm in arms.items():
            if not key.startswith(prefixes):
                continue
            scale = arm["params"]["b"] / arm["params"]["n"] if in_epochs else 1.0
            for q in ("t2", "t2_sustained"):
                p = paired_relative(times[key][q] * scale, gd[q], boot)
                detail[f"{key}:{q}"] = p
                ok &= p["censored"] == 0 and p.get("max_abs", math.inf) <= spec["tolerance"]
            gaps[key] = detail[f"{key}:t2"]
        preds.append(dict(id=pid, statement=spec["statement"], verdict=verdict(ok), tolerance=spec["tolerance"], detail=detail))
        return ok

    ok2 = clock_check("p2_matched_steps", ("steps_b", "reshuffle_b"), False)
    ok3 = clock_check("p3_matched_epochs", ("epochs_b",), True)
    spec, ratios = frozen["p4_t1_order"], {}
    for key, arm in arms.items():
        if key.startswith(("steps_b", "reshuffle_b", "epochs_b")):
            scale = arm["params"]["b"] / arm["params"]["n"] if key.startswith("epochs_b") else 1.0
            ratios[key] = summarize(times[key]["t1"] * scale / gd["t1"])
    med = ratios[spec["arm"]].get("median", math.nan)
    preds.append(dict(id="p4_t1_order", statement=spec["statement"], verdict=verdict(bool(med <= spec["max_ratio"])),
                      max_ratio=spec["max_ratio"], expected=spec["expected_ratio"], detail=ratios))
    ok5, detail5 = True, {}
    for key in ("nowd_gd", "nowd_b1"):
        arm = arms[key]
        test = arm["curves"]["row_loss"] + arm["curves"]["null_loss"]
        floor = np.array(arm["null_loss_0"])
        crossed = {repr(thr): int(np.isfinite(times_a(arm, thr, thr)["t2"]).sum()) for thr in d["sweep"]}
        above = bool((test >= floor[None, :] * (1 - 1e-12)).all())
        detail5[key] = dict(seeds_that_crossed=crossed, test_loss_never_below_initial_null_loss=above,
                            initial_null_loss=summarize(floor), smallest_test_loss=float(test.min()))
        ok5 &= above and not any(crossed.values())
    preds.append(dict(id="p5_no_weight_decay", statement=frozen["p5_no_weight_decay"]["statement"], verdict=verdict(ok5),
                      detail=detail5))
    delay = {k: summarize(times[k]["t2"] - null_clock(a, c)) for k, a in arms.items()}
    stop = not (ok2 and ok3)
    md += ["## Decision rule", frozen["decision_rule"], "",
           f"**{'STOP: a t2 gap needs explaining.' if stop else 'No unexplained t2 gap.'}**", "",
           "## Paired t2 against exact GD (relative difference per seed, in the arm's matched unit)",
           md_table(["arm", "pairs", "censored", "mean", "95% CI", "median", "largest |diff|"],
                    [[k, g["pairs"], g["censored"], pct(g.get("mean")), pct(g.get("ci_low")) + " to " + pct(g.get("ci_high")),
                      pct(g.get("median")), pct(g.get("max_abs"))] for k, g in gaps.items()]), "",
           "## t1 relative to GD (matched unit)",
           md_table(["arm", "median ratio", "IQR", "censored"],
                    [[k, fmt(r.get("median")), fmt(r.get("q25")) + " to " + fmt(r.get("q75")), f"{r['censored']}/{r['seeds']}"]
                     for k, r in ratios.items()]), "",
           "## Steps from the null-space-only crossing to t2 (what the row-space loss adds)",
           md_table(["arm", "median", "IQR"], [[k, fmt(s.get("median")), fmt(s.get("q25")) + " to " + fmt(s.get("q75"))]
                                               for k, s in delay.items()]), "",
           "## Unpaired spread, steps (examples = b × steps)", spread_rows(arms, times)]

    figs = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    panels = [("matched steps (x: steps)", [k for k in arms if k == "gd" or k.startswith(("steps_b", "reshuffle_b"))], False),
              ("matched epochs (x: epochs)", [k for k in arms if k == "gd" or k.startswith("epochs_b")], True),
              ("no weight decay (x: steps)", ["gd", "nowd_gd", "nowd_b1"], False)]
    for ax, (title, keys, in_epochs) in zip(axes, panels):
        for i, k in enumerate(keys):
            cv, p = arms[k]["curves"], arms[k]["params"]
            scale = p["b"] / p["n"] if in_epochs else 1.0
            _median_curve(ax, cv["times"], cv["row_loss"] + cv["null_loss"], COLORS[i], f"{k} test", "-", scale)
            _median_curve(ax, cv["times"], cv["train_mse"], COLORS[i], f"{k} train", "--", scale)
        _style(ax, "epochs" if in_epochs else "steps", "loss (median over seeds)", title, logy=True)
        _threshold_line(ax, c)
        ax.set_ylim(1e-4, 5)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    figs.append(_finish(fig, out / "a1_curves.png", ctx["smoke"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, k in zip(axes, ["gd", next(k for k in arms if k.startswith("steps_b"))]):
        cv = arms[k]["curves"]
        _median_curve(ax, cv["times"], cv["row_loss"] + cv["null_loss"], COLORS[0], "test loss")
        _median_curve(ax, cv["times"], cv["null_loss"], COLORS[1], "null-space loss", "--")
        _median_curve(ax, cv["times"], cv["row_loss"], COLORS[2], "row-space loss", "-.")
        _style(ax, "steps", "loss (median over seeds)", k, logy=True)
        _threshold_line(ax, c)
        ax.set_ylim(1e-12, 5)
        ax.legend(fontsize=8, frameon=False)
    figs.append(_finish(fig, out / "a1_split.png", ctx["smoke"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _dot_panel(axes[0], [(k, np.array(g.get("values", [])) * 100, COLORS[0]) for k, g in gaps.items()],
               "t2 relative to GD, % (one dot per seed, bar = median)",
               [-100 * frozen["p2_matched_steps"]["tolerance"], 100 * frozen["p2_matched_steps"]["tolerance"]], "t2")
    t1r = []
    for k, a in arms.items():
        if k in ratios:
            scale = a["params"]["b"] / a["params"]["n"] if k.startswith("epochs_b") else 1.0
            t1r.append((k, times[k]["t1"] * scale / gd["t1"], COLORS[1]))
    _dot_panel(axes[1], t1r, "t1 / t1 of GD", [frozen["p4_t1_order"]["max_ratio"]], "t1")
    figs.append(_finish(fig, out / "a1_paired.png", ctx["smoke"]))
    return dict(predictions=preds, stop_study=stop, paired_t2=gaps, t1_ratio=ratios, row_space_delay=delay,
                spread={k: {q: summarize(v) for q, v in t.items()} for k, t in times.items()},
                sweep=sweep_table(arms, times_a, d["sweep"]), figures=figs, markdown="\n".join(md))


# ---------------------------------------------------------------- A2

def analyze_a2(run, frozen, ctx):
    arms, probes, d, boot, out = run["arms"], run["probes"], ctx["defaults"], ctx["defaults"]["bootstrap"], ctx["out"]
    eps, c, cfg = d["eps"], d["c"], run["config"]["config"]
    times = {k: times_a(a, eps, c) for k, a in arms.items()}
    ns, fracs = cfg["ns"], cfg["eta_fractions"]
    main = frozen["p3_e2_ratio"]["fraction"] if frozen["p3_e2_ratio"]["fraction"] in fracs else fracs[0]
    key = lambda n, f, b: f"n{n}_f{f}_b{b}"
    clock = lambda k: arms[k]["params"]["eta"] * arms[k]["params"]["lam"] * times[k]["t2"]
    preds, md = [], []

    spec, detail, ok_main = frozen["p1_clock"], {}, True
    for n in ns:
        for f in fracs:
            for b in ex.a2_batch_sizes(cfg, n)[:-1]:
                p = paired_relative(clock(key(n, f, b)), clock(key(n, f, n)), boot)
                detail[key(n, f, b)] = p
                if f == main:
                    ok_main &= p["censored"] == 0 and p.get("max_abs", math.inf) <= spec["tolerance"]
    preds.append(dict(id="p1_clock", statement=spec["statement"], verdict=verdict(ok_main), tolerance=spec["tolerance"],
                      main_fraction=main, detail=detail))

    spec, probe_tab, ok_gd, ok_b1 = frozen["p2_eta_max"], {}, True, True
    for n in ns:
        pg, p1 = probes[f"probe_n{n}_b{n}"], probes[f"probe_n{n}_b1"]
        rg = np.array(pg["eta_max_per_seed"]) / np.array(pg["formula_gd"])
        r1 = np.array(p1["eta_max_per_seed"]) / np.array(p1["formula_b1"])
        ok_gd &= bool(np.abs(rg - 1).max() <= spec["tolerance_gd"])
        ok_b1 &= bool(spec["band_b1"][0] <= np.median(r1) <= spec["band_b1"][1])
        probe_tab[f"n{n}"] = dict(gd_measured_over_formula=summarize(rg), b1_measured_over_formula=summarize(r1),
                                  eta_max={str(b): probes[f"probe_n{n}_b{b}"]["eta_max"] for b in ex.a2_batch_sizes(cfg, n)})
    preds.append(dict(id="p2_eta_max", statement=spec["statement"], verdict=verdict(ok_gd and ok_b1),
                      gd_part=verdict(ok_gd), b1_part=verdict(ok_b1), band_b1=spec["band_b1"],
                      tolerance_gd=spec["tolerance_gd"], a0_expectation=spec["a0_expectation"], detail=probe_tab))

    spec, ratio_tab, ok3, dots = frozen["p3_e2_ratio"], {}, True, []
    for i, n in enumerate(ns):
        gd_arm = arms[key(n, main, n)]
        measured = n * times[key(n, main, n)]["t2"] / times[key(n, main, 1)]["t2"]
        predicted = np.array(gd_arm["kappa_max"]) / np.array(gd_arm["max_row_norm2"])
        r = measured / predicted
        s = summarize(r)
        inside = bool(spec["band"][0] <= s.get("median", math.nan) <= spec["band"][1])
        ok3 &= inside
        ratio_tab[f"n{n}"] = dict(inside_band=inside, measured_over_predicted=s, measured=summarize(measured),
                                  predicted=summarize(predicted),
                                  marchenko_pastur_value=ex.kappa_bar_max(n, cfg["m"]))
        dots.append((f"n = {n} (n/m = {n / cfg['m']:g})", r, COLORS[i]))
    preds.append(dict(id="p3_e2_ratio", statement=spec["statement"], verdict=verdict(ok3), band=spec["band"],
                      a0_expectation=spec["a0_expectation"], dependence=spec["dependence"], detail=ratio_tab))

    shape = {}
    for n in ns:
        e1 = times[key(n, main, 1)]["t2"]
        shape[f"n{n}"] = {str(b): dict(e2=summarize(b * times[key(n, main, b)]["t2"]),
                                       e2_over_e2_b1=summarize(b * times[key(n, main, b)]["t2"] / e1),
                                       eta_max_over_b_eta_max_1=probes[f"probe_n{n}_b{b}"]["eta_max"] / (b * probes[f"probe_n{n}_b1"]["eta_max"]),
                                       t2_over_t1=summarize(times[key(n, main, b)]["t2"] / times[key(n, main, b)]["t1"]))
                          for b in ex.a2_batch_sizes(cfg, n)}
    preds.append(dict(id="p4_shape", statement=frozen["p4_shape"]["statement"], verdict="descriptive", detail=shape))

    md += [f"Main step-size fraction: {main} of eta_max(n, b). Other fractions are the robustness check.", "",
           "## eta_max from the probe seeds",
           md_table(["n", "GD measured/formula (range)", "b = 1 measured / (2/max‖x‖²), median (range)", "eta_max by b"],
                    [[n, _range(v["gd_measured_over_formula"]), _range(v["b1_measured_over_formula"]),
                      ", ".join(f"b={b}: {fmt(e)}" for b, e in v["eta_max"].items())]
                     for n, v in ((n, probe_tab[f"n{n}"]) for n in ns)]), "",
           "## eta·lam·t2 against the exact GD arm (paired)",
           md_table(["arm", "pairs", "censored", "mean", "95% CI", "largest |diff|"],
                    [[k, p["pairs"], p["censored"], pct(p.get("mean")), pct(p.get("ci_low")) + " to " + pct(p.get("ci_high")),
                      pct(p.get("max_abs"))] for k, p in detail.items()]), "",
           "## Examples to grok: E2(GD)/E2(b = 1), measured over predicted kappa_max/max‖x‖²",
           md_table(["n", "measured (median)", "predicted (median)", "measured/predicted (median, IQR)", "inside band"],
                    [[n, fmt(v["measured"].get("median")), fmt(v["predicted"].get("median")),
                      fmt(v["measured_over_predicted"].get("median")) + f" ({fmt(v['measured_over_predicted'].get('q25'))} to {fmt(v['measured_over_predicted'].get('q75'))})",
                      v["inside_band"]] for n, v in ((n, ratio_tab[f"n{n}"]) for n in ns)]), "",
           "## E2 by batch size (median over seeds)",
           md_table(["n", "b", "E2", "E2/E2(b=1)", "eta_max(b)/(b·eta_max(1))", "t2/t1"],
                    [[n, b, fmt(v["e2"].get("median")), fmt(v["e2_over_e2_b1"].get("median")), fmt(v["eta_max_over_b_eta_max_1"]),
                      fmt(v["t2_over_t1"].get("median"))] for n in ns for b, v in shape[f"n{n}"].items()]), "",
           "## Unpaired spread, steps", spread_rows(arms, times)]

    figs = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, n in enumerate(ns):
        bs = ex.a2_batch_sizes(cfg, n)
        e2 = [np.nanmedian(b * times[key(n, main, b)]["t2"]) for b in bs]
        axes[0].plot(bs, e2, color=COLORS[i], lw=2, marker="o", ms=6, label=f"n = {n}")
        axes[0].annotate(f"n = {n}", (bs[-1], e2[-1]), textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK, va="center")
    _style(axes[0], "batch size b (b = n is exact GD)", "examples to grok, E2 = b·t2 (median)",
           f"E2 at eta = {main}·eta_max(n, b)", logy=True)
    axes[0].legend(fontsize=8, frameon=False)
    _dot_panel(axes[1], dots, "measured / predicted E2(GD)/E2(b = 1) (one dot per seed, bar = median)", spec["band"],
               "prediction 3")
    figs.append(_finish(fig, out / "a2_examples.png", ctx["smoke"]))
    return dict(predictions=preds, eta_max=probe_tab, clock=detail, e2_ratio=ratio_tab, shape=shape,
                spread={k: {q: summarize(v) for q, v in t.items()} for k, t in times.items()},
                sweep=sweep_table(arms, times_a, d["sweep"]), figures=figs, markdown="\n".join(md))


def _range(s):
    return f"{fmt(s['median'])} ({fmt(s['min'])} to {fmt(s['max'])})"


# ---------------------------------------------------------------- A3

def analyze_a3(run, frozen, ctx):
    arms, d, boot, out = run["arms"], ctx["defaults"], ctx["defaults"]["bootstrap"], ctx["out"]
    eps, c, cfg = d["eps"], d["c"], run["config"]["config"]
    conds, rows, preds = {}, [], []
    for ratio in cfg["ratios"]:
        for sigma in cfg["sigmas"]:
            tag = f"r{ratio}_s{sigma}"
            gd, sgd = arms[f"{tag}_gd"], arms[f"{tag}_b{cfg['batch_sizes'][0]}"]
            tg, ts = times_a(gd, eps, c, excess=True), times_a(sgd, eps, c, excess=True)
            t1_abs_gd, t1_abs_sgd = times_a(gd, eps, c)["t1"], times_a(sgd, eps, c)["t1"]
            grok = summarize(tg["t2"] / t1_abs_gd)
            pronounced = bool(grok.get("median", 0) >= frozen["pronounced"]["min_ratio"] and grok["censored_fraction"] < 0.5)
            delta = np.array(sgd["tail_mean_test"]) - np.array(gd["limits"]["test"])
            gd_tail = np.array(gd["tail_mean_test"]) - np.array(gd["limits"]["test"])
            conds[tag] = dict(
                ratio=ratio, sigma=sigma, lam=gd["params"]["lam"], pronounced=pronounced, gd_t2_over_t1=grok,
                delta_L=summarize(delta), delta_L_over_c=float(delta.mean() / c), gd_tail_minus_limit=float(np.abs(gd_tail).max()),
                gd_limit_test=summarize(gd["limits"]["test"]), gd_limit_train_mse=summarize(gd["limits"]["train_mse"]),
                kappa_ratio_per_seed=summarize(gd["params"]["n"] * gd["params"]["lam"] / np.array(gd["kappa_min"])),
                t2=paired_relative(ts["t2"], tg["t2"], boot), t2_sustained=paired_relative(ts["t2_sustained"], tg["t2_sustained"], boot),
                t1_gd=summarize(t1_abs_gd), t1_sgd=summarize(t1_abs_sgd), t1_gd_relative_to_limit=summarize(tg["t1"]),
                t1_sgd_relative_to_limit=summarize(ts["t1"]), t2_gd=summarize(tg["t2"]), t2_sgd=summarize(ts["t2"]))
            v = conds[tag]
            rows.append([ratio, sigma, fmt(v["lam"]), fmt(grok.get("median")), f"{grok['censored']}/{grok['seeds']}",
                         "yes" if pronounced else "no", fmt(delta.mean(), 3), fmt(v["delta_L_over_c"], 3),
                         pct(v["t2"].get("mean")), pct(v["t2"].get("ci_low")) + " to " + pct(v["t2"].get("ci_high")),
                         pct(v["t2_sustained"].get("mean"))])
    pron = [v for v in conds.values() if v["pronounced"]]
    p1 = all(v["delta_L"]["mean"] < c for v in pron) if pron else None
    preds.append(dict(id="p1_noise_small", statement=frozen["p1_noise_small"]["statement"], verdict=verdict(p1),
                      pronounced_conditions=len(pron)))
    reached = [v for v in conds.values() if v["delta_L"]["mean"] >= c]
    p2 = None if not reached else all(v["ratio"] >= frozen["p2_noise_where"]["min_ratio"] for v in reached)
    preds.append(dict(id="p2_noise_where", statement=frozen["p2_noise_where"]["statement"], verdict=verdict(p2),
                      conditions_reaching_c=len(reached), largest_delta_L_over_c=max(v["delta_L_over_c"] for v in conds.values()),
                      a0_expectation=frozen["p2_noise_where"]["a0_expectation"]))
    tol = frozen["p3_t2_unchanged"]["tolerance"]
    bad = [f"r{v['ratio']}_s{v['sigma']}" for v in pron
           if max(abs(v["t2"].get("mean", math.inf)), abs(v["t2_sustained"].get("mean", math.inf))) > tol]
    p3 = None if not pron else not bad
    preds.append(dict(id="p3_t2_unchanged", statement=frozen["p3_t2_unchanged"]["statement"], verdict=verdict(p3), tolerance=tol,
                      conditions_over_tolerance=bad,
                      claim="'SGD noise can only matter where grokking has already disappeared' is "
                            + ("REJECTED" if bad else "not rejected")))
    md = ["t2 is measured against GD's exact limit plus c. t1 uses the absolute threshold; t1 relative to GD's limit is in summary.json.", "",
          md_table(["n·lam/kappa_bar_min", "sigma", "lam", "GD t2/t1 (median)", "t2/t1 censored", "pronounced", "DeltaL (mean)",
                    "DeltaL / c", "t2 change (mean)", "95% CI", "sustained t2 change"], rows)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, sigma in enumerate(cfg["sigmas"]):
        vs = [conds[f"r{r}_s{sigma}"] for r in cfg["ratios"]]
        axes[0].plot(cfg["ratios"], [max(v["delta_L_over_c"], 1e-12) for v in vs], color=COLORS[i], lw=2, marker="o", ms=6,
                     label=f"sigma = {sigma:g}")
        axes[1].plot(cfg["ratios"], [v["gd_t2_over_t1"].get("median", np.nan) for v in vs], color=COLORS[i], lw=2, marker="o",
                     ms=6, label=f"sigma = {sigma:g}")
    _style(axes[0], "n·lam / kappa_bar_min", "DeltaL / c (mean over seeds)", "extra stationary test loss of b = 1 SGD", logy=True)
    axes[0].axhline(1, color=MUTED, lw=1, ls=":")
    axes[0].annotate("DeltaL = c", xy=(0.995, 1), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=7, color=MUTED)
    _style(axes[1], "n·lam / kappa_bar_min", "t2 / t1 of exact GD (median)", "how pronounced grokking is", logy=True)
    axes[1].axhline(frozen["pronounced"]["min_ratio"], color=MUTED, lw=1, ls=":")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    figs = [_finish(fig, out / "a3_noise.png", ctx["smoke"])]
    return dict(predictions=preds, conditions=conds, figures=figs, markdown="\n".join(md),
                sweep=sweep_table(arms, lambda a, e, cc: times_a(a, e, cc, excess=True), d["sweep"]))


# ---------------------------------------------------------------- B1, B2, B3

def _network_curves(arms, keys, out_path, smoke, c, title_of=lambda k: k):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, k in enumerate(keys):
        cv = arms[k]["curves"]
        _median_curve(ax, cv["times"], cv["test_mse"], COLORS[i % len(COLORS)], f"{title_of(k)} test")
        _median_curve(ax, cv["times"], cv["train_mse"], COLORS[i % len(COLORS)], f"{title_of(k)} train", "--")
    _style(ax, "steps", "loss (median over seeds)", None, logy=True)
    _threshold_line(ax, c)
    ax.set_ylim(1e-4, 5)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return _finish(fig, out_path, smoke)


def analyze_b1(run, frozen, ctx):
    arms, d, boot, out = run["arms"], ctx["defaults"], ctx["defaults"]["bootstrap"], ctx["out"]
    eps, c = d["eps"], d["c"]
    times = {k: times_b(a, eps, c) for k, a in arms.items()}
    spec, gaps, changed = frozen["p1_clock"], {}, []
    for k, a in arms.items():
        if a["baseline"] is None:
            continue
        g = {q: paired_relative(times[k][q], times[a["baseline"]][q], boot) for q in ("t2", "t2_sustained", "t2_grid", "t1")}
        gaps[k] = g
        p = g["t2"]
        if "mean" in p and abs(p["mean"]) > spec["tolerance"] and not (p["ci_low"] <= 0 <= p["ci_high"]):
            changed.append(k)
    untestable = [k for k, g in gaps.items() if g["t2"]["pairs"] == 0]
    flag = None if len(untestable) == len(gaps) else not changed
    preds = [dict(id="p1_clock", statement=spec["statement"], verdict=verdict(flag), tolerance=spec["tolerance"],
                  arms_with_changed_clock=changed, arms_without_pairs=untestable,
                  consequence="B3 becomes a priority" if changed else "none", detail=gaps)]
    unstable = [k for k, a in arms.items() if a["diverged"]]
    md = ["## Paired gaps against full-batch GD (relative difference per seed)",
          md_table(["arm", "quantity", "pairs", "censored", "mean", "95% CI", "median"],
                   [[k, q, p["pairs"], p["censored"], pct(p.get("mean")), pct(p.get("ci_low")) + " to " + pct(p.get("ci_high")),
                     pct(p.get("median"))] for k, g in gaps.items() for q, p in g.items()]), "",
          f"Arms that diverged: {unstable or 'none'}", "", "## Unpaired spread, steps (examples = b × steps)", spread_rows(arms, times), "",
          "## Norms at the end of each run (median over seeds)",
          md_table(["arm", "‖a‖ / ‖a(0)‖", "‖W P_X‖ / ‖W P_X(0)‖", "steps"],
                   [[k, fmt(np.median(a["curves"]["a_norm"][-1] / a["curves"]["a_norm"][0])),
                     fmt(np.median(a["curves"]["w_par_norm"][-1] / a["curves"]["w_par_norm"][0])), a["steps"]] for k, a in arms.items()])]
    figs = [_network_curves(arms, list(arms), out / "b1_curves.png", ctx["smoke"], c)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    _dot_panel(ax, [(k, np.array(g["t2"].get("values", [])) * 100, COLORS[0]) for k, g in gaps.items()],
               "t2 relative to GD, % (one dot per seed, bar = median)", [-100 * spec["tolerance"], 100 * spec["tolerance"]])
    figs.append(_finish(fig, out / "b1_paired.png", ctx["smoke"]))
    return dict(predictions=preds, paired=gaps, diverged=unstable,
                spread={k: {q: summarize(v) for q, v in t.items()} for k, t in times.items()},
                sweep=sweep_table(arms, times_b, d["sweep"]), figures=figs, markdown="\n".join(md))


def _proxy_series(cv, kind):
    """Norm proxies of the test loss, each scaled to its value at t = 0. [E, S]."""
    a = (cv["a_norm"] / cv["a_norm"][0]) ** 2
    if kind == "perp":
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(cv["w_perp_norm"][0] > 0, (cv["w_perp_norm"] / cv["w_perp_norm"][0]) ** 2, 0.0)
        return cv["null_out_mse"][0] * w * a
    w = (cv["w_par_norm"] / cv["w_par_norm"][0]) ** 2
    return cv["test_mse"][0] * {"product": a * w, "a": a, "w_par": w}[kind]


def analyze_b2(run, frozen, ctx):
    arms, d, out = run["arms"], ctx["defaults"], ctx["out"]
    eps, c, cfg = d["eps"], d["c"], run["config"]["config"]
    mech, f2, rows, per, preds = frozen["mechanism"], frozen["fact2"], [], {}, []
    worst_f2 = 0.0
    for k, a in arms.items():
        cv, t = a["curves"], times_b(a, eps, c)
        t2 = t["t2"]
        t2_par = crossings_from_curve(cv["times"], cv["test_mse_par"], c)["first"] if a["has_null"] else t2
        proxy = {kind: crossings_from_curve(cv["times"], _proxy_series(cv, kind), c)["first"] for kind in ("perp", "product", "a", "w_par")}
        frac = summarize(t2_par / t2)
        prox = summarize(np.abs(proxy["perp"] - t2) / t2)
        c1 = bool(a["has_null"] and frac.get("median", math.inf) <= mech["max_parallel_fraction"])
        c2 = bool(a["has_null"] and prox.get("median", math.inf) <= mech["proxy_tolerance"])
        idx2 = [int(np.argmin(np.abs(cv["times"] - x))) if np.isfinite(x) else -1 for x in t2]
        flips = np.array([cv["act_change_train"][i, s] if i >= 0 else np.nan for s, i in enumerate(idx2)])
        dev = float(cv["w_perp_dev"].max()) if a["has_null"] else 0.0
        worst_f2 = max(worst_f2, dev)
        per[k] = dict(n=a["params"]["n"], d=a["params"]["d"], b=a["params"]["b"], has_null=a["has_null"],
                      explains=c1 and c2, condition_1=c1, condition_2=c2, t2=summarize(t2), t1=summarize(t["t1"]),
                      t2_par=summarize(t2_par), t2_par_over_t2=frac, proxy_error=prox, max_w_perp_error=dev,
                      null_output_mse_at_start=summarize(cv["null_out_mse"][0]),
                      test_flip_fraction_at_start=summarize(cv["flip_frac_test"][0]),
                      tracks_t2={kind: summarize(proxy[kind] / t2) for kind in ("product", "a", "w_par", "perp")},
                      train_activations_flipped_at_t2=summarize(flips),
                      vs_gd=None if a["baseline"] is None else paired_relative(t2, times_b(arms[a["baseline"]], eps, c)["t2"], d["bootstrap"]))
        v = per[k]
        rows.append([k, f"{v['n']} {'<' if a['has_null'] else '>='} {v['d']}", fmt(v["t2"].get("median")), fmt(v["t2_par"].get("median")),
                     fmt(frac.get("median"), 3), fmt(prox.get("median"), 3), "yes" if v["explains"] else "no",
                     fmt(v["tracks_t2"]["product"].get("median"), 3), fmt(v["tracks_t2"]["a"].get("median"), 3),
                     fmt(v["tracks_t2"]["w_par"].get("median"), 3), fmt(v["train_activations_flipped_at_t2"].get("median"), 3)])
    any_null = any(a["has_null"] for a in arms.values())
    preds.append(dict(id="fact2", statement=f2["statement"], verdict=verdict(worst_f2 <= f2["tolerance"] if any_null else None),
                      tolerance=f2["tolerance"], largest_error=worst_f2))
    preds.append(dict(id="mechanism", statement=mech["statement"], verdict="per n: see detail",
                      explained_at=[k for k, v in per.items() if v["explains"]],
                      not_explained_at=[k for k, v in per.items() if not v["explains"]], detail=per))
    md = [f"Fact 2: largest relative error of W_perp against (1 − eta·lam)^t over all arms with n < d: {worst_f2:.2e} "
          f"(tolerance {f2['tolerance']:g}).", "",
          "Columns 'tracks t2' give the median ratio to t2 of the time a norm proxy of the test loss crosses c.", "",
          md_table(["arm", "n vs d", "t2 (median)", "t2 ablated", "ablated / t2", "proxy error", "mechanism explains",
                    "‖a‖‖W P_X‖ tracks t2", "‖a‖ tracks t2", "‖W P_X‖ tracks t2", "train activations flipped at t2"], rows)]
    ns = cfg["ns"]
    fig, axes = plt.subplots(2, len(ns), figsize=(4.2 * len(ns), 7.4), sharex=True, squeeze=False)
    for j, n in enumerate(ns):
        a = arms[f"n{n}_gd"]
        cv = a["curves"]
        ax = axes[0, j]
        _median_curve(ax, cv["times"], cv["test_mse"], COLORS[0], "test loss")
        _median_curve(ax, cv["times"], cv["test_mse_par"], COLORS[1], "ablated (W P_X)", "--")
        if a["has_null"]:
            _median_curve(ax, cv["times"], cv["null_out_mse"], COLORS[2], "null-space part of output", "-.")
            _median_curve(ax, cv["times"], _proxy_series(cv, "perp"), COLORS[3], "norm proxy of it", ":")
        _median_curve(ax, cv["times"], cv["train_mse"], COLORS[4], "training loss", (0, (1, 1)))
        _style(ax, "", "loss (median over seeds)" if j == 0 else "", f"n = {n}, GD", logy=True)
        _threshold_line(ax, c)
        ax.set_ylim(1e-4, 5)
        ax = axes[1, j]
        _median_curve(ax, cv["times"], cv["a_norm"] / cv["a_norm"][0], COLORS[0], "‖a‖")
        _median_curve(ax, cv["times"], cv["w_par_norm"] / cv["w_par_norm"][0], COLORS[1], "‖W P_X‖", "--")
        if a["has_null"]:
            _median_curve(ax, cv["times"], cv["w_perp_norm"] / cv["w_perp_norm"][0], COLORS[2], "‖W_perp‖", "-.")
        _style(ax, "steps", "norm relative to t = 0" if j == 0 else "", None, logy=True)
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=7, frameon=False)
    figs = [_finish(fig, out / "b2_mechanism.png", ctx["smoke"])]
    return dict(predictions=preds, per_arm=per, figures=figs, markdown="\n".join(md),
                sweep=sweep_table(arms, times_b, d["sweep"]))


def analyze_b3(run, frozen, ctx):
    arms, d, boot, out = run["arms"], ctx["defaults"], ctx["defaults"]["bootstrap"], ctx["out"]
    eps, c, spec = d["eps"], d["c"], frozen["p1_clock"]
    times = {k: times_b(a, eps, c) for k, a in arms.items()}
    clock = {k: a["params"]["eta"] * a["params"]["lam"] * times[k]["t2"] for k, a in arms.items()}
    stable = {k: bool(not a["diverged"] and (a["curves"]["train_mse"][-1] < a["curves"]["train_mse"][0]).all()) for k, a in arms.items()}
    ref = next(k for k, a in arms.items() if a["baseline"] is None)
    detail, ok = {}, stable[ref]
    for k, a in arms.items():
        if k == ref or not stable[k]:
            continue
        p = paired_relative(clock[k], clock[ref], boot)
        detail[k] = p
        ok &= p["pairs"] > 0 and abs(p.get("mean", math.inf)) <= spec["tolerance"]
    preds = [dict(id="p1_clock", statement=spec["statement"], verdict=verdict(ok if stable[ref] and detail else None),
                  tolerance=spec["tolerance"], reference=ref, unstable_arms=[k for k, s in stable.items() if not s], detail=detail)]
    md = [f"Reference arm: {ref}. Unstable arms (excluded): {[k for k, s in stable.items() if not s] or 'none'}", "",
          md_table(["arm", "eta", "b", "eta·lam·t2 (median)", "paired difference to reference (mean)", "95% CI", "pairs"],
                   [[k, a["params"]["eta"], a["params"]["b"], fmt(summarize(clock[k]).get("median")),
                     pct(detail.get(k, {}).get("mean")), pct(detail.get(k, {}).get("ci_low")) + " to " + pct(detail.get(k, {}).get("ci_high")),
                     detail.get(k, {}).get("pairs", "reference" if k == ref else "unstable")] for k, a in arms.items()])]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    _dot_panel(ax, [(k, clock[k], COLORS[0]) for k in arms], "eta·lam·t2 (one dot per seed, bar = median)")
    figs = [_finish(fig, out / "b3_clock.png", ctx["smoke"]), _network_curves(arms, list(arms), out / "b3_curves.png", ctx["smoke"], c)]
    return dict(predictions=preds, stable=stable, figures=figs, markdown="\n".join(md),
                spread={k: {q: summarize(v) for q, v in t.items()} for k, t in times.items()},
                sweep=sweep_table(arms, times_b, d["sweep"]))


ANALYZERS = dict(a1=analyze_a1, a2=analyze_a2, a3=analyze_a3, b1=analyze_b1, b2=analyze_b2, b3=analyze_b3)


def _clean(obj):
    """JSON without NaN or numpy types."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def analyze(run_dir, predictions=None):
    """Analyse a finished run. Returns the summary dict and writes <run>/analysis/."""
    run = load_run(run_dir)
    if run["summary"] is None:
        raise FileNotFoundError(f"{run_dir} has no results.json: the run is not finished")
    name, smoke = run["config"]["experiment"], bool(run["config"]["smoke"])
    path = Path(predictions) if predictions else ex.predictions_path()
    frozen = json.loads(path.read_text())
    sha_now, sha_run = file_sha256(path), run["config"].get("predictions_sha256")
    if sha_run is None:
        freeze = "the run started without a frozen predictions file (allowed for smoke runs only)"
    elif sha_run == sha_now:
        freeze = "the predictions file is the one frozen when the run started"
    else:
        freeze = "WARNING: the predictions file CHANGED after the run started"
    out = run["dir"] / "analysis"
    out.mkdir(exist_ok=True)
    ctx = dict(defaults=frozen["defaults"], smoke=smoke, out=out)
    res = ANALYZERS[name](run, frozen[name], ctx)
    label = run["summary"]["label"]
    summary = dict(experiment=name, run=str(run["dir"].name), label=label, smoke=smoke, seeds=run["config"]["seeds"],
                   predictions_file=str(path.name), predictions_sha256_now=sha_now, predictions_sha256_at_run_start=sha_run,
                   freeze_check=freeze, **{k: v for k, v in res.items() if k != "markdown"})
    if smoke:
        summary["warning"] = SMOKE_STAMP
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    head = [f"# {name.upper()} analysis: run `{run['dir'].name}`", ""]
    if smoke:
        head += [f"**{SMOKE_STAMP.upper()}. Small data, few seeds. Nothing below is evidence about the predictions.**", ""]
    head += [f"- Label: {label}", f"- Seeds: {run['config']['seeds']}", f"- Freeze check: {freeze}",
             f"- Thresholds: eps = c = {frozen['defaults']['eps']}; t1 on L_n = MSE/2", "", "## Predictions", ""]
    rows = [[p["id"], p["statement"], ("smoke, not a result: " if smoke else "") + str(p["verdict"])] for p in res["predictions"]]
    body = [md_table(["id", "prediction", "verdict"], rows), "", res["markdown"], "", "## Figures", ""]
    body += [f"- `{f}`" for f in res["figures"]]
    (out / "summary.md").write_text("\n".join(head + body) + "\n")
    return summary
