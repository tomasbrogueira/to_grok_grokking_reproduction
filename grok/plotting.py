"""Figures in the paper's layouts: same axes, scales and panel order. Agg backend only."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import curve_times  # noqa: E402
from .runner import aggregate_curves, aggregate_timing, load_results  # noqa: E402

BLUE, RED, SALMON = "#0000ff", "#ff0000", "#ffa07a"
COLORS = [BLUE, RED, SALMON]
FLOOR = 0.01  # U7
DECADES = [10**k for k in range(7)]


def _curve_times(c):
    """Grid points only. NN runs also log search-schedule evaluations; plots use the paper's grid."""
    keep = np.isin(c["times"], curve_times())
    keep &= c["times"] >= 1
    return keep


def _loss_axes(ax, ymax=None, title=None):
    ax.set_xscale("log")
    ax.set_xticks(DECADES)
    ax.set_xlim(0.55, 2e6)
    ax.set_ylim(FLOOR, ymax)
    ax.set_xlabel("training steps")
    ax.set_ylabel("square loss")
    ax.grid(True, ls="--", lw=1.2, color="0.7")
    if title:
        ax.set_title(title)


def band_panel(ax, res, title=None):
    """Fig 1 panel: mean and +/- 1 std over seeds."""
    c = aggregate_curves(res)
    k = _curve_times(c)
    t = c["times"][k]
    for key, col, lab in (("train_mse", RED, "training loss"), ("test_mse", BLUE, "test loss")):
        y = c[key][:, k]
        m, s = y.mean(0), y.std(0)
        ax.plot(t, m, color=col, label=lab)
        ax.fill_between(t, m - s, m + s, color=col, alpha=0.25, lw=0)
    _loss_axes(ax, None, title)
    ax.legend(frameon=False)


def overlay_panel(ax, items, title=None, ymax=None):
    """Figs 3, 6R: dashed train and solid test per setting, seed mean."""
    for (label, res), col in zip(items, COLORS):
        c = aggregate_curves(res)
        k = _curve_times(c)
        ax.plot(c["times"][k], c["train_mse"][:, k].mean(0), "--", color=col, label=label)
        ax.plot(c["times"][k], c["test_mse"][:, k].mean(0), "-", color=col, label=label)
    _loss_axes(ax, ymax, title)
    ax.legend()


def single_panel(ax, res, title=None, ymax=None):
    c = aggregate_curves(res)
    k = _curve_times(c)
    ax.plot(c["times"][k], c["train_mse"][:, k].mean(0), "--", color=BLUE, label="training")
    ax.plot(c["times"][k], c["test_mse"][:, k].mean(0), "-", color=BLUE, label="test")
    _loss_axes(ax, ymax, title)
    ax.legend()


def timing_panel(ax, conds, title, xlabel, xscale, yscale, theory):
    xs = [c["value"] for c in conds]
    ag = [aggregate_timing(c) for c in conds]
    series = [("t1", BLUE, "$t_1$ empirical", "-"), ("t2", RED, "$t_2$ empirical", "-")]
    if theory:
        series = [("t1", BLUE, "$t_1$ empirical", "-"), ("t1_bound", BLUE, "$t_1$ theory", "--"),
                  ("t2", RED, "$t_2$ empirical", "-"), ("t2_bound", RED, "$t_2$ theory", "--")]
    for key, col, lab, ls in series:
        m = [np.nanmean(a[key]) for a in ag]
        s = [np.nanstd(a[key]) for a in ag]
        ax.errorbar(xs, m, yerr=s, fmt="o" + ls, color=col, label=lab, ms=3, capsize=2, lw=1)
    if xscale == "log2":
        ax.set_xscale("log", base=2)
    elif xscale == "log10":
        ax.set_xscale("log")
    if yscale == "log2":
        ax.set_yscale("log", base=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("training steps")
    ax.grid(True, ls="--", color="0.7")
    ax.legend(fontsize=8)


def _panel(results, name):
    return [r for r in results.values() if r["panel"] == name]


def _value_label(sym, res):
    return f"{sym}={res['value']:g}"


def fig1(runs, out, stamp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, run in zip(axes, runs):
        band_panel(ax, next(iter(load_results(run).values())))
    finish(fig, out, stamp)


def fig2(runs, out, stamp, theory=True):
    r = load_results(runs[0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    spec = [("a", r"Dependence on $\lambda$", "weight decay $\\lambda$", "log2", "log2"),
            ("b", "Dependence on $n$", "sample size $n$", "log2", "log2"),
            ("c", "Dependence on $m$", "width $m$", "linear", "log2"),
            ("d", r"Dependence on $\nu^2$", r"initialization scale $\nu^2$",
             "log10" if theory else "log2", "linear")]
    for ax, (p, title, xl, xs, ys) in zip(axes.flat, spec):
        conds = _panel(r, p)
        if conds:
            timing_panel(ax, conds, title, xl, xs, ys, theory)
    finish(fig, out, stamp)


def fig3(runs, out, stamp):
    r = load_results(runs[0])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    spec = [("a", r"Tuning $\lambda$", r"$\lambda$", None), ("b", "Tuning $n$", "$n$", None),
            ("c", "Tuning $m$", "$m$", None), ("d", r"Tuning $\nu^2$", r"$\nu^2$", 4.0)]
    for ax, (p, title, sym, ymax) in zip(axes.flat, spec):
        items = [(_value_label(sym, c), c) for c in _panel(r, p)]
        if items:
            overlay_panel(ax, items, title, ymax)
    finish(fig, out, stamp)


def fig5(runs, out, stamp):
    res = next(iter(load_results(runs[0]).values()))
    c = aggregate_curves(res)
    k = _curve_times(c)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    a.plot(c["times"][k], c["train_acc"][0, k], "--", color=BLUE, label="training")
    a.plot(c["times"][k], c["test_acc"][0, k], "-", color=BLUE, label="test")
    a.set_xscale("log")
    a.set_xticks(DECADES)
    a.set_xlabel("training steps")
    a.set_ylabel("accuracy")
    a.grid(True, ls="--", color="0.7")
    a.legend()
    single_panel(b, res)
    finish(fig, out, stamp)


def fig6(runs, out, stamp):
    r = load_results(runs[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    single_panel(a, _panel(r, "left")[0])
    overlay_panel(b, [(_value_label("std", c), c) for c in _panel(r, "right")])
    finish(fig, out, stamp)


def fig7(runs, out, stamp):
    res = next(iter(load_results(runs[0]).values()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    single_panel(ax, res, "RFF")
    finish(fig, out, stamp)


def finish(fig, out, stamp):
    if stamp:
        fig.suptitle(stamp, color="crimson", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


FIGURES = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=lambda r, o, s: fig2(r, o, s, theory=False),
               fig5=fig5, fig6=fig6, fig7=fig7)
