"""Run everything, then draw every figure.

  python -u reproduce_all.py --cpu-only     linear rows and CPU diagnostics
  python -u reproduce_all.py --gpu-only     network rows
  python -u reproduce_all.py --smoke        small version of all of it, on CPU
Finished experiments are skipped, so the same command also resumes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from grok.plotting import FIGURES
from grok.runner import execute, smoke

BANNER = "diagnostic: not the paper's stated setup"
CPU = ["fig2", "fig1l", "fig5", "fig6", "fig7", "fig3", "x1", "x4", "a1", "a2", "a3"]
GPU = ["fig1r", "x2", "fig4", "x3"]
DRAW = [  # figure function, output, experiments, banner
    ("fig1", "fig1.png", ["fig1l", "fig1r"], ""),
    ("fig2", "fig2.png", ["fig2"], ""),
    ("fig3", "fig3.png", ["fig3"], ""),
    ("fig4", "fig4.png", ["fig4"], ""),
    ("fig5", "fig5.png", ["fig5"], ""),
    ("fig6", "fig6.png", ["fig6"], ""),
    ("fig7", "fig7.png", ["fig7"], ""),
    ("fig7", "diagnostics/x1_fig7.png", ["x1"], BANNER),
    ("fig1", "diagnostics/x2_fig1.png", ["fig1l", "x2"], BANNER),
    ("fig4", "diagnostics/x3_fig4bc.png", ["x3"], BANNER),
    ("fig3", "diagnostics/x4_fig3.png", ["x4"], BANNER),
    ("fig1", "alternatives/a1_fig1l_identity.png", ["a1", "fig1r"], "alternative reading of a gap"),
    ("fig1", "alternatives/a2_fig1l_sphere.png", ["a2", "fig1r"], "alternative reading of a gap"),
    ("fig7", "alternatives/a3_fig7_independent.png", ["a3"], "alternative reading of a gap"),
]


def run_name(exp, is_smoke):
    prefix = "smoke_" if is_smoke else ("diag_" if exp.startswith("x") else "alt_" if exp.startswith("a") else "")
    return f"{prefix}{exp}" + ("" if is_smoke else "_full")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args()
    exps = (CPU if not a.gpu_only else []) + (GPU if not a.cpu_only else [])
    base = Path(a.runs_dir)
    for exp in exps:
        name = run_name(exp, a.smoke)
        if (base / name / "results.json").exists():
            print(f"skip {name}: finished", flush=True)
            continue
        if a.smoke:
            smoke(exp, name, a.runs_dir)
        else:
            execute(exp, name, a.runs_dir, resume=(base / name).exists(), device="cpu" if exp in CPU else "auto")
    root = Path("figures/smoke" if a.smoke else "figures")
    for fig, out, sources, banner in DRAW:
        runs = [base / run_name(s, a.smoke) for s in sources]
        if all((r / "results.json").exists() for r in runs):
            stamp = "smoke test: not a result" if a.smoke else banner
            cmd = [sys.executable, "make_figure.py", "--figure", fig, "--out", str(root / out), "--stamp", stamp,
                   "--runs", *map(str, runs)]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    assert FIGURES
    main()
