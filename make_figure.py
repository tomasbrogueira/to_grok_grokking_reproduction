"""Draw a figure from finished runs. Example: python make_figure.py --figure fig2 --runs runs/fig2_full"""
import argparse
from pathlib import Path

from grok.plotting import FIGURES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", required=True, choices=sorted(FIGURES))
    ap.add_argument("--runs", nargs="+", required=True, help="run folders; fig1 takes left then right")
    ap.add_argument("--out", help="output file, default figures/<figure>.png")
    ap.add_argument("--stamp", default="", help="banner text, used for diagnostics and smoke figures")
    a = ap.parse_args()
    out = Path(a.out or f"figures/{a.figure}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    FIGURES[a.figure](a.runs, out, a.stamp)
    print(out)


if __name__ == "__main__":
    main()
