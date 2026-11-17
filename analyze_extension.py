"""Analyse a finished extension run against the frozen predictions.

  python analyze_extension.py --run runs/a1_full
Writes runs/<run-name>/analysis/: summary.md, summary.json and figures.
"""
import argparse

from grok.sgd.analysis import analyze


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, nargs="+", help="one or more run folders")
    a = ap.parse_args()
    for run in a.run:
        s = analyze(run)
        print(f"{run}: {s['label']}; {s['freeze_check']}")
        for p in s["predictions"]:
            print(f"  {p['id']}: {'smoke, not a result: ' if s['smoke'] else ''}{p['verdict']}")
        print(f"  written to {run}/analysis/")


if __name__ == "__main__":
    main()
