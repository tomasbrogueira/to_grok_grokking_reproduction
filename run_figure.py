"""Run one experiment. Example: python -u run_figure.py --figure fig2 --run-name fig2_full"""
import argparse

from grok.runner import execute, smoke


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", required=True, help="config name: fig1l, fig1r, fig2 ... fig7, x1 ... x4, a1 ... a3")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--smoke", action="store_true", help="small CPU run: stop, resume, finish")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--seeds", type=int, help="override the seed count of the config")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args()
    if a.smoke:
        print(smoke(a.figure, a.run_name, a.runs_dir, "cpu" if a.device == "auto" else a.device))
    else:
        print(execute(a.figure, a.run_name, a.runs_dir, resume=a.resume, seeds=a.seeds, device=a.device))


if __name__ == "__main__":
    main()
