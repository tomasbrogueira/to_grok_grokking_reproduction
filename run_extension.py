"""Run one SGD-extension experiment (EXTENSION_EXPERIMENTS.md).

  python -u run_extension.py --experiment a1 --run-name a1_full
  python -u run_extension.py --experiment a1 --run-name a1_full --resume
  python -u run_extension.py --experiment b1 --run-name smoke_b1 --smoke
"""
import argparse

from grok.sgd.experiments import EXPERIMENTS
from grok.sgd.runner import execute, smoke


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--smoke", action="store_true", help="small CPU run: stop, resume, finish, analyse")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--seeds", type=int, help="override the seed count of the config")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda. auto is the CPU for a0-a3, the GPU if present for b1-b3")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args()
    if a.smoke:
        print(smoke(a.experiment, a.run_name, a.runs_dir, "cpu" if a.device == "auto" else a.device))
    else:
        print(execute(a.experiment, a.run_name, a.runs_dir, resume=a.resume, seeds=a.seeds, device=a.device))


if __name__ == "__main__":
    main()
