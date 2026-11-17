# Running the SGD extension on Google Colab

Run folders go to Google Drive, so a disconnected session loses at most the steps since the last checkpoint. Colab already has every package the code needs.

## 1. Start a session

Use **Runtime → Change runtime type**: a CPU runtime is enough for A1 to A3; B1 to B3 need a GPU. Then run:

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone -b sgd-extension https://github.com/tomasbrogueira/to_grok_grokking_reproduction.git /content/grok
%cd /content/grok
RUNS = "/content/drive/MyDrive/grok_runs"
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
```

If the repository is private, clone with a GitHub personal access token: `https://<token>@github.com/...`. Do not save the notebook with the token in it.

## 2. Check the session (about 2 minutes)

```python
!python -m pytest -q tests/test_sgd_*.py
!python -u run_extension.py --experiment b1 --run-name smoke_b1 --smoke --runs-dir /content/smoke --device cuda
```

All tests should pass. On a GPU runtime the smoke run should end with `finished: smoke test`; on a CPU runtime drop `--device cuda`. The smoke output is a code check, not a result. The GPU path has not been run before this, so stop here if it fails.

## 3. Run

Run one experiment per cell, in this order. Each prints its progress and writes `MyDrive/grok_runs/<run-name>/`.

```python
!python -u run_extension.py --experiment a1 --run-name a1_full --runs-dir "{RUNS}"
!python -u run_extension.py --experiment a2 --run-name a2_full --runs-dir "{RUNS}"
!python -u run_extension.py --experiment a3 --run-name a3_full --runs-dir "{RUNS}"
!python -u run_extension.py --experiment b1 --run-name b1_full --runs-dir "{RUNS}"
!python -u run_extension.py --experiment b2 --run-name b2_full --runs-dir "{RUNS}"
```

Run B3 only if B1 or B2 deviates from the linear picture (plan, section B3):

```python
!python -u run_extension.py --experiment b3 --run-name b3_full --runs-dir "{RUNS}"
```

**After a disconnect**, mount Drive and repeat the setup cell, then rerun the interrupted command with `--resume` added. Finished arms are skipped. Without `--resume` the command refuses to touch an existing run, so nothing is overwritten by accident.

**If GPU time runs short**, B1 and B2 may use 20 seeds, as the plan allows: add `--seeds 20` to the command, and keep it when resuming.

## 4. Analyse

```python
!python analyze_extension.py --run "{RUNS}/a1_full" "{RUNS}/a2_full" "{RUNS}/a3_full" "{RUNS}/b1_full" "{RUNS}/b2_full"
```

Add `"{RUNS}/b3_full"` if B3 was run.

Each run gets an `analysis/` folder with `summary.md`, `summary.json` and figures. The first lines of `summary.md` give the verdict for every frozen prediction and confirm that the predictions file is the one recorded when the run started.

## 5. Bring the results back

Copy the run folders from `MyDrive/grok_runs/` into `runs/` of the local project, keeping their names. The checkpoints are not needed for the analysis and can be left out.

## Expected cost (estimate)

These are planning figures from step counts and CPU timing in the local sandbox. GPU speed has not been measured: check the step rate in the log after the first 10 minutes. The network runs in float64, and T4 GPUs are slow at float64, so an A100 or L4 runtime is several times faster when one is available.

| Experiment | Runtime | Time (estimate) | Memory (estimate) |
|---|---|---|---|
| A1 | CPU | 30 to 60 minutes | under 1 GB |
| A2 | CPU | 20 to 40 minutes | under 1 GB |
| A3 | CPU | 10 to 20 minutes | under 1 GB |
| B1 | GPU (T4) | 1.5 to 3 hours | about 2 to 3 GB of GPU memory |
| B2 | GPU (T4) | 2 to 4 hours | about 2 to 3 GB of GPU memory |
| B3 | GPU (T4) | about 1 hour | about 2 GB of GPU memory |

The network's per-step cost measured on 11 CPU threads is 16 ms for B1 (50 seeds); a B1 arm runs about 2.7e5 steps.
