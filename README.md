# To Grok Grokking: reproduction and SGD extension

Code for reproducing *To Grok Grokking: Provable Grokking in Ridge Regression* (Xu, Vardi and Safran, 2026), and for an extension that asks what minibatch SGD does to the paper's weight-decay clock.

## Layout

| Path | Contents |
|---|---|
| `grok/` | Replication: data, exact GD in closed form (`spectral.py`), the two-layer network (`nn.py`), crossing times, run folders |
| `grok/sgd/` | Extension: SGD simulators for ridge regression (`linear.py`) and the network (`network.py`), experiment definitions, runner and analysis |
| `configs/` | Replication configs (`fig1l` … `fig7`, `alternatives/`, `diagnostics/`) |
| `configs/extension/` | Extension configs `a0` … `b3`. Each has a `smoke:` section with small settings |
| `anchors/` | Frozen numbers read off the paper (`anchors.json`) and the extension's frozen predictions (`extension_predictions.json`) |
| `tests/` | Unit tests, including checks against autograd and the closed form |
| `EXTENSION_EXPERIMENTS.md` | The extension's experiment plan |
| `COLAB.md` | How to run the extension on Google Colab |

## Setup

Python 3.10 or newer, then `pip install -r requirements.txt`. No data is downloaded: every dataset is generated from recorded seeds.

```bash
python -m pytest -q
```

## Replication

```bash
python -u run_figure.py --figure fig2 --run-name fig2_full        # one experiment
python -u reproduce_all.py --cpu-only                              # all CPU experiments, then figures
python make_figure.py --figure fig2 --runs runs/fig2_full          # redraw a figure
```

## Extension

The plan, its predictions and its decision rules are in `EXTENSION_EXPERIMENTS.md`. The predictions were frozen in `anchors/extension_predictions.json` before any full run; every run records that file's SHA-256, and the analysis reports whether it has changed since.

```bash
python -u run_extension.py --experiment a1 --run-name a1_full     # run
python -u run_extension.py --experiment a1 --run-name a1_full --resume
python analyze_extension.py --run runs/a1_full                     # writes runs/a1_full/analysis/
```

| Experiment | Question | Device |
|---|---|---|
| `a0` | Correctness checks: SGD at b = n against exact GD, the null-space identities, the approximations behind the predictions | CPU |
| `a1` | Does minibatching change the weight-decay clock in ridge regression? | CPU |
| `a2` | How many examples does grokking cost at each batch size's own stable step size? | CPU |
| `a3` | Can SGD noise delay grokking? | CPU |
| `b1` | Does the ridge-regression result carry over to the two-layer network? | GPU |
| `b2` | Does the paper's linear mechanism explain the network's delay? | GPU |
| `b3` | Larger step sizes in the network. Run only if B1 or B2 deviates | GPU |

The replication also has configs named `a1` to `a3` (alternative setups). They are run with `run_figure.py`, not `run_extension.py`.

## Conventions

- **Runs.** Every run writes to `runs/<run-name>/`: the resolved config with every seed, `log.txt`, `metrics.jsonl`, checkpoints and per-arm results. An existing run is never overwritten; `--resume` continues it from the latest checkpoint.
- **Smoke tests.** `--smoke` runs a small version on CPU in seconds. It stops part way, resumes, finishes and analyses. Its output is stamped as a smoke test and is not a result.
- **Paired seeds.** All arms of an extension experiment share data, initialization and teacher; only the batch sampling differs.
- **Device.** `--device auto` uses the GPU when one is present, except for setting A (`a0` to `a3`), whose small steps run faster on the CPU.
