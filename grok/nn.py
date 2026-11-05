"""Two-layer ReLU network of Section 5.3, all seeds of a condition in one batch.

Eq (1) with theta = (W, a): the penalty covers both layers. Gradients are written by
hand for speed; tests/test_nn.py checks them against autograd.
"""
import math
from pathlib import Path

import torch

from .data import DTYPE, randn
from .metrics import curve_times

# One test chunk allocates about three tensors of this size, so 400 MB peaks near 1.2 GB.
MAX_CHUNK_BYTES = {"cuda": 400 * 2**20, "cpu": 2 * 2**30}


class Interrupted(Exception):
    """Raised when a run stops on purpose (used by smoke and resume tests)."""


class Batch:
    def __init__(self, p, seeds, device):
        self.p, self.device, self.R = p, device, len(seeds)
        n, m, d, nt = p["n"], p["m"], p["d"], p["n_test"]
        if p["teacher"] != "zero":
            raise ValueError("Section 5.3 uses the zero teacher only")
        if p["x_dist"] != "gaussian":
            raise ValueError("NN runs use x ~ N(0, I_d) (U1)")
        Xs, Xts, Ws, As = [], [], [], []
        for s in seeds:
            g = torch.Generator().manual_seed(s)
            Xs.append(randn((n, d), g, device))
            Xts.append(randn((nt, d), g, device))
            Ws.append(randn((m, d), g, device) * math.sqrt(p["nu2"] / d))  # w ~ N(0, nu^2/d I_d)
            As.append(randn((m,), g, device) * math.sqrt(1 / m))  # a ~ N(0, 1/m)
        self.X, self.Xt = torch.stack(Xs), torch.stack(Xts)
        self.W, self.a = torch.stack(Ws), torch.stack(As)
        bytes_per_point = self.R * m * 8
        limit = MAX_CHUNK_BYTES["cuda" if str(device).startswith("cuda") else "cpu"]
        self.chunk = max(1, min(nt, limit // bytes_per_point))

    def forward_train(self):
        """Returns raw activations, active mask, hidden outputs and predictions."""
        pre = self.X @ self.W.transpose(1, 2)
        mask = pre > 0
        h = pre * mask
        return mask, h, (h * self.a[:, None, :]).sum(2)

    def grads(self, mask, h, out):
        n, lam = self.p["n"], self.p["lam"]
        r = out  # zero teacher: residual equals prediction
        ga = (h.transpose(1, 2) @ r[:, :, None])[:, :, 0] / n + lam * self.a
        weights = (mask * (r[:, :, None] * self.a[:, None, :])).transpose(1, 2)
        gw = (weights @ self.X) / n + lam * self.W
        return gw, ga

    def test_mse(self):
        total = torch.zeros(self.R, dtype=DTYPE, device=self.device)
        nt = self.Xt.shape[1]
        for i in range(0, nt, self.chunk):
            h = torch.relu(self.Xt[:, i : i + self.chunk] @ self.W.transpose(1, 2))
            total += ((h * self.a[:, None, :]).sum(2) ** 2).sum(1)
        return total / nt


def next_eval(t, near):
    """Evaluation schedule: 2% spacing on the plateau, 0.5% when close to the crossing (U12)."""
    return t + max(1, math.ceil((0.005 if near else 0.02) * t))


def run_condition(p, seeds, device, ckpt_path, log, on_metrics, interrupt_at=None):
    b = Batch(p, seeds, device)
    R, eps, c, eta = len(seeds), p["eps"], p["c"], p["eta"]
    grid = [t for t in curve_times(p["max_steps"])]
    ckpt_path = Path(ckpt_path)
    state = dict(t=0, nxt=0, last_ckpt=0, records=[],
                 t1=torch.full((R,), -1, dtype=torch.long, device=device),
                 t1_unhalved=torch.full((R,), -1, dtype=torch.long, device=device),
                 t2=torch.full((R,), -1, dtype=torch.long, device=device))
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        b.W, b.a = saved.pop("W"), saved.pop("a")
        state.update(saved)
        log(f"resumed at step {state['t']}")

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({**state, "W": b.W, "a": b.a}, tmp)
        tmp.replace(ckpt_path)

    finished = False
    while True:
        t = state["t"]
        mask, h, out = b.forward_train()
        mse = (out**2).mean(1)
        if not torch.isfinite(mse).all():
            log(f"non-finite loss at step {t}")
            break
        state["t1"][0.5 * mse >= eps] = t
        state["t1_unhalved"][mse >= eps] = t
        if t == state["nxt"]:
            test = b.test_mse()
            crossed = (state["t2"] < 0) & (test <= c)
            state["t2"][crossed] = t
            state["records"].append({"t": t, "train_mse": mse.tolist(), "test_mse": test.tolist()})
            on_metrics({"t": t, "train_mse_mean": mse.mean().item(), "test_mse_mean": test.mean().item()})
            open_seeds = state["t2"] < 0
            near = bool(open_seeds.any() and (test[open_seeds] <= 2 * c).any())
            nxt = next_eval(t, near)
            later = [g for g in grid if g > t]
            state["nxt"] = min(nxt, later[0]) if later else nxt
            if t - state["last_ckpt"] >= p["checkpoint_every"]:
                state["last_ckpt"] = t
                save()
                log(f"step {t}: train {mse.mean():.5f} test {test.mean():.5f}")
            if (state["t2"] >= 0).all() and (test < c / 2).all():
                finished = True
                break
        if t >= p["max_steps"]:
            finished = True
            break
        if interrupt_at is not None and t == interrupt_at:
            state["last_ckpt"] = t
            save()
            raise Interrupted(f"stopped at step {t}")
        gw, ga = b.grads(mask, h, out)
        b.W.add_(gw, alpha=-eta)
        b.a.add_(ga, alpha=-eta)
        state["t"] = t + 1
    save()
    none = lambda v: [int(x) if x >= 0 else None for x in v.tolist()]
    return dict(seeds=seeds, finished=finished, steps=state["t"], t1=none(state["t1"]),
                t1_unhalved=none(state["t1_unhalved"]), t2=none(state["t2"]),
                times=[r["t"] for r in state["records"]],
                train_mse=[r["train_mse"] for r in state["records"]],
                test_mse=[r["test_mse"] for r in state["records"]])
