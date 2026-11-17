"""Setting B: minibatch SGD on the two-layer ReLU network of Section 5.3.

Data, initialization, full-batch gradient and test loss are grok.nn.Batch, unchanged, so the
full-batch arm follows the replication's trajectory. Added here: minibatch gradients, the split
W = W P_X + W_perp of Fact 2, the ablated test loss of B2, and per-step tracking of t1.

Update:  (W, a) <- (1 - eta lam) (W, a) - (eta / b) sum_{i in B} grad_i,  loss_i = f(x_i)^2 / 2.
"""
import math
from pathlib import Path

import torch

from .. import nn
from ..data import DTYPE
from ..metrics import curve_times
from .crossing import Tracker
from .seeds import BatchStream

FLUSH = 256  # steps of training loss buffered before the tracker sees them
EVAL_KEYS = ("train_mse", "test_mse", "test_mse_par", "null_out_mse", "flip_frac_test", "w_perp_norm",
             "w_par_norm", "a_norm", "w_perp_dev", "act_change_train")


class SGDNet(nn.Batch):
    def __init__(self, p, seeds, device):
        super().__init__(p, seeds, device)
        n, d = p["n"], p["d"]
        self.has_null = n < d  # for n >= d the inputs span R^d: P_X = I and W_perp = 0
        if self.has_null:
            Q = torch.linalg.qr(self.X.transpose(1, 2)).Q  # [R, d, n], orthonormal basis of span{x_i}
            self.P = Q @ Q.transpose(1, 2)
            self.Xt_par = self.Xt @ self.P  # (W P_X) x = W (P_X x): ablating W_perp is projecting the input
            self.W0_perp = self.w_perp().clone()
        self.mask0 = (self.X @ self.W.transpose(1, 2)) > 0

    def w_perp(self):
        return self.W - self.W @ self.P

    def grads_minibatch(self, mask, h, out, idx):
        """Gradient of (1/2b) sum_{i in idx} f(x_i)^2 + (lam/2)(|W|^2 + |a|^2). idx: [R, b], repeats allowed."""
        b, lam = idx.shape[1], self.p["lam"]
        r = out.gather(1, idx)
        ex = idx[:, :, None].expand(-1, -1, h.shape[2])
        hb, mb = h.gather(1, ex), mask.gather(1, ex)
        xb = self.X.gather(1, idx[:, :, None].expand(-1, -1, self.X.shape[2]))
        ga = (hb.transpose(1, 2) @ r[:, :, None])[:, :, 0] / b + lam * self.a
        gw = ((mb * (r[:, :, None] * self.a[:, None, :])).transpose(1, 2) @ xb) / b + lam * self.W
        return gw, ga

    def evaluate(self, t, mask, train_mse):
        """Test loss and the B2 diagnostics, one value per seed."""
        R, nt, zero = self.R, self.Xt.shape[1], torch.zeros(self.R, dtype=DTYPE, device=self.device)
        full, par, diff, flips = zero.clone(), zero.clone(), zero.clone(), zero.clone()
        Wt = self.W.transpose(1, 2)
        for i in range(0, nt, self.chunk):
            pre = self.Xt[:, i:i + self.chunk] @ Wt
            f = (torch.relu(pre) * self.a[:, None, :]).sum(2)
            full += (f**2).sum(1)
            if self.has_null:
                pre_par = self.Xt_par[:, i:i + self.chunk] @ Wt
                f_par = (torch.relu(pre_par) * self.a[:, None, :]).sum(2)
                par += (f_par**2).sum(1)
                diff += ((f - f_par) ** 2).sum(1)
                flips += ((pre > 0) != (pre_par > 0)).sum((1, 2))
        out = dict(train_mse=train_mse, test_mse=full / nt, a_norm=self.a.norm(dim=1),
                   act_change_train=(mask != self.mask0).to(DTYPE).mean((1, 2)))
        if self.has_null:
            wp = self.w_perp()
            rho_t = math.exp(t * math.log1p(-self.p["eta"] * self.p["lam"]))
            out.update(test_mse_par=par / nt, null_out_mse=diff / nt, flip_frac_test=flips / (nt * self.W.shape[1]),
                       w_perp_norm=wp.norm(dim=(1, 2)), w_par_norm=(self.W - wp).norm(dim=(1, 2)),
                       w_perp_dev=(wp - rho_t * self.W0_perp).norm(dim=(1, 2)) / self.W0_perp.norm(dim=(1, 2)))
        else:  # the ablation is the identity
            out.update(test_mse_par=out["test_mse"], null_out_mse=zero, flip_frac_test=zero, w_perp_norm=zero,
                       w_par_norm=self.W.norm(dim=(1, 2)), w_perp_dev=zero)
        return out


def predicted_t2(nu2, c, eta, lam):
    """Planning value for the step cap only: test loss ~ (nu^2/2)(1 - eta lam)^(4t), both layers decaying."""
    return math.log(0.5 * nu2 / c) / (-4 * math.log1p(-eta * lam))


def run_arm(p, seeds, batch_seeds, train_thresholds, test_thresholds, device, ckpt_path, log, on_metrics,
            interrupt_at=None):
    """Train one arm, all seeds in a batch. p holds the model settings plus b, sampling, max_steps, checkpoint_every.

    Training loss is tracked at every step. Test loss and diagnostics follow the replication's schedule
    (grok.nn.next_eval): 2% spacing, 0.5% once any seed is between c_min/2 and 2 c_max.
    """
    net = SGDNet(p, seeds, device)
    R, eta, n, b, sampling = len(seeds), p["eta"], p["n"], p["b"], p["sampling"]
    if sampling == "full" and b != n:
        raise ValueError("full-batch sampling needs b = n")
    stream = None if sampling == "full" else BatchStream(batch_seeds, n, b, sampling)
    c_min, c_max = min(test_thresholds), max(test_thresholds)
    grid = curve_times(p["max_steps"])
    tracker = Tracker(R, train_thresholds, device)
    ckpt_path = Path(ckpt_path)
    state = dict(t=0, nxt=0, last_ckpt=0, records=[], diverged=False)
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.W, net.a = saved.pop("W"), saved.pop("a")
        tracker.load(saved.pop("tracker"))
        state.update(saved)
        log(f"resumed at step {state['t']}")
    buf_t, buf_v = [], []

    def flush(before=None):
        """Hand buffered training losses to the tracker; with `before`, only those of earlier steps."""
        k = len(buf_t) if before is None else sum(1 for x in buf_t if x < before)
        if k:
            tracker.update(buf_t[:k], torch.stack(buf_v[:k]))
            del buf_t[:k], buf_v[:k]

    def save(final=False):
        # A checkpoint holds the parameters of step t, whose loss is computed again after a resume,
        # so the tracker inside it must stop at step t - 1.
        flush(None if final else state["t"])
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({**state, "W": net.W, "a": net.a, "tracker": tracker.state()}, tmp)
        tmp.replace(ckpt_path)

    idx_block, idx_start, finished = None, -1, False
    while True:
        t = state["t"]
        mask, h, out = net.forward_train()
        mse = (out**2).mean(1)
        buf_t.append(t)
        buf_v.append(mse)
        if len(buf_t) >= FLUSH:
            flush()
        if not torch.isfinite(mse).all():
            log(f"non-finite training loss at step {t}: arm is unstable")
            state["diverged"] = True
            break
        if t == state["nxt"]:
            ev = net.evaluate(t, mask, mse)
            state["records"].append({"t": t, **{k: ev[k].cpu() for k in EVAL_KEYS}})
            test = ev["test_mse"]
            on_metrics({"t": t, "train_mse_mean": mse.mean().item(), "test_mse_mean": test.mean().item(),
                        "test_mse_par_mean": ev["test_mse_par"].mean().item(), "a_norm_mean": ev["a_norm"].mean().item()})
            near = bool(((test <= 2 * c_max) & (test >= c_min / 2)).any())
            nxt = nn.next_eval(t, near)
            later = [g for g in grid if g > t]
            state["nxt"] = min(nxt, later[0]) if later else nxt
            if t - state["last_ckpt"] >= p["checkpoint_every"]:
                state["last_ckpt"] = t
                save()
                log(f"step {t}: train {mse.mean():.5f} test {test.mean():.5f}")
            if bool((test < c_min / 2).all()):
                finished = True
                break
        if t >= p["max_steps"]:
            break
        if interrupt_at is not None and t == interrupt_at:
            state["last_ckpt"] = t
            save()
            raise nn.Interrupted(f"stopped at step {t}")
        if stream is None:
            gw, ga = net.grads(mask, h, out)
        else:
            if not idx_start <= t < idx_start + (0 if idx_block is None else len(idx_block)):
                idx_block, idx_start = stream.block(t, t + 1024).to(device), t
            gw, ga = net.grads_minibatch(mask, h, out, idx_block[t - idx_start])
        net.W.add_(gw, alpha=-eta)
        net.a.add_(ga, alpha=-eta)
        state["t"] = t + 1
    save(final=True)
    rec = state["records"]
    curves = {"times": torch.tensor([r["t"] for r in rec])}
    curves.update({k: torch.stack([r[k] for r in rec]) for k in EVAL_KEYS})
    return dict(seeds=list(seeds), batch_seeds=list(batch_seeds), finished=finished, diverged=state["diverged"],
                steps=state["t"], has_null=net.has_null, train_tracker=tracker.export()), curves
