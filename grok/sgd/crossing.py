"""Crossing times of a per-seed loss series seen at increasing integer times.

For every threshold the tracker keeps three raw numbers per seed:
  last_ge   last time with value >= threshold   (t1 is this time, EXTENSION_EXPERIMENTS.md)
  first_le  first time with value <= threshold  (t2, first crossing)
  last_gt   last time with value >  threshold   (t2 sustained is this time + 1)
-1 means "never". Non-finite values count as +inf, so a diverged run can never look converged.
"""
import torch


class Tracker:
    def __init__(self, n_seeds, thresholds, device="cpu", dtype=torch.float64):
        self.thresholds = [float(x) for x in thresholds]
        self.thr = torch.tensor(self.thresholds, dtype=dtype, device=device)
        shape = (len(self.thresholds), n_seeds)
        self.last_ge = torch.full(shape, -1, dtype=torch.long, device=device)
        self.first_le = torch.full(shape, -1, dtype=torch.long, device=device)
        self.last_gt = torch.full(shape, -1, dtype=torch.long, device=device)
        self.last_t = -1

    def update(self, times, values):
        """times: [T] increasing integers after all earlier ones. values: [T, seeds]."""
        times = torch.as_tensor(times, dtype=torch.long, device=self.thr.device)
        if int(times[0]) <= self.last_t or (len(times) > 1 and bool((times[1:] <= times[:-1]).any())):
            raise ValueError("times must increase")
        v = torch.where(torch.isfinite(values), values, torch.full_like(values, float("inf")))
        order = torch.arange(1, len(times) + 1, device=times.device)[None, :, None]
        for name, mask in (("last_ge", v[None] >= self.thr[:, None, None]), ("last_gt", v[None] > self.thr[:, None, None])):
            pos = (mask * order).amax(dim=1)  # 0 if never in this block, else 1 + index of the last hit
            hit = pos > 0
            getattr(self, name)[hit] = times[(pos - 1).clamp_min(0)][hit]
        le = v[None] <= self.thr[:, None, None]
        any_le = le.any(dim=1)
        first = le.to(torch.uint8).argmax(dim=1)
        new = any_le & (self.first_le < 0)
        self.first_le[new] = times[first][new]
        self.last_t = int(times[-1])

    def state(self):
        return dict(last_ge=self.last_ge, first_le=self.first_le, last_gt=self.last_gt, last_t=self.last_t)

    def load(self, state):
        dev = self.thr.device
        self.last_ge, self.first_le, self.last_gt = (state[k].to(dev) for k in ("last_ge", "first_le", "last_gt"))
        self.last_t = state["last_t"]

    def export(self):
        """JSON-ready raw state: {threshold: {last_ge, first_le, last_gt}} plus the last time seen."""
        out = {}
        for i, thr in enumerate(self.thresholds):
            out[repr(thr)] = {k: getattr(self, k)[i].tolist() for k in ("last_ge", "first_le", "last_gt")}
        return dict(last_t=self.last_t, thresholds=out)


def t1_from(raw, last_t):
    """Last time at or above the threshold. None if never above, or still above at the end (censored)."""
    return None if raw < 0 or raw >= last_t else int(raw)


def first_from(raw):
    return None if raw < 0 else int(raw)


def sustained_from(raw_last_gt, last_t):
    """First time from which the series stays at or below the threshold. None if above at the end."""
    return None if raw_last_gt >= last_t else int(raw_last_gt) + 1


def crossing_times(export, threshold):
    """Per-seed lists of t1, first crossing and sustained crossing for one threshold."""
    e, last_t = export["thresholds"][repr(float(threshold))], export["last_t"]
    return dict(t1=[t1_from(r, last_t) for r in e["last_ge"]],
                first=[first_from(r) for r in e["first_le"]],
                sustained=[sustained_from(r, last_t) for r in e["last_gt"]])
