"""Online crossing tracker against brute force and against the replication's definitions."""
import numpy as np
import pytest
import torch

from grok import metrics
from grok.sgd.crossing import Tracker, crossing_times


def brute(values, thr):
    ge = [i for i, v in enumerate(values) if v >= thr]
    le = [i for i, v in enumerate(values) if v <= thr]
    gt = [i for i, v in enumerate(values) if v > thr]
    last = len(values) - 1
    t1 = None if not ge or ge[-1] == last else ge[-1]
    first = le[0] if le else None
    sustained = None if gt and gt[-1] == last else (gt[-1] + 1 if gt else 0)
    return t1, first, sustained


@pytest.mark.parametrize("cuts", [[500], [1, 499], [137, 200, 163], [100] * 5])
def test_matches_brute_force_on_noisy_series(cuts):
    g = torch.Generator().manual_seed(0)
    T, S = 500, 7
    decay = torch.exp(-torch.arange(T, dtype=torch.float64) / 60)[:, None]
    vals = decay * (1 + 0.8 * torch.rand((T, S), generator=g, dtype=torch.float64))  # non-monotone
    vals[:, 5] = 2.0  # never comes down
    vals[:, 6] = 1e-6  # never above
    thresholds = [0.005, 0.01, 0.05, 0.5]
    tr, t = Tracker(S, thresholds), 0
    for n in cuts:
        tr.update(list(range(t, t + n)), vals[t:t + n])
        t += n
    ex = tr.export()
    for thr in thresholds:
        got = crossing_times(ex, thr)
        for s in range(S):
            want = brute(vals[:, s].tolist(), thr)
            assert (got["t1"][s], got["first"][s], got["sustained"][s]) == want
    one = crossing_times(ex, 0.05)
    assert one["first"][0] < one["sustained"][0]  # the noise makes the two t2 definitions differ
    assert one["first"][5] is None and one["sustained"][5] is None and one["t1"][5] is None  # censored
    assert one["t1"][6] is None and one["first"][6] == 0 and one["sustained"][6] == 0


def test_agrees_with_replication_metrics_on_a_monotone_curve():
    f = lambda t: 2.0 * 0.99**t
    grid = metrics.search_grid(10**5, 200)
    seq = np.array([f(int(t)) for t in grid])
    tr = Tracker(1, [0.01])
    tr.update(list(range(3000)), torch.tensor([f(t) for t in range(3000)], dtype=torch.float64)[:, None])
    got = crossing_times(tr.export(), 0.01)
    assert got["t1"][0] == metrics.last_above(seq, grid, 0.01, f)
    assert got["first"][0] == metrics.first_at_or_below(seq, grid, 0.01, f)
    assert got["sustained"][0] == got["first"][0]  # monotone: both definitions of t2 agree


def test_sparse_times_and_state_round_trip():
    tr = Tracker(2, [1.0])
    tr.update([0, 10, 25], torch.tensor([[5.0, 0.5], [0.5, 3.0], [2.0, 0.2]], dtype=torch.float64))
    again = Tracker(2, [1.0])
    again.load({k: (v.clone() if torch.is_tensor(v) else v) for k, v in tr.state().items()})
    for t in (tr, again):
        t.update([40], torch.tensor([[0.1, 0.1]], dtype=torch.float64))
    assert tr.export() == again.export()
    got = crossing_times(tr.export(), 1.0)
    assert got == dict(t1=[25, 10], first=[10, 0], sustained=[26, 11])
    with pytest.raises(ValueError):
        tr.update([40], torch.tensor([[0.1, 0.1]], dtype=torch.float64))


def test_non_finite_values_count_as_above():
    tr = Tracker(1, [1.0])
    tr.update([0, 1, 2], torch.tensor([[5.0], [0.5], [float("nan")]], dtype=torch.float64))
    got = crossing_times(tr.export(), 1.0)
    assert got["t1"][0] is None and got["sustained"][0] is None  # diverged at the end: censored, not converged
    assert got["first"][0] == 1
