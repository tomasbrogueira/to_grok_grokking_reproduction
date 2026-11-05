import numpy as np

from grok import metrics


def seq(f, grid):
    return np.array([f(int(t)) for t in grid])


def test_t1_t2_exact_on_monotone():
    f = lambda t: 2.0 * 0.99**t
    grid = metrics.search_grid(10**5, 200)
    t1 = metrics.last_above(seq(f, grid), grid, 0.01, f)
    t2 = metrics.first_at_or_below(seq(f, grid), grid, 0.01, f)
    brute = np.arange(2000)
    vals = seq(f, brute)
    assert t1 == brute[vals >= 0.01].max()
    assert t2 == brute[vals <= 0.01].min()


def test_definition_on_non_monotone_sequences():
    v = [5, 1, 4, 0.5, 2, 0.1, 0.05]
    assert metrics.last_above_sequence(v, 1) == 4  # last index at or above
    assert metrics.first_below_sequence(v, 1) == 1  # first index at or below


def test_t1_is_last_crossing_on_non_monotone_function():
    f = lambda t: 1.0 if t < 10 else (0.0 if t < 100 else (1.0 if t < 1000 else 0.0))
    grid = metrics.search_grid(10**5, 600)
    assert metrics.last_above(seq(f, grid), grid, 0.5, f) == 999


def test_never_reached_and_never_above():
    f = lambda t: 1.0
    grid = metrics.search_grid(1000, 50)
    assert metrics.first_at_or_below(seq(f, grid), grid, 0.1, f) is None
    assert metrics.last_above(seq(f, grid), grid, 5.0, f) is None


def test_curve_times_cover_paper_grid():
    ts = metrics.curve_times()
    assert ts[0] == 0
    for t in (1, 10, 100, 1000, 10**4, 15000, 10**5, 150000, 500000, 10**6, 1500000, 5000000):
        assert t in ts
