"""Paired seeds and minibatch index streams of the SGD extension."""
import pytest
import torch

from grok.sgd.seeds import BatchStream, batch_seed, data_seed


def test_data_seed_ignores_the_arm_and_batch_seed_does_not():
    assert data_seed("a1", "n=100", 3) == data_seed("a1", "n=100", 3)
    assert data_seed("a1", "n=100", 3) != data_seed("a1", "n=100", 4)
    assert data_seed("a1", "n=100", 3) != data_seed("a2", "n=100", 3)
    assert data_seed("a2", "n=100", 3) != data_seed("a2", "n=300", 3)
    assert batch_seed("a1", "steps_b1", 3) != batch_seed("a1", "steps_b10", 3)
    assert batch_seed("a1", "steps_b1", 3) != data_seed("a1", "steps_b1", 3)


@pytest.mark.parametrize("sampling,n,b", [("replace", 20, 3), ("reshuffle", 20, 5), ("reshuffle", 12, 1)])
def test_stream_is_a_pure_function_of_seed_and_step(sampling, n, b):
    seeds = [11, 22, 33]
    whole = BatchStream(seeds, n, b, sampling, chunk_steps=64).block(0, 300)
    assert whole.shape == (300, 3, b)
    pieces = BatchStream(seeds, n, b, sampling, chunk_steps=64)
    cut = torch.cat([pieces.block(0, 7), pieces.block(7, 130), pieces.block(130, 300)])
    assert torch.equal(whole, cut)  # how the steps are cut into blocks does not matter
    late = BatchStream(seeds, n, b, sampling, chunk_steps=64).block(250, 300)
    assert torch.equal(whole[250:], late)  # a resumed run sees the same indices
    alone = BatchStream([22], n, b, sampling, chunk_steps=64).block(0, 300)
    assert torch.equal(whole[:, 1:2], alone)  # a seed's stream does not depend on its batch mates
    assert not torch.equal(whole[:, 0], whole[:, 1])


def test_replace_is_uniform_with_duplicates():
    idx = BatchStream([5], 10, 10, "replace").block(0, 4000)[:, 0]
    assert idx.min() == 0 and idx.max() == 9
    counts = torch.bincount(idx.flatten(), minlength=10).double() / idx.numel()
    assert (counts - 0.1).abs().max() < 0.01
    distinct = torch.tensor([len(set(row.tolist())) for row in idx[:500]]).double().mean()
    assert distinct == pytest.approx(10 * (1 - 0.9**10), rel=0.05)  # b = n with replacement is NOT the full batch


def test_reshuffle_passes_over_every_point_once_per_epoch():
    n, b = 20, 5
    idx = BatchStream([5], n, b, "reshuffle", chunk_steps=64).block(0, 400)[:, 0]
    epochs = idx.reshape(100, n)
    assert all(sorted(e.tolist()) == list(range(n)) for e in epochs)
    assert len({tuple(e.tolist()) for e in epochs}) == 100  # a new permutation every epoch


def test_reshuffle_needs_b_to_divide_n():
    with pytest.raises(ValueError):
        BatchStream([1], 10, 3, "reshuffle")
    with pytest.raises(ValueError):
        BatchStream([1], 10, 2, "sequential")
