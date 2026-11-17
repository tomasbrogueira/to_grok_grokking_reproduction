"""Paired seeds and stateless minibatch index streams.

Data seeds depend on the experiment, a problem group and the seed index, never on the arm,
so every arm of a group sees the same data, initialization and teacher. Batch seeds depend
on the arm as well. Batch indices are a pure function of (batch seed, step), so a resumed
run needs no generator state and a seed's stream does not depend on which other seeds are
simulated in the same batch.
"""
import hashlib

import torch

CHUNK_STEPS = 4096  # part of the stream definition: changing it changes the indices


def _h(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def data_seed(experiment, group, index):
    return _h(f"sgd-ext|{experiment}|data|{group}|{index}")


def batch_seed(experiment, arm, index):
    return _h(f"sgd-ext|{experiment}|batch|{arm}|{index}")


class BatchStream:
    """Indices [steps, seeds, b] for 'replace' (uniform with replacement) or 'reshuffle'
    (a fresh permutation of the n points per epoch, cut into n/b consecutive batches)."""

    def __init__(self, seeds, n, b, sampling, chunk_steps=CHUNK_STEPS):
        if sampling not in ("replace", "reshuffle"):
            raise ValueError(f"unknown sampling {sampling!r}")
        self.seeds, self.n, self.b, self.sampling = list(seeds), n, b, sampling
        if sampling == "reshuffle":
            if n % b:
                raise ValueError(f"reshuffling needs b to divide n, got n={n}, b={b}")
            self.per_epoch = n // b
            chunk_steps = max(1, chunk_steps // self.per_epoch) * self.per_epoch
        self.chunk_steps = chunk_steps
        self._cache = (None, None)

    def _chunk(self, k):
        if self._cache[0] == k:
            return self._cache[1]
        out = []
        for s in self.seeds:
            g = torch.Generator().manual_seed(_h(f"{s}|{self.sampling}|{self.n}|{self.b}|{k}"))
            if self.sampling == "replace":
                idx = torch.randint(self.n, (self.chunk_steps, self.b), generator=g)
            else:
                epochs = self.chunk_steps // self.per_epoch
                perm = torch.rand((epochs, self.n), generator=g, dtype=torch.float64).argsort(dim=1)
                idx = perm.reshape(self.chunk_steps, self.b)
            out.append(idx)
        chunk = torch.stack(out, dim=1)
        self._cache = (k, chunk)
        return chunk

    def block(self, t0, t1):
        """Indices used by the updates of steps t0, ..., t1-1."""
        parts, t = [], t0
        while t < t1:
            k, off = divmod(t, self.chunk_steps)
            take = min(t1 - t, self.chunk_steps - off)
            parts.append(self._chunk(k)[off:off + take])
            t += take
        return parts[0] if len(parts) == 1 else torch.cat(parts)
