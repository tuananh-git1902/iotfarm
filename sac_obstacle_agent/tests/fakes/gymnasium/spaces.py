import numpy as np


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is None:
            shape = np.shape(low)
        self.shape, self.dtype = tuple(shape), dtype
        self.low = np.broadcast_to(np.asarray(low, dtype=np.float64), self.shape)
        self.high = np.broadcast_to(np.asarray(high, dtype=np.float64), self.shape)
        self._rng = np.random.default_rng(0)

    def sample(self):
        lo, hi = np.clip(self.low, -1, 1), np.clip(self.high, -1, 1)
        return self._rng.uniform(lo, hi).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and np.all(x >= self.low - 1e-6) and np.all(x <= self.high + 1e-6)


class Dict:
    def __init__(self, spaces):
        self.spaces = dict(spaces)
