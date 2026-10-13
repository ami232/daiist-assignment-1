import numpy as np


def block_bootstrap_sums(
    x: np.ndarray,
    block: int = 7,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    # moving-block bootstrap of sum(x), keeps autocorrelation
    n = len(x)
    if n == 0:
        return np.zeros(n_boot)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return x[idx].sum(axis=1)
