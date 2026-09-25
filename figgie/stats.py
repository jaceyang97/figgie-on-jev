"""Bootstrap confidence intervals, as in the paper's strategy comparison."""

from __future__ import annotations

import random


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling whole games."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
