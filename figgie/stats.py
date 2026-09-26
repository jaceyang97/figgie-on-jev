"""Bootstrap confidence intervals, as in the paper's strategy comparison."""

from __future__ import annotations

import random


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, groups=None) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling whole games.

    With `groups` (one label per value, e.g. the game number), whole groups are
    resampled, so several values from one game count as one draw.
    """
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    clusters = {}
    for i, x in enumerate(xs):
        clusters.setdefault(i if groups is None else groups[i], []).append(x)
    clusters = list(clusters.values())
    k = len(clusters)
    means = []
    for _ in range(n_boot):
        draw = [x for _ in range(k) for x in clusters[rng.randrange(k)]]
        means.append(sum(draw) / len(draw))
    means.sort()
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
