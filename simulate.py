"""Simulation utilities without external dependencies."""
from __future__ import annotations

import random
from typing import Iterable, List, Sequence

RANDOM_SEED = 42


def ct_independent(probs: Sequence[float]) -> float:
    result = 1.0
    for p in probs:
        if p < 0:
            p = 0.0
        if p > 1:
            p = 1.0
        result *= p
    return result


def plackett_luce_sample(weights: Sequence[float], rng: random.Random) -> List[int]:
    remaining = list(weights)
    indices = list(range(len(weights)))
    order: List[int] = []
    while indices:
        total = sum(remaining[i] for i in range(len(remaining)))
        if total <= 0:
            order.extend(indices)
            break
        threshold = rng.uniform(0, total)
        cumulative = 0.0
        for idx, weight in enumerate(remaining):
            cumulative += weight
            if cumulative >= threshold:
                order.append(indices[idx])
                remaining.pop(idx)
                indices.pop(idx)
                break
    return order


def estimate_ct_monte_carlo(probs: Sequence[float], mc_iters: int = 2000) -> float:
    rng = random.Random(RANDOM_SEED)
    weights = [max(p, 1e-6) for p in probs]
    success = 0
    for _ in range(mc_iters):
        ranking = plackett_luce_sample(weights, rng)
        if set(ranking[:3]) == {0, 1, 2}:
            success += 1
    return success / mc_iters if mc_iters > 0 else 0.0


def compute_ct(
    probs: Sequence[float],
    method: str = "independent",
    mc_iters: int = 2000,
) -> float:
    if method == "independent":
        return ct_independent(probs)
    if method == "mc":
        return estimate_ct_monte_carlo(probs, mc_iters)
    raise ValueError(f"Unknown CT estimation method: {method}")

