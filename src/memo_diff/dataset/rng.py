"""Deterministic RNG helpers for reproducible dataset generation."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def derive_seed(*parts: object) -> int:
    """Derive a stable 63-bit seed from arbitrary parts."""
    material = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def make_rng(*parts: object) -> random.Random:
    return random.Random(derive_seed(*parts))


def choice(rng: random.Random, items: Sequence[T]) -> T:
    if not items:
        raise ValueError("Cannot choose from an empty sequence")
    return items[rng.randrange(len(items))]


def sample_unique(rng: random.Random, items: Sequence[T], k: int) -> list[T]:
    if k > len(items):
        raise ValueError(f"Cannot sample {k} unique items from {len(items)}")
    indices = list(range(len(items)))
    rng.shuffle(indices)
    return [items[i] for i in indices[:k]]


def weighted_choice(rng: random.Random, items: Sequence[T], weights: Sequence[float]) -> T:
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if threshold <= cumulative:
            return item
    return items[-1]


def stable_shuffle(items: Iterable[T], *seed_parts: object) -> list[T]:
    values = list(items)
    rng = make_rng(*seed_parts)
    rng.shuffle(values)
    return values
