"""Deterministic smoke-sample selection compatible with legacy notebooks."""

from __future__ import annotations

import numpy as np


def smoke_indices(total_rows: int, size: int = 20, seed: int = 42) -> list[int]:
    """Match pandas sample(random_state=seed).sort_index()."""
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if size < 0 or size > total_rows:
        raise ValueError("size must be between zero and total_rows")
    random_state = np.random.RandomState(seed)
    return sorted(int(index) for index in random_state.choice(total_rows, size=size, replace=False))


def select_smoke_rows(rows: list[object], size: int = 20, seed: int = 42) -> list[object]:
    return [rows[index] for index in smoke_indices(len(rows), size=size, seed=seed)]
