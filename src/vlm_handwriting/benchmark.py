"""Model-independent benchmark row selection and summary helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from vlm_handwriting.smoke import select_smoke_rows

BENCHMARK_MODES = ("one", "smoke", "test")


def select_benchmark_rows(
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    mode: str,
    *,
    smoke_size: int = 20,
    smoke_seed: int = 42,
    allow_test: bool = False,
) -> tuple[list[dict[str, str]], str, str]:
    """Select fixed validation smoke rows while enforcing the frozen-test gate."""
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode: {mode!r}")

    smoke_rows = select_smoke_rows(validation_rows, size=smoke_size, seed=smoke_seed)
    if mode == "one":
        return smoke_rows[:1], "validation_one", "one"
    if mode == "smoke":
        return smoke_rows, f"validation_smoke{smoke_size}", f"smoke{smoke_size}"
    if not allow_test:
        raise PermissionError(
            "The frozen test set is gated. Freeze the inference configuration, then re-run "
            "with --allow-test."
        )
    return test_rows, "test", "test"


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    """Return stable latency summary statistics without a dataframe dependency."""
    if not latencies:
        raise ValueError("at least one latency is required")
    ordered = sorted(float(value) for value in latencies)
    p90_index = max(0, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "latency_mean_sec": statistics.fmean(ordered),
        "latency_p50_sec": statistics.median(ordered),
        "latency_p90_sec": ordered[p90_index],
    }


def prediction_metrics(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Extract aligned reference/prediction strings from artifact rows."""
    return (
        [str(row["ground_truth"]) for row in rows],
        [str(row["prediction"]) for row in rows],
    )
