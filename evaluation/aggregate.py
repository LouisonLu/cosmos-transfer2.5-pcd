"""Scene-to-split aggregation for raw evaluation JSONL results."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable


def aggregate_records(
    records: Iterable[dict[str, Any]], bootstrap_samples: int = 0, seed: int = 2025, include_combined: bool = False
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        value = record.get("value")
        if record.get("status") != "ok" or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        grouped[(record["method"], record["split"], record["metric"])].append(float(value))

    if include_combined:
        combined: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (method, _split, metric), values in grouped.items():
            combined[(method, metric)].extend(values)
        for (method, metric), values in combined.items():
            grouped[(method, "combined", metric)] = values

    generator = random.Random(seed)
    summaries: list[dict[str, Any]] = []
    for (method, split, metric), values in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "method": method,
            "split": split,
            "metric": metric,
            "valid_scenes": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        if bootstrap_samples > 0 and values:
            means = [statistics.fmean(generator.choices(values, k=len(values))) for _ in range(bootstrap_samples)]
            means.sort()
            summary["bootstrap_95ci_low"] = means[int(0.025 * (bootstrap_samples - 1))]
            summary["bootstrap_95ci_high"] = means[int(0.975 * (bootstrap_samples - 1))]
        summaries.append(summary)
    return summaries
