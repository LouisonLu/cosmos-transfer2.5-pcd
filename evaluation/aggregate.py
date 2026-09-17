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


def paired_delta_records(
    records: Iterable[dict[str, Any]],
    baseline_method: str,
    treatment_method: str,
    include_combined: bool = False,
    tie_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """Join methods by scene and calculate treatment minus baseline."""
    indexed: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        value = record.get("value")
        if record.get("status") != "ok" or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if record.get("method") not in (baseline_method, treatment_method):
            continue
        key = (record["scene_id"], record["split"], record["metric"], record["cohort"], record["method"])
        indexed[key] = record

    output: list[dict[str, Any]] = []
    base_keys = sorted({key[:4] for key in indexed})
    for scene_id, split, metric, cohort in base_keys:
        baseline = indexed.get((scene_id, split, metric, cohort, baseline_method))
        treatment = indexed.get((scene_id, split, metric, cohort, treatment_method))
        if baseline is None or treatment is None:
            continue
        higher_is_better = treatment.get("higher_is_better")
        if not isinstance(higher_is_better, bool):
            continue
        delta = float(treatment["value"]) - float(baseline["value"])
        if abs(delta) <= tie_tolerance:
            outcome = "tied"
        elif (higher_is_better and delta > 0) or (not higher_is_better and delta < 0):
            outcome = "improved"
        else:
            outcome = "worsened"
        output.append(
            {
                "scene_id": scene_id,
                "split": split,
                "cohort": cohort,
                "metric": metric,
                "baseline_method": baseline_method,
                "treatment_method": treatment_method,
                "baseline_value": float(baseline["value"]),
                "treatment_value": float(treatment["value"]),
                "delta": delta,
                "higher_is_better": higher_is_better,
                "improvement_direction": "delta > 0" if higher_is_better else "delta < 0",
                "outcome": outcome,
            }
        )
    if include_combined:
        output.extend(dict(row, split="combined") for row in list(output))
    return output


def aggregate_paired_deltas(
    deltas: Iterable[dict[str, Any]], bootstrap_samples: int = 0, seed: int = 2025
) -> list[dict[str, Any]]:
    """Summarize paired deltas and improved/worsened/tied counts."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        grouped[(row["baseline_method"], row["treatment_method"], row["split"], row["metric"])].append(row)

    generator = random.Random(seed)
    summaries: list[dict[str, Any]] = []
    for (baseline, treatment, split, metric), rows in sorted(grouped.items()):
        values = [float(row["delta"]) for row in rows]
        summary: dict[str, Any] = {
            "comparison": f"{treatment} - {baseline}",
            "baseline_method": baseline,
            "treatment_method": treatment,
            "split": split,
            "metric": metric,
            "valid_scenes": len(values),
            "mean_delta": statistics.fmean(values),
            "median_delta": statistics.median(values),
            "std_delta": statistics.stdev(values) if len(values) > 1 else 0.0,
            "improved": sum(row["outcome"] == "improved" for row in rows),
            "worsened": sum(row["outcome"] == "worsened" for row in rows),
            "tied": sum(row["outcome"] == "tied" for row in rows),
            "improvement_direction": rows[0]["improvement_direction"],
        }
        if bootstrap_samples > 0 and values:
            means = [statistics.fmean(generator.choices(values, k=len(values))) for _ in range(bootstrap_samples)]
            means.sort()
            summary["bootstrap_95ci_low"] = means[int(0.025 * (bootstrap_samples - 1))]
            summary["bootstrap_95ci_high"] = means[int(0.975 * (bootstrap_samples - 1))]
        summaries.append(summary)
    return summaries
