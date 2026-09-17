"""Reusable config-driven paper evaluator for Cosmos 360-video experiments.

The evaluator intentionally performs no downloads and discovers no files by
recursion.  The frozen manifest is the complete source of truth for scenes;
every predicted/reference/input path is resolved from the experiment config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.aggregate import aggregate_records
from evaluation.config import load_config, resolve_dataset_path, resolve_path, resolve_prediction_path
from evaluation.datasets.benchmark_manifest import BenchmarkScene, load_manifest, manifest_sha256
from evaluation.metrics.base import MetricPlugin, SceneContext
from evaluation.metrics.registry import plugins


RAW_NAME = "raw_metrics.jsonl"
PER_SCENE_NAME = "per_scene.csv"
SUMMARY_NAME = "summary.csv"
AUDIT_NAME = "audit.json"
MARKDOWN_NAME = "summary.md"


def parse_name_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [item for value in values for item in value.split(",") if item]


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def selected_plugins(registry: dict[str, MetricPlugin], values: list[str], config: dict[str, Any]) -> list[MetricPlugin]:
    requested = values or list(config.get("metrics", []))
    if not requested:
        raise ValueError("Select metrics with --metrics or config.metrics")
    names = sorted(registry) if requested == ["all"] else requested
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown)}. Available: {', '.join(sorted(registry))}")
    return [registry[name] for name in names]


def selected_methods(config: dict[str, Any], values: list[str]) -> dict[str, dict[str, Any]]:
    all_methods = config["methods"]
    names = values or list(all_methods)
    unknown = sorted(set(names) - set(all_methods))
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}. Available: {', '.join(sorted(all_methods))}")
    return {name: {"id": name, **all_methods[name]} for name in names}


def select_scenes(
    scenes: list[BenchmarkScene], splits: list[str], requested_scene_ids: list[str]
) -> list[BenchmarkScene]:
    selected = [scene for scene in scenes if scene.split in splits]
    if requested_scene_ids:
        wanted = set(requested_scene_ids)
        known = {scene.scene_id for scene in scenes}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown manifest scene IDs: {', '.join(unknown)}")
        selected = [scene for scene in selected if scene.scene_id in wanted]
    if not selected:
        raise ValueError("No scenes remain after --split/--scenes filtering")
    return selected


def make_context(config: dict[str, Any], method: dict[str, Any], scene: BenchmarkScene) -> SceneContext:
    dataset = config["datasets"].get(scene.split)
    if not isinstance(dataset, dict):
        raise ValueError(f"Config datasets is missing split '{scene.split}'")
    return SceneContext(
        scene_id=scene.scene_id,
        stem=scene.stem,
        split=scene.split,
        cohort=scene.cohort,
        method=method["id"],
        prediction=resolve_prediction_path(method, scene.split, scene.cohort, scene.stem),
        reference=resolve_dataset_path(dataset, "reference_root", scene.stem, ".mp4"),
        input_rgb=resolve_dataset_path(dataset, "input_rgb_root", scene.stem, ".mp4"),
        mask=resolve_dataset_path(dataset, "mask_root", scene.stem, "_mask.mp4"),
        policy=config["evaluation"],
    )


def required_input_problems(plugin: MetricPlugin, context: SceneContext) -> list[str]:
    problems = []
    if not context.prediction.is_file() or context.prediction.stat().st_size == 0:
        problems.append(f"missing prediction: {context.prediction}")
    if plugin.spec.requires_gt and (context.reference is None or not context.reference.is_file() or context.reference.stat().st_size == 0):
        problems.append(f"missing reference: {context.reference}")
    if plugin.spec.requires_mask and (context.mask is None or not context.mask.is_file() or context.mask.stat().st_size == 0):
        problems.append(f"missing mask: {context.mask}")
    if plugin.spec.requires_input_frame and (
        context.input_rgb is None or not context.input_rgb.is_file() or context.input_rgb.stat().st_size == 0
    ):
        problems.append(f"missing input RGB: {context.input_rgb}")
    if plugin.spec.requires_full_360_gt and context.policy.get("projection_convention") != "equirectangular_360":
        problems.append("metric requires full 360 GT but evaluation.projection_convention is not equirectangular_360")
    return problems


def base_record(plugin: MetricPlugin, context: SceneContext, metric: str, status: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "scene_id": context.scene_id,
        "stem": context.stem,
        "split": context.split,
        "cohort": context.cohort,
        "method": context.method,
        "plugin": plugin.spec.name,
        "metric": metric,
        "higher_is_better": plugin.spec.higher_is_better,
        "status": status,
        "reason": reason,
        "prediction": str(context.prediction),
        "reference": str(context.reference) if context.reference is not None else None,
        "input_rgb": str(context.input_rgb) if context.input_rgb is not None else None,
        "mask": str(context.mask) if context.mask is not None else None,
        "value": None,
    }


def has_completed_plugin(records: list[dict[str, Any]], plugin: MetricPlugin, context: SceneContext) -> bool:
    expected = set(plugin.spec.output_metrics)
    completed = {
        record["metric"]
        for record in records
        if record.get("status") == "ok"
        and record.get("scene_id") == context.scene_id
        and record.get("method") == context.method
        and record.get("plugin") == plugin.spec.name
    }
    return expected <= completed


def evaluate_plugin(plugin: MetricPlugin, context: SceneContext) -> list[dict[str, Any]]:
    availability = plugin.availability()
    if not availability.available:
        return [base_record(plugin, context, plugin.spec.name, "unavailable", availability.reason)]
    problems = required_input_problems(plugin, context)
    if problems:
        return [base_record(plugin, context, plugin.spec.name, "skipped", "; ".join(problems))]
    if plugin.spec.granularity != "scene":
        return [
            base_record(
                plugin,
                context,
                plugin.spec.name,
                "unavailable",
                "Dataset-level execution is deliberately disabled until this plugin has a pinned official implementation.",
            )
        ]
    try:
        values = plugin.evaluate_scene(context)
    except Exception as error:  # a scene failure must not hide later manifest items
        return [base_record(plugin, context, plugin.spec.name, "failed", f"{type(error).__name__}: {error}")]
    missing_outputs = set(plugin.spec.output_metrics) - set(values)
    if missing_outputs:
        return [base_record(plugin, context, plugin.spec.name, "failed", f"Plugin omitted outputs: {sorted(missing_outputs)}")]
    records = []
    for metric, value in values.items():
        record = base_record(plugin, context, metric, "ok")
        record["value"] = value
        if isinstance(value, float) and not math.isfinite(value):
            record["special_value"] = "infinity"
        records.append(record)
    return records


def markdown_summary(config: dict[str, Any], summaries: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation Summary: {config['name']}",
        "",
        f"- Git commit: `{audit['git_commit']}`",
        f"- Manifest SHA256: `{audit['manifest_sha256']}`",
        f"- Projection: `{audit['evaluation_policy'].get('projection_convention')}`",
        f"- Frame policy: `{audit['evaluation_policy'].get('frame_sampling')}`",
        f"- Evaluated scenes: {len(audit['selected_scenes'])}",
        "",
        "| Method | Split | Metric | Valid scenes | Mean | Median | Std | 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        ci = ""
        if "bootstrap_95ci_low" in summary:
            ci = f"[{summary['bootstrap_95ci_low']:.6f}, {summary['bootstrap_95ci_high']:.6f}]"
        lines.append(
            f"| {summary['method']} | {summary['split']} | {summary['metric']} | {summary['valid_scenes']} | "
            f"{summary['mean']:.6f} | {summary['median']:.6f} | {summary['std']:.6f} | {ci} |"
        )
    if not summaries:
        lines.append("| No finite successful metrics were produced. | | | | | | | |")
    lines.extend(["", "## Status Counts", ""])
    for status, count in sorted(audit["status_counts"].items()):
        lines.append(f"- `{status}`: {count}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML. Also accepts a path relative to evaluation/.")
    parser.add_argument("--metrics", nargs="+", help="Metric names, comma-separated names, or exactly 'all'.")
    parser.add_argument("--methods", nargs="+", help="Method names declared by the config.")
    parser.add_argument("--split", nargs="+", choices=("id", "ood"), default=["id", "ood"])
    parser.add_argument("--scenes", nargs="+", help="Optional manifest scene ID subset.")
    parser.add_argument("--output-dir", help="Override config.output_dir.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and write an audit without decoding videos or scoring.")
    parser.add_argument("--resume", action="store_true", help="Reuse only successful scene/plugin records from raw_metrics.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the framework's known output files in the selected output directory.")
    parser.add_argument("--bootstrap-samples", type=int, default=None, help="Override evaluation.bootstrap_samples; 0 disables CI.")
    parser.add_argument("--combined", action="store_true", help="Also report an explicitly requested ID+OOD combined summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path, config = load_config(args.config)
    manifest_path = resolve_path(config_path, config["manifest"])
    manifest = load_manifest(manifest_path)
    registry = plugins()
    metric_plugins = selected_plugins(registry, parse_name_list(args.metrics), config)
    methods = selected_methods(config, parse_name_list(args.methods))
    scenes = select_scenes(manifest, args.split, parse_name_list(args.scenes))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else resolve_path(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    known_output_paths = [output_dir / name for name in (RAW_NAME, PER_SCENE_NAME, SUMMARY_NAME, AUDIT_NAME, MARKDOWN_NAME)]
    raw_path = output_dir / RAW_NAME
    if args.overwrite:
        for path in known_output_paths:
            if path.exists():
                path.unlink()
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"Existing results: {raw_path}. Use --resume or --overwrite.")

    previous_records = read_jsonl(raw_path) if args.resume else []
    planned_contexts = [make_context(config, method, scene) for method in methods.values() for scene in scenes]
    metric_audit = {
        plugin.spec.name: {
            "spec": {
                "name": plugin.spec.name,
                "version": plugin.spec.version,
                "higher_is_better": plugin.spec.higher_is_better,
                "requires_GT": plugin.spec.requires_gt,
                "requires_full_360_GT": plugin.spec.requires_full_360_gt,
                "requires_mask": plugin.spec.requires_mask,
                "requires_input_frame": plugin.spec.requires_input_frame,
                "requires_multiple_generated_views": plugin.spec.requires_multiple_generated_views,
                "required_dependencies": plugin.spec.required_dependencies,
                "expected_input": plugin.spec.expected_input,
                "output_metrics": plugin.spec.output_metrics,
                "granularity": plugin.spec.granularity,
            },
            "availability": plugin.availability().__dict__,
        }
        for plugin in metric_plugins
    }
    audit: dict[str, Any] = {
        "framework": "evaluation.run_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": sha256_json(config),
        "resolved_config": config,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256(manifest_path),
        "evaluation_policy": config["evaluation"],
        "selected_scenes": [scene.__dict__ for scene in scenes],
        "selected_methods": list(methods),
        "selected_metrics": [plugin.spec.name for plugin in metric_plugins],
        "metric_plugins": metric_audit,
        "dry_run": args.dry_run,
        "resumed_successful_plugins": [],
        "planned_inputs": [],
        "status_counts": {},
    }
    for context in planned_contexts:
        audit["planned_inputs"].append(
            {
                "scene_id": context.scene_id,
                "method": context.method,
                "prediction": str(context.prediction),
                "reference": str(context.reference) if context.reference else None,
                "input_rgb": str(context.input_rgb) if context.input_rgb else None,
                "mask": str(context.mask) if context.mask else None,
            }
        )

    new_records: list[dict[str, Any]] = []
    if args.dry_run:
        for context in planned_contexts:
            for plugin in metric_plugins:
                availability = plugin.availability()
                if not availability.available:
                    new_records.append(base_record(plugin, context, plugin.spec.name, "unavailable", availability.reason))
                    continue
                problems = required_input_problems(plugin, context)
                if problems:
                    new_records.append(base_record(plugin, context, plugin.spec.name, "skipped", "; ".join(problems)))
                    continue
                for metric in plugin.spec.output_metrics:
                    new_records.append(base_record(plugin, context, metric, "dry_run", "Input validation passed; computation not started."))
    else:
        for context in planned_contexts:
            for plugin in metric_plugins:
                if args.resume and has_completed_plugin(previous_records, plugin, context):
                    audit["resumed_successful_plugins"].append(
                        {"scene_id": context.scene_id, "method": context.method, "plugin": plugin.spec.name}
                    )
                    continue
                new_records.extend(evaluate_plugin(plugin, context))
    records = previous_records + new_records
    status_counts = Counter(record["status"] for record in records)
    audit["status_counts"] = dict(sorted(status_counts.items()))
    summaries = aggregate_records(
        records,
        bootstrap_samples=args.bootstrap_samples
        if args.bootstrap_samples is not None
        else int(config["evaluation"].get("bootstrap_samples", 0)),
        seed=int(config["evaluation"].get("random_seed", 2025)),
        include_combined=args.combined,
    )
    write_jsonl(raw_path, records)
    write_csv(
        output_dir / PER_SCENE_NAME,
        records,
        ["scene_id", "stem", "split", "cohort", "method", "plugin", "metric", "higher_is_better", "status", "reason", "value", "special_value", "prediction", "reference", "input_rgb", "mask"],
    )
    write_csv(
        output_dir / SUMMARY_NAME,
        summaries,
        ["method", "split", "metric", "valid_scenes", "mean", "median", "std", "bootstrap_95ci_low", "bootstrap_95ci_high"],
    )
    write_json(output_dir / AUDIT_NAME, audit)
    (output_dir / MARKDOWN_NAME).write_text(markdown_summary(config, summaries, audit), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(f"scenes={len(scenes)} methods={len(methods)} metrics={len(metric_plugins)} dry_run={args.dry_run}")
    print("status_counts=" + json.dumps(audit["status_counts"], sort_keys=True))
    has_input_problem = status_counts.get("skipped", 0) > 0
    return 0 if status_counts.get("failed", 0) == 0 and not has_input_problem else 1


if __name__ == "__main__":
    raise SystemExit(main())
