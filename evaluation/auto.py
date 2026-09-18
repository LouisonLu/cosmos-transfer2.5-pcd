"""Auto-discovered evaluation mode for explicitly supplied prediction folders."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


KNOWN_OUTPUT_NAMES = ("raw_metrics.jsonl", "per_scene.csv", "summary.csv", "paired_deltas.csv", "audit.json", "summary.md")


def parse_labeled_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Expected non-empty LABEL=PATH, got: {value}")
    return label, Path(raw_path).expanduser().resolve()


def discover_videos(root: Path, max_depth: int = 4) -> list[Path]:
    """Discover direct-child MP4s through a bounded directory depth."""
    if not root.is_dir():
        raise ValueError(f"Missing data directory: {root}")
    discovered: list[Path] = []
    frontier = [root]
    visited: set[Path] = set()
    for depth in range(max_depth + 1):
        next_frontier: list[Path] = []
        for directory in frontier:
            directory = directory.resolve()
            if directory in visited:
                continue
            visited.add(directory)
            for entry in sorted(directory.iterdir(), key=lambda item: item.name):
                if entry.is_file() and entry.suffix.lower() == ".mp4":
                    if not entry.name.endswith("_input_surrogate.mp4"):
                        discovered.append(entry)
                elif entry.is_dir() and depth < max_depth:
                    next_frontier.append(entry)
        frontier = next_frontier
    return discovered


def normalized_auxiliary_stem(stem: str) -> str:
    for suffix in ("_mask", "_prompt"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_path_index(paths: list[Path], auxiliary: bool = False) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in paths:
        key = normalized_auxiliary_stem(path.stem) if auxiliary else path.stem
        if key in index:
            raise ValueError(f"Duplicate scene key '{key}': {index[key]} and {path}")
        index[key] = path
    return index


def match_path(prediction_stem: str, candidates: dict[str, Path]) -> Path | None:
    exact = candidates.get(prediction_stem)
    if exact is not None:
        return exact
    matches = [
        (key, path)
        for key, path in candidates.items()
        if prediction_stem.startswith(key)
        and prediction_stem[len(key) :].startswith(("_", "__"))
    ]
    if not matches:
        return None
    longest = max(len(key) for key, _path in matches)
    best = [(key, path) for key, path in matches if len(key) == longest]
    if len(best) != 1:
        joined = ", ".join(str(path) for _key, path in best)
        raise ValueError(f"Ambiguous reference match for '{prediction_stem}': {joined}")
    return best[0][1]


def infer_split_and_cohort(path: Path, root: Path, default_split: str) -> tuple[str, str]:
    parts = {part.lower() for part in path.relative_to(root).parts}
    if "ood" in parts or "ood_nuscenes" in parts:
        split = "ood"
    elif "test" in parts:
        split = "id"
    else:
        split = default_split
    if "different_scene" in parts:
        cohort = "different_scene"
    elif "different_pose" in parts:
        cohort = "different_pose"
    else:
        cohort = "all"
    return split, cohort


def validate_video_pair(context: Any, needs_reference: bool, needs_input_and_mask: bool) -> list[str]:
    """Validate video properties before metric execution starts."""
    from evaluation.video import frame_count_from_policy, has_opencv, open_video, require_exact_match

    if not has_opencv():
        return []
    requested = frame_count_from_policy(context.policy)
    problems: list[str] = []
    try:
        with open_video(context.prediction) as (_capture, prediction_info):
            if prediction_info.reported_frames < requested:
                problems.append(f"prediction has {prediction_info.reported_frames} frames, needs {requested}: {context.prediction}")
            if needs_reference:
                if context.reference is None:
                    problems.append(f"missing paired GT: {context.scene_id}")
                else:
                    with open_video(context.reference) as (_ref_capture, reference_info):
                        require_exact_match(prediction_info, reference_info, "GT preflight")
                        if reference_info.reported_frames < requested:
                            problems.append(f"GT has {reference_info.reported_frames} frames, needs {requested}: {context.reference}")
            if needs_input_and_mask:
                if context.input_rgb is None or context.mask is None:
                    problems.append(f"missing input RGB or mask: {context.scene_id}")
                else:
                    with open_video(context.input_rgb) as (_input_capture, input_info), open_video(
                        context.mask
                    ) as (_mask_capture, mask_info):
                        require_exact_match(prediction_info, input_info, "Input RGB preflight")
                        require_exact_match(prediction_info, mask_info, "Mask preflight")
    except Exception as error:
        problems.append(f"video preflight failed for {context.scene_id}: {type(error).__name__}: {error}")
    return problems


def run_auto(args: Any) -> int:
    from evaluation.aggregate import aggregate_paired_deltas, aggregate_records, paired_delta_records
    from evaluation.metrics.registry import plugins
    from evaluation.run_eval import (
        AUDIT_NAME,
        MARKDOWN_NAME,
        PAIRED_NAME,
        PER_SCENE_NAME,
        RAW_NAME,
        SUMMARY_NAME,
        EvaluationProgress,
        base_record,
        evaluate_plugin,
        git_commit,
        has_completed_plugin,
        json_safe,
        markdown_summary,
        read_jsonl,
        required_input_problems,
        selected_plugins,
        sha256_json,
        write_csv,
        write_json,
        write_jsonl,
    )

    if not args.data_root:
        raise ValueError("Auto mode requires at least one --data-root LABEL=PATH")
    if not args.output_dir:
        raise ValueError("Auto mode requires --output-dir")
    if args.config:
        raise ValueError("Do not combine --config with --data-root auto mode")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    known_output_paths = [output_dir / name for name in KNOWN_OUTPUT_NAMES]
    raw_path = output_dir / RAW_NAME
    if args.overwrite:
        for path in known_output_paths:
            if path.exists():
                path.unlink()
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"Existing results: {raw_path}. Use --resume or --overwrite.")

    labeled_roots: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in args.data_root:
        label, root = parse_labeled_root(value)
        if label in labels:
            raise ValueError(f"Duplicate data-root label: {label}")
        labels.add(label)
        labeled_roots.append((label, root))

    reference_roots = [Path(value).expanduser().resolve() for value in (args.reference_root or [])]
    input_roots = [Path(value).expanduser().resolve() for value in (args.input_root or [])]
    mask_roots = [Path(value).expanduser().resolve() for value in (args.mask_root or [])]
    reference_paths = [path for root in reference_roots for path in discover_videos(root, args.max_depth)]
    input_paths = [path for root in input_roots for path in discover_videos(root, args.max_depth)]
    mask_paths = [path for root in mask_roots for path in discover_videos(root, args.max_depth)]
    reference_index = build_path_index(reference_paths) if reference_paths else None
    input_index = build_path_index(input_paths) if input_paths else None
    mask_index = build_path_index(mask_paths, auxiliary=True) if mask_paths else None

    registry = plugins()
    metric_config = {"metrics": args.metrics or []}
    metric_plugins = selected_plugins(registry, [item for value in (args.metrics or []) for item in value.split(",")], metric_config)
    availability_by_metric = {plugin.spec.name: plugin.availability() for plugin in metric_plugins}
    policy = {
        "projection_convention": "equirectangular_360",
        "frame_count": args.frame_count,
        "frame_sampling": "all_frames_0_to_requested_minus_1_in_file_order",
        "resize_policy": "exact_match_required_no_resize",
        "interpolation_policy": "none",
        "image_value_normalization": "RGB_float32_0_to_1",
        "random_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples or 0,
    }

    from evaluation.metrics.base import SceneContext

    contexts: list[SceneContext] = []
    method_counts: dict[str, int] = {}
    preflight_errors: list[str] = []
    context_keys: dict[tuple[str, str, str, str], Path] = {}
    for label, root in labeled_roots:
        videos = discover_videos(root, args.max_depth)
        if not videos:
            preflight_errors.append(f"No MP4 predictions found for {label}: {root}")
            method_counts[label] = 0
            continue
        method_counts[label] = len(videos)
        for prediction in videos:
            reference = match_path(prediction.stem, reference_index) if reference_index is not None else None
            input_rgb = match_path(prediction.stem, input_index) if input_index is not None else None
            mask = match_path(prediction.stem, mask_index) if mask_index is not None else None
            split, cohort = infer_split_and_cohort(prediction, root, args.default_split)
            scene_id = reference.stem if reference is not None else prediction.stem
            key = (scene_id, split, cohort, label)
            if key in context_keys:
                preflight_errors.append(f"Duplicate prediction key for {label}: {scene_id}")
                continue
            context_keys[key] = prediction
            contexts.append(
                SceneContext(
                    scene_id=scene_id,
                    stem=scene_id,
                    split=split,
                    cohort=cohort,
                    method=label,
                    prediction=prediction,
                    reference=reference,
                    input_rgb=input_rgb,
                    mask=mask,
                    policy=policy,
                )
            )

    needs_reference = any(plugin.spec.requires_gt and availability_by_metric[plugin.spec.name].available for plugin in metric_plugins)
    needs_input_and_mask = any(
        plugin.spec.requires_input_frame and availability_by_metric[plugin.spec.name].available for plugin in metric_plugins
    )
    for context in contexts:
        for plugin in metric_plugins:
            availability = availability_by_metric[plugin.spec.name]
            if not availability.available:
                continue
            for problem in required_input_problems(plugin, context):
                preflight_errors.append(f"{context.method}/{context.scene_id}/{plugin.spec.name}: {problem}")
        if needs_reference or needs_input_and_mask:
            preflight_errors.extend(
                f"{context.method}/{context.scene_id}: {problem}"
                for problem in validate_video_pair(context, needs_reference, needs_input_and_mask)
            )

    paired_config = None
    if args.paired_methods:
        baseline, treatment = args.paired_methods
        if baseline not in method_counts or treatment not in method_counts:
            preflight_errors.append(f"--paired-methods must name supplied data roots: {baseline}, {treatment}")
        else:
            baseline_keys = {(item.scene_id, item.split, item.cohort) for item in contexts if item.method == baseline}
            treatment_keys = {(item.scene_id, item.split, item.cohort) for item in contexts if item.method == treatment}
            if baseline_keys != treatment_keys:
                missing_treatment = sorted(baseline_keys - treatment_keys)
                missing_baseline = sorted(treatment_keys - baseline_keys)
                preflight_errors.append(
                    f"Paired methods do not contain identical scenes; missing treatment={missing_treatment[:10]}, "
                    f"missing baseline={missing_baseline[:10]}"
                )
            paired_config = {
                "baseline_method": baseline,
                "treatment_method": treatment,
                "tie_tolerance": args.tie_tolerance,
            }

    preflight = {
        "mode": "auto_discovery",
        "ok": not preflight_errors,
        "data_roots": {label: str(root) for label, root in labeled_roots},
        "method_counts": method_counts,
        "total_videos": sum(method_counts.values()),
        "metrics": [plugin.spec.name for plugin in metric_plugins],
        "reference_roots": [str(root) for root in reference_roots],
        "errors": preflight_errors,
    }
    write_json(output_dir / "preflight.json", preflight)
    if preflight_errors:
        print(f"ERROR: preflight failed; no metric computation started. errors={len(preflight_errors)}", file=sys.stderr)
        for error in preflight_errors[:20]:
            print(f"- {error}", file=sys.stderr)
        if len(preflight_errors) > 20:
            print(f"- ... {len(preflight_errors) - 20} more; see {output_dir / 'preflight.json'}", file=sys.stderr)
        return 2

    previous_records = read_jsonl(raw_path) if args.resume else []
    auto_config: dict[str, Any] = {
        "name": "auto_discovery_evaluation",
        "evaluation": policy,
        "paired_comparison": paired_config,
    }
    audit: dict[str, Any] = {
        "framework": "evaluation.run_eval",
        "mode": "auto_discovery",
        "git_commit": git_commit(),
        "config_sha256": sha256_json(auto_config),
        "resolved_config": auto_config,
        "manifest": None,
        "manifest_sha256": None,
        "evaluation_policy": policy,
        "selected_scenes": [
            {"scene_id": context.scene_id, "stem": context.stem, "split": context.split, "cohort": context.cohort}
            for context in contexts
        ],
        "selected_methods": list(method_counts),
        "selected_metrics": [plugin.spec.name for plugin in metric_plugins],
        "metric_plugins": {
            plugin.spec.name: {
                "spec": {
                    "name": plugin.spec.name,
                    "version": plugin.spec.version,
                    "higher_is_better": plugin.spec.higher_is_better,
                    "requires_GT": plugin.spec.requires_gt,
                    "requires_full_360_GT": plugin.spec.requires_full_360_gt,
                    "requires_mask": plugin.spec.requires_mask,
                    "requires_input_frame": plugin.spec.requires_input_frame,
                    "required_dependencies": plugin.spec.required_dependencies,
                    "expected_input": plugin.spec.expected_input,
                    "output_metrics": plugin.spec.output_metrics,
                },
                "availability": availability_by_metric[plugin.spec.name].__dict__,
            }
            for plugin in metric_plugins
        },
        "data_roots": {label: str(root) for label, root in labeled_roots},
        "method_counts": method_counts,
        "total_videos": sum(method_counts.values()),
        "dry_run": args.dry_run,
        "resumed_successful_plugins": [],
        "status_counts": {},
    }

    new_records: list[dict[str, Any]] = []
    label = f"Evaluation | videos={sum(method_counts.values())}; " + ", ".join(
        f"{method}={count}" for method, count in method_counts.items()
    )
    progress = EvaluationProgress(len(contexts) * len(metric_plugins), label=label)
    try:
        for context in contexts:
            for plugin in metric_plugins:
                progress.start(context, plugin)
                active_context = replace(context, progress=progress.update_current)
                availability = availability_by_metric[plugin.spec.name]
                if not availability.available:
                    new_records.append(base_record(plugin, context, plugin.spec.name, "unavailable", availability.reason))
                    progress.complete("unavailable")
                    continue
                if args.dry_run:
                    for metric in plugin.spec.output_metrics:
                        new_records.append(base_record(plugin, context, metric, "dry_run", "Preflight passed; computation not started."))
                    progress.complete("dry_run")
                    continue
                if args.resume and has_completed_plugin(previous_records, plugin, context):
                    audit["resumed_successful_plugins"].append(
                        {"scene_id": context.scene_id, "method": context.method, "plugin": plugin.spec.name}
                    )
                    progress.complete("resumed")
                    continue
                plugin_records = evaluate_plugin(plugin, active_context, availability)
                new_records.extend(plugin_records)
                progress.complete(plugin_records[0]["status"] if plugin_records else "failed")
    finally:
        progress.close()

    records = previous_records + new_records
    status_counts = Counter(record["status"] for record in records)
    audit["status_counts"] = dict(sorted(status_counts.items()))
    bootstrap_samples = args.bootstrap_samples or 0
    summaries = aggregate_records(records, bootstrap_samples=bootstrap_samples, seed=args.seed, include_combined=args.combined)
    paired_rows: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    if paired_config is not None:
        paired_rows = paired_delta_records(
            records,
            baseline_method=paired_config["baseline_method"],
            treatment_method=paired_config["treatment_method"],
            include_combined=args.combined,
            tie_tolerance=paired_config["tie_tolerance"],
        )
        paired_summaries = aggregate_paired_deltas(paired_rows, bootstrap_samples=bootstrap_samples, seed=args.seed)

    summary_rows = summaries + [
        {
            "method": row["comparison"],
            "split": row["split"],
            "metric": row["metric"],
            "valid_scenes": row["valid_scenes"],
            "mean": row["mean_delta"],
            "median": row["median_delta"],
            "std": row["std_delta"],
            "bootstrap_95ci_low": row.get("bootstrap_95ci_low"),
            "bootstrap_95ci_high": row.get("bootstrap_95ci_high"),
        }
        for row in paired_summaries
    ]
    write_jsonl(raw_path, records)
    write_csv(
        output_dir / PER_SCENE_NAME,
        records,
        ["scene_id", "stem", "split", "cohort", "method", "plugin", "metric", "higher_is_better", "status", "reason", "value", "special_value", "prediction", "reference", "input_rgb", "mask"],
    )
    write_csv(
        output_dir / SUMMARY_NAME,
        summary_rows,
        ["method", "split", "metric", "valid_scenes", "mean", "median", "std", "bootstrap_95ci_low", "bootstrap_95ci_high"],
    )
    write_csv(
        output_dir / PAIRED_NAME,
        paired_rows,
        ["scene_id", "split", "cohort", "metric", "baseline_method", "treatment_method", "baseline_value", "treatment_value", "delta", "higher_is_better", "improvement_direction", "outcome"],
    )
    write_json(output_dir / AUDIT_NAME, audit)
    (output_dir / MARKDOWN_NAME).write_text(markdown_summary(auto_config, summaries, paired_summaries, audit), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(f"videos={sum(method_counts.values())} methods={len(method_counts)} metrics={len(metric_plugins)} dry_run={args.dry_run}")
    print("status_counts=" + json.dumps(json_safe(audit["status_counts"]), sort_keys=True))
    return 0 if status_counts.get("failed", 0) == 0 else 1
