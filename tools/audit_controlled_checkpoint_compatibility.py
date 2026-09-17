#!/usr/bin/env python3
"""Forensically validate local Cosmos checkpoints against one inference model.

This tool intentionally compares every tensor before calling
``load_state_dict(strict=False)``. It never filters, renames, reshapes, or
otherwise adapts a checkpoint. A load exception is a failed compatibility
result, not a reason to coerce the checkpoint into loading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    output_name: str
    output_dir: Path


GROUPS: dict[str, tuple[str, ...]] = {
    "base_dit_transformer": ("net.", "transformer", "blocks", "layers"),
    "depth_controlnet": ("control", "vace", "hint"),
    "image_reference_context": ("image_context", "img_context", "reference", "ref_image"),
    "vae_related": ("vae", "tokenizer", "autoencoder"),
    "text_context_projection": ("text_encoder", "t5", "text_proj", "context_proj"),
    "partial_hardlock": ("partial_hardlock",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io

    payload = easy_io.load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint is not a state-dict mapping: {type(payload)!r}")
    invalid = [key for key, value in payload.items() if not isinstance(key, str) or not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(f"Checkpoint is not a direct tensor state dict; invalid entries: {invalid[:8]}")
    return payload


def instantiate_inference_model(config_file: str, experiment: str):
    """Compose exactly the entry point's model config without loading weights."""
    from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

    # These are the same config overrides appended by the partial-hardlock pipeline.
    return load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir="",
        config_file=config_file,
        load_ema_to_reg=True,
        cache_text_encoder=False,
        skip_load_model=True,
        experiment_opts=["++model.config.base_load_from=null", "~data_train"],
    )


def tensor_summary(keys: set[str], model_state: dict[str, torch.Tensor], checkpoint: dict[str, torch.Tensor]) -> dict[str, Any]:
    same_shape = {key for key in keys if tuple(model_state[key].shape) == tuple(checkpoint[key].shape)}
    dtype_mismatches = sorted(
        key for key in same_shape if model_state[key].dtype != checkpoint[key].dtype
    )
    shape_mismatches = sorted(key for key in keys if key not in same_shape)
    return {
        "matched_key_count": len(keys),
        "shape_compatible_key_count": len(same_shape),
        "shape_mismatches": [
            {
                "key": key,
                "model_shape": list(model_state[key].shape),
                "checkpoint_shape": list(checkpoint[key].shape),
            }
            for key in shape_mismatches
        ],
        "dtype_mismatches": [
            {
                "key": key,
                "model_dtype": str(model_state[key].dtype),
                "checkpoint_dtype": str(checkpoint[key].dtype),
            }
            for key in dtype_mismatches
        ],
    }


def group_matches(group: str, key: str) -> bool:
    lowered = key.lower()
    terms = GROUPS[group]
    if group == "base_dit_transformer":
        return lowered.startswith("net.") and not any(
            term in lowered for term in GROUPS["depth_controlnet"] + GROUPS["image_reference_context"]
        )
    return any(term in lowered for term in terms)


def group_report(model_state: dict[str, torch.Tensor], checkpoint: dict[str, torch.Tensor]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for group in GROUPS:
        model_keys = {key for key in model_state if group_matches(group, key)}
        checkpoint_keys = {key for key in checkpoint if group_matches(group, key)}
        shared = model_keys & checkpoint_keys
        compatible = {
            key for key in shared if tuple(model_state[key].shape) == tuple(checkpoint[key].shape)
        }
        report[group] = {
            "model_key_count": len(model_keys),
            "checkpoint_key_count": len(checkpoint_keys),
            "shape_compatible_key_count": len(compatible),
            "missing_model_keys": sorted(model_keys - checkpoint_keys),
            "unexpected_checkpoint_keys": sorted(checkpoint_keys - model_keys),
            "shape_mismatch_keys": sorted(shared - compatible),
        }
    return report


def audit_checkpoint(
    spec: CheckpointSpec,
    config_file: str,
    experiment: str,
    request: dict[str, Any],
    resolved_root: Path,
) -> dict[str, Any]:
    model, config = instantiate_inference_model(config_file, experiment)
    from cosmos_transfer2._src.imaginaire.lazy_config import LazyConfig

    resolved_dir = resolved_root / spec.label
    resolved_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolved_dir / "resolved_inference_config.yaml"
    request_path = resolved_dir / "resolved_request.json"
    run_path = resolved_dir / "resolved_run.json"
    LazyConfig.save_yaml(config, config_path)
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    run_metadata = {
        "checkpoint_path": str(spec.path.resolve()),
        "output_name": spec.output_name,
        "output_directory": str(spec.output_dir.resolve()),
        "config_file": config_file,
        "experiment": experiment,
    }
    run_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    model_state = model.state_dict()
    checkpoint = checkpoint_state(spec.path)
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint)
    shared = model_keys & checkpoint_keys
    summary = tensor_summary(shared, model_state, checkpoint)
    shape_compatible = {
        key for key in shared if tuple(model_state[key].shape) == tuple(checkpoint[key].shape)
    }
    parameter_keys = set(dict(model.named_parameters()))
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    matched_parameter_count = sum(model_state[key].numel() for key in shape_compatible & parameter_keys)
    coverage = 0.0 if total_parameter_count == 0 else 100.0 * matched_parameter_count / total_parameter_count

    raw_load: dict[str, Any]
    try:
        load_info = model.load_state_dict(checkpoint, strict=False)
        raw_load = {
            "returned": True,
            "missing_keys": list(load_info.missing_keys),
            "unexpected_keys": list(load_info.unexpected_keys),
            "error": None,
        }
    except RuntimeError as error:
        raw_load = {"returned": False, "missing_keys": [], "unexpected_keys": [], "error": str(error)}

    partial_keys = [key for key in model_state if "partial_hardlock" in key.lower()]
    structural_failure = bool(summary["shape_mismatches"] or raw_load["error"])
    learned_missing = sorted(set(raw_load["missing_keys"]) & parameter_keys)
    learned_unexpected = sorted(set(raw_load["unexpected_keys"]) & checkpoint_keys)
    if not structural_failure and not learned_missing and not learned_unexpected and coverage == 100.0:
        if raw_load["missing_keys"] or raw_load["unexpected_keys"]:
            status = "PASS_WITH_EXPECTED_NONTRAINABLE_DIFFERENCES"
        else:
            status = "PASS"
    else:
        status = "FAIL"

    return {
        "label": spec.label,
        "checkpoint_path": str(spec.path.resolve()),
        "checkpoint_bytes": spec.path.stat().st_size,
        "checkpoint_sha256": sha256(spec.path),
        "model_class": type(model).__name__,
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "matched_key_count": len(shared),
        "missing_keys_before_load": sorted(model_keys - checkpoint_keys),
        "unexpected_keys_before_load": sorted(checkpoint_keys - model_keys),
        "shape_mismatches": summary["shape_mismatches"],
        "dtype_mismatches": summary["dtype_mismatches"],
        "matched_parameter_count": matched_parameter_count,
        "total_model_parameter_count": total_parameter_count,
        "parameter_coverage_percent": coverage,
        "load_state_dict_strict_false": raw_load,
        "learned_missing_keys": learned_missing,
        "learned_unexpected_keys": learned_unexpected,
        "subsystems": group_report(model_state, checkpoint),
        "partial_hardlock_trainable_keys": partial_keys,
        "partial_hardlock_trainable_key_count": len(partial_keys),
        "resolved_request": request,
        "resolved_run": run_metadata,
        "resolved_artifacts": {
            "config": str(config_path.resolve()),
            "request": str(request_path.resolve()),
            "run": str(run_path.resolve()),
        },
        "status": status,
        "resolved_config_repr": repr(config),
    }


def request_diff(reports: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = reports[0]
    allowed = {"checkpoint_path", "output_name", "output_directory"}
    differences: list[dict[str, Any]] = []
    for report in reports[1:]:
        for section in ("resolved_request", "resolved_run"):
            left = baseline[section]
            right = report[section]
            for key in sorted(set(left) | set(right)):
                if left.get(key) != right.get(key):
                    differences.append(
                        {
                            "baseline": baseline["label"],
                            "comparison": report["label"],
                            "field": f"{section}.{key}",
                            "baseline_value": left.get(key),
                            "comparison_value": right.get(key),
                            "allowed": key in allowed,
                        }
                    )
    return {"allowed_difference_fields": sorted(allowed), "differences": differences}


def parse_checkpoint(value: str) -> CheckpointSpec:
    label, path, output_name, output_dir = value.split("|", 3)
    checkpoint = Path(path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Missing or empty checkpoint for {label}: {checkpoint}")
    return CheckpointSpec(label, checkpoint, output_name, Path(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="label|path|output_name|output_directory")
    parser.add_argument("--request", type=Path, required=True, help="One exact shared no-video-path request JSON")
    parser.add_argument("--config-file", default="cosmos_transfer2/singleview_partial_hardlock_config.py")
    parser.add_argument(
        "--experiment", default="transfer2_singleview_partial_hardlock_pcd_rgb_image_context_example"
    )
    parser.add_argument("--audit-root", type=Path, default=Path("audits"))
    args = parser.parse_args()

    if len(args.checkpoint) != 3:
        parser.error("Provide exactly three --checkpoint specifications.")
    try:
        specs = [parse_checkpoint(value) for value in args.checkpoint]
        request = json.loads(args.request.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(request, dict):
        print("ERROR: --request must contain one JSON object.", file=sys.stderr)
        raise SystemExit(2)

    args.audit_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for spec in specs:
        print(f"Auditing {spec.label}: {spec.path}", flush=True)
        reports.append(audit_checkpoint(spec, args.config_file, args.experiment, request, args.audit_root / "runtime_resolved"))

    payload = {
        "audit_type": "runtime_state_dict_preflight",
        "entry_point": "examples/inference_partial_hardlock_no_video_path.py",
        "config_file": args.config_file,
        "experiment": args.experiment,
        "reports": reports,
    }
    diff = request_diff(reports)
    (args.audit_root / "controlled_checkpoint_compatibility_runtime_2026-09-17.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.audit_root / "runtime_request_diff.json").write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    with (args.audit_root / "runtime_checkpoint_key_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label", "status", "model_key_count", "checkpoint_key_count", "matched_key_count",
                "shape_mismatch_count", "dtype_mismatch_count", "parameter_coverage_percent", "checkpoint_bytes",
                "checkpoint_sha256",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "label": report["label"], "status": report["status"],
                    "model_key_count": report["model_key_count"], "checkpoint_key_count": report["checkpoint_key_count"],
                    "matched_key_count": report["matched_key_count"],
                    "shape_mismatch_count": len(report["shape_mismatches"]),
                    "dtype_mismatch_count": len(report["dtype_mismatches"]),
                    "parameter_coverage_percent": report["parameter_coverage_percent"],
                    "checkpoint_bytes": report["checkpoint_bytes"], "checkpoint_sha256": report["checkpoint_sha256"],
                }
            )
    lines = ["# Runtime Checkpoint Compatibility Audit", "", f"Entry: `{payload['entry_point']}`", ""]
    for report in reports:
        lines.extend([
            f"## {report['label']}", "",
            f"- Status: `{report['status']}`", f"- Checkpoint: `{report['checkpoint_path']}`",
            f"- SHA256: `{report['checkpoint_sha256']}`", f"- Model class: `{report['model_class']}`",
            f"- Keys: model={report['model_key_count']}, checkpoint={report['checkpoint_key_count']}, matched={report['matched_key_count']}",
            f"- Parameter coverage: {report['parameter_coverage_percent']:.6f}%", f"- Shape mismatches: {len(report['shape_mismatches'])}",
            f"- Dtype mismatches: {len(report['dtype_mismatches'])}",
            f"- strict=False returned: {report['load_state_dict_strict_false']['returned']}",
            f"- Missing keys returned: {len(report['load_state_dict_strict_false']['missing_keys'])}",
            f"- Unexpected keys returned: {len(report['load_state_dict_strict_false']['unexpected_keys'])}",
            f"- Partial-hardlock trainable keys: {report['partial_hardlock_trainable_key_count']}", "",
        ])
        if report["load_state_dict_strict_false"]["error"]:
            lines.extend(["```text", report["load_state_dict_strict_false"]["error"], "```", ""])
    (args.audit_root / "controlled_checkpoint_compatibility_runtime_2026-09-17.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Runtime compatibility artifacts written to", args.audit_root, flush=True)
    if any(report["status"] == "FAIL" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
