from pathlib import Path

import numpy as np

from evaluation.aggregate import aggregate_records
from evaluation.datasets.benchmark_manifest import load_manifest
from evaluation.metrics.panorama.seam import cyclic_seam_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark40_manifest_is_frozen_and_balanced() -> None:
    scenes = load_manifest(REPO_ROOT / "evaluation/manifests/benchmark40.csv")

    assert len(scenes) == 40
    assert sum(scene.split == "id" for scene in scenes) == 20
    assert sum(scene.split == "ood" for scene in scenes) == 20
    assert sum(scene.cohort == "different_scene" for scene in scenes) == 20
    assert sum(scene.cohort == "different_pose" for scene in scenes) == 20


def test_cyclic_seam_metrics_detects_no_jump_for_a_cyclic_frame() -> None:
    frame = np.full((4, 6, 3), 0.5, dtype=np.float32)

    values = cyclic_seam_metrics(frame)

    assert values["seam_l1"] == 0.0
    assert values["seam_percentile"] == 100.0
    assert values["edge_slope_mismatch"] == 0.0


def test_aggregate_keeps_id_and_ood_separate_unless_requested() -> None:
    records = [
        {"method": "stage2", "split": "id", "metric": "psnr", "status": "ok", "value": 20.0},
        {"method": "stage2", "split": "id", "metric": "psnr", "status": "ok", "value": 22.0},
        {"method": "stage2", "split": "ood", "metric": "psnr", "status": "ok", "value": 10.0},
    ]

    separate = aggregate_records(records)
    combined = aggregate_records(records, include_combined=True)

    assert {(row["split"], row["valid_scenes"]) for row in separate} == {("id", 2), ("ood", 1)}
    assert any(row["split"] == "combined" and row["valid_scenes"] == 3 for row in combined)
