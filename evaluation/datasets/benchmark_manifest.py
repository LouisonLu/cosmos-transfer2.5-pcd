"""Frozen 40-scene benchmark-manifest loading and validation."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = ("scene_id", "stem", "split", "dataset", "cohort")


@dataclass(frozen=True)
class BenchmarkScene:
    scene_id: str
    stem: str
    split: str
    dataset: str
    cohort: str


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[BenchmarkScene]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"Manifest columns must be exactly {REQUIRED_COLUMNS}: {path}")
        scenes = [BenchmarkScene(**{key: row[key].strip() for key in REQUIRED_COLUMNS}) for row in reader]
    if not scenes:
        raise ValueError(f"Benchmark manifest is empty: {path}")
    scene_ids = [scene.scene_id for scene in scenes]
    stems = [scene.stem for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)) or len(stems) != len(set(stems)):
        raise ValueError(f"Benchmark manifest has duplicate scene IDs or stems: {path}")
    invalid_splits = sorted({scene.split for scene in scenes} - {"id", "ood"})
    invalid_cohorts = sorted({scene.cohort for scene in scenes} - {"different_scene", "different_pose"})
    if invalid_splits or invalid_cohorts:
        raise ValueError(f"Invalid manifest values, splits={invalid_splits}, cohorts={invalid_cohorts}")
    return scenes
