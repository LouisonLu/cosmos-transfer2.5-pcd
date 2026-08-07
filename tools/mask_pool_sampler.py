#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_FAMILY_WEIGHTS = {
    "av2": 60.0,
    "native_fov": 10.0,
    "viewdrop": 10.0,
    "renderer_confidence": 10.0,
    "mixed": 10.0,
}

DEFAULT_AV2_BUCKET_WEIGHTS = {
    "mask_10_15_percent": 20.0,
    "mask_15_20_percent": 20.0,
    "mask_20_25_percent": 20.0,
}


def _normalize_weights(weights: dict[str, float], available_keys: set[str]) -> dict[str, float]:
    filtered = {k: float(v) for k, v in weights.items() if k in available_keys and float(v) > 0}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError(f"No positive weights remain. requested={weights}, available={sorted(available_keys)}")
    return {k: v / total for k, v in filtered.items()}


class MaskPoolSampler:
    """Weighted sampler over the DB235 mask-pool folder layout.

    Supports either:
    - pool_root=/.../mask_pool  with split="train" or "val"
    - pool_root=/.../mask_pool/train
    - pool_root=/.../mask_pool/val
    """

    def __init__(
        self,
        pool_root: str | Path,
        seed: int = 0,
        split: str = "train",
        family_weights: dict[str, float] | None = None,
        av2_bucket_weights: dict[str, float] | None = None,
    ) -> None:
        self.root = Path(pool_root)
        self.split = split
        self.rng = random.Random(seed)

        self.split_root = self._resolve_split_root(self.root, split=split)
        self.family_dirs = {
            p.name: p for p in self.split_root.iterdir() if p.is_dir()
        }
        if not self.family_dirs:
            raise FileNotFoundError(f"No family directories found under {self.split_root}")

        requested_family_weights = dict(family_weights or DEFAULT_FAMILY_WEIGHTS)
        self.family_weights = _normalize_weights(requested_family_weights, set(self.family_dirs.keys()))

        self.family_pngs: dict[str, list[Path]] = {}
        for family, family_dir in self.family_dirs.items():
            if family == "av2":
                continue
            pngs = sorted(family_dir.rglob("*.png"))
            if pngs:
                self.family_pngs[family] = pngs

        self.av2_bucket_pngs: dict[str, list[Path]] = {}
        if "av2" in self.family_dirs:
            av2_dir = self.family_dirs["av2"]
            for bucket_dir in sorted([p for p in av2_dir.iterdir() if p.is_dir()]):
                pngs = sorted(bucket_dir.rglob("*.png"))
                if pngs:
                    self.av2_bucket_pngs[bucket_dir.name] = pngs

        if "av2" in self.family_weights and not self.av2_bucket_pngs:
            raise ValueError(f"Family weight includes av2, but no AV2 bucket masks found under {self.family_dirs['av2']}")

        requested_av2_bucket_weights = dict(av2_bucket_weights or DEFAULT_AV2_BUCKET_WEIGHTS)
        if self.av2_bucket_pngs:
            self.av2_bucket_weights = _normalize_weights(requested_av2_bucket_weights, set(self.av2_bucket_pngs.keys()))
        else:
            self.av2_bucket_weights = {}

        self._validate_non_av2_families()

    def _resolve_split_root(self, root: Path, split: str) -> Path:
        if (root / "av2").is_dir() or (root / "mixed").is_dir():
            return root
        candidate = root / split
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"Could not resolve split root from pool_root={root} and split={split}. "
            f"Expected either family dirs directly under root or a split dir like root/{split}."
        )

    def _validate_non_av2_families(self) -> None:
        missing = [
            family
            for family in self.family_weights
            if family != "av2" and not self.family_pngs.get(family)
        ]
        if missing:
            raise ValueError(
                f"Requested family weights include families with no PNG files: {missing}. "
                f"Available non-empty families: {sorted(k for k, v in self.family_pngs.items() if v)}"
            )

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        keys = list(weights.keys())
        vals = [weights[k] for k in keys]
        return self.rng.choices(keys, weights=vals, k=1)[0]

    def sample_path(self) -> tuple[Path, dict[str, Any]]:
        family = self._weighted_choice(self.family_weights)

        if family == "av2":
            bucket = self._weighted_choice(self.av2_bucket_weights)
            path = self.rng.choice(self.av2_bucket_pngs[bucket])
            meta = {
                "family": family,
                "bucket": bucket,
                "path": str(path),
                "split": self.split_root.name,
                "sampling": "weighted_family_then_bucket_uniform_file",
            }
            return path, meta

        path = self.rng.choice(self.family_pngs[family])
        meta = {
            "family": family,
            "bucket": None,
            "path": str(path),
            "split": self.split_root.name,
            "sampling": "weighted_family_uniform_file",
        }
        return path, meta

    def sample(self) -> tuple[np.ndarray, dict[str, Any]]:
        path, meta = self.sample_path()
        with Image.open(path) as im:
            mask = np.asarray(im.convert("L"), dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if mask.shape != (512, 1024) or not values.issubset({0, 255}):
            raise ValueError(f"Invalid mask contract: {path}, shape={mask.shape}, values={sorted(values)}")
        return mask, meta
