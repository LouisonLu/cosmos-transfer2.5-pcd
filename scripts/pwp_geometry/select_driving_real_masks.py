# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick the real driving camera masks that differ the most from each other, for the driving_real mask family.

Reads every PNG under <old pool>/train/<family>/ for the camera-view families (AV2, Waymo, nuScenes, PandaSet
camera footprints; renderer_confidence is left out: it is sparse PCD-confidence speckle, which every IoU calls
"most different", and the near_black family already covers it), splits the files
into train / val by a hash of the file name (1 in 10 goes to val, so the splits never share a mask), and runs a
greedy farthest-point selection with distance 1 - soft IoU (sum of minima / sum of maxima) of the visible
fraction on a 64x32 grid, which compares layouts and ignores pixel speckle. Copies the picks to
<out>/<split>/driving_real/ as <family>__<bucket>__<original name> and writes <out>/driving_real_manifest.csv.

Usage:
    python scripts/pwp_geometry/select_driving_real_masks.py --old-pool OLD_MASK_POOL --out NEW_MASK_POOL \
        --train-count 200 --val-count 50
"""

import argparse
import csv
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def small_visible(path: Path) -> np.ndarray:
    visible = np.asarray(Image.open(path).convert("L")) >= 128
    return visible.reshape(32, 16, 64, 16).mean(axis=(1, 3)).astype(np.float32)


def soft_iou(flat: np.ndarray, one: np.ndarray) -> np.ndarray:
    return np.minimum(flat, one).sum(1) / np.maximum(np.maximum(flat, one).sum(1), 1e-6)


def farthest_point(small: np.ndarray, count: int, seed: int) -> list[int]:
    flat = small.reshape(len(small), -1)
    # Start from the mask least like the average layout, then keep adding the mask least like any picked one.
    start = soft_iou(flat, flat.mean(0)) + np.random.default_rng(seed).uniform(0, 1e-6, len(flat))
    chosen = [int(np.argmin(start))]
    distance = np.full(len(flat), np.inf)
    for _ in range(min(count, len(flat)) - 1):
        iou = soft_iou(flat, flat[chosen[-1]])
        distance = np.minimum(distance, 1 - iou)
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-pool", type=Path, required=True, help="Old mask_pool root (contains train/av2 ...)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", nargs="+", default=["av2", "viewdrop", "mixed", "native_fov"])
    args = parser.parse_args()

    paths = sorted(
        p for family in args.families for p in (args.old_pool / "train" / family).rglob("*.png")
        if not p.name.startswith("._")
    )
    with ProcessPoolExecutor() as pool:
        small = np.stack(list(pool.map(small_visible, paths, chunksize=64)))
    is_val = np.array([zlib.crc32(p.name.encode()) % 10 == 0 for p in paths])

    rows = []
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        candidates = np.where(is_val == (split == "val"))[0]
        picked = candidates[farthest_point(small[candidates], count, args.seed)]
        folder = args.out / split / "driving_real"
        folder.mkdir(parents=True, exist_ok=False)
        for rank, index in enumerate(picked):
            source = paths[index]
            family, bucket = source.relative_to(args.old_pool / "train").parts[:2]
            name = f"{family}__{bucket}__{source.name}"
            shutil.copy2(source, folder / name)
            visible = (np.asarray(Image.open(source).convert("L")) >= 128).mean()
            rows.append({"split": split, "pick_order": rank, "file": f"{split}/driving_real/{name}",
                         "source": str(source.relative_to(args.old_pool)), "visible_fraction": f"{visible:.4f}"})
        print(f"{split}: picked {len(picked)} of {len(candidates)} candidates")

    with open(args.out / "driving_real_manifest.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
