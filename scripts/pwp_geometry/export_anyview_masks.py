# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export any-view masks to PNG: the exact masks of a training run, or the fixed val set.

training: the mask of every (epoch, video) that PwP Stage 1 will draw, with the same sampler, config and seed.
    Videos are numbered like the dataset (sorted <dataset-dir>/rgb_videos/*.mp4), or from --video-list.
    Writes <out>/epoch_<e>/<index>__<video>__<family>.png, <out>/masks.csv and an overview of the first 500.
val: --count masks of the val split (val driving_real files, another seed), to evaluate unseen views.
    Writes <out>/<family>/val_<i>.png and <out>/masks.csv.

Usage:
    python scripts/pwp_geometry/export_anyview_masks.py training --mask-pool-root POOL --dataset-dir DATA \
        --epochs 2 --out OUT
    python scripts/pwp_geometry/export_anyview_masks.py val --mask-pool-root POOL --count 500 --seed 1 --out OUT
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SAMPLER_PY = REPO / "cosmos_transfer2/_src/transfer2/pwp_geometry/anyview_masks.py"
DEFAULT_CONFIG = REPO / "scripts/pwp_geometry/anyview_mask_sampler_config.json"


def load_sampler_module():
    # Loaded by path, as the dataset does with mask_pool_sampler_py.
    spec = importlib.util.spec_from_file_location("pwp_anyview_masks", SAMPLER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_overview(paths: list[Path], out: Path) -> None:
    tiles = [np.pad(np.asarray(Image.open(p).convert("L").resize((128, 64))), 1, constant_values=110) for p in paths]
    tiles += [np.full_like(tiles[0], 40)] * (-len(tiles) % 20)
    Image.fromarray(np.vstack([np.hstack(tiles[i : i + 20]) for i in range(0, len(tiles), 20)])).save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["training", "val"])
    parser.add_argument("--mask-pool-root", type=Path, required=True, help="Has <split>/driving_real/*.png")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="mask_pool_sampler_kwargs_json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="training: mask_pool_sampler_seed; val: val seed")
    parser.add_argument("--dataset-dir", type=Path, help="training: the Stage 1 dataset_dir")
    parser.add_argument("--video-list", type=Path, help="training: file of rgb_videos names instead of a dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--count", type=int, default=500, help="val: number of masks")
    args = parser.parse_args()

    module = load_sampler_module()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    rows, written = [], []

    if args.mode == "training":
        if args.dataset_dir:
            names = [p.name for p in (args.dataset_dir / "rgb_videos").glob("*.mp4")]
        elif args.video_list:
            names = [line.strip() for line in args.video_list.read_text().splitlines() if line.strip()]
        else:
            parser.error("training needs --dataset-dir or --video-list")
        names = sorted(names)  # same order as the dataset's sorted video_paths
        sampler = module.MaskPoolSampler(args.mask_pool_root, **config)
        for epoch in range(args.epochs):
            (args.out / f"epoch_{epoch}").mkdir()
            for index, name in enumerate(names):
                mask, meta = module.training_mask(sampler, args.seed, epoch, index)
                path = args.out / f"epoch_{epoch}" / f"{index:05d}__{Path(name).stem}__{meta['family']}.png"
                Image.fromarray(mask).save(path)
                written.append(path)
                rows.append({"epoch": epoch, "video_index": index, "video": name, **meta,
                             "visible_fraction": f"{(mask > 0).mean():.4f}", "png": str(path.relative_to(args.out))})
    else:
        sampler = module.MaskPoolSampler(args.mask_pool_root, **{**config, "split": "val"})
        for index in range(args.count):
            mask, meta = module.training_mask(sampler, args.seed, 0, index)
            folder = args.out / meta["family"]
            folder.mkdir(exist_ok=True)
            path = folder / f"val_{index:04d}.png"
            Image.fromarray(mask).save(path)
            written.append(path)
            rows.append({"val_index": index, **meta, "visible_fraction": f"{(mask > 0).mean():.4f}",
                         "png": str(path.relative_to(args.out))})

    with open(args.out / "masks.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_overview(written[:500], args.out / "overview_first_500.png")
    families = {}
    for row in rows:
        families[row["family"]] = families.get(row["family"], 0) + 1
    print(f"wrote {len(rows)} masks to {args.out}")
    print({family: f"{count / len(rows) * 100:.1f}%" for family, count in sorted(families.items())})


if __name__ == "__main__":
    main()
