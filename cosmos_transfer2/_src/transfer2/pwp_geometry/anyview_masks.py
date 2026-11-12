# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Any-view masks for PwP Stage 1, drawn while training: each mask is what a set of cameras sees on the 360° sphere.

A mask is 512x1024 ERP, 255 = visible (kept in the first frame and the PCD), 0 = hidden. The family is drawn by
weight; driving_real reads the real camera masks under <pool_root>/<split>/driving_real/, every other family is
generated from camera parameters. Camera families can additionally lose a few objects inside the view.

Use it as the dataset's ``mask_pool_sampler_py`` with ``scripts/pwp_geometry/anyview_mask_sampler_config.json``.
With ``PwPSingleViewTransferDatasetMask`` the mask of a sample depends only on (mask_pool_sampler_seed, epoch,
video index), so ``training_mask`` reproduces the exact masks of a run (scripts/pwp_geometry/export_anyview_masks.py).

The file only imports numpy and PIL because the dataset loads it by path.
ERP convention: image center = yaw 0 (front), yaw grows to the right, top row = up.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

H, W = 512, 1024

FAMILY_DESCRIPTIONS = {
    "driving_real": "real AV2 / Waymo / nuScenes / PandaSet camera masks (files, picked to differ in layout)",
    "driving_front": "one forward car camera, as in the single front-view benchmarks",
    "driving_multi": "car rigs of 3-7 cameras with 0-3 cameras dropped",
    "single_camera": "one rectangular camera anywhere: any yaw, tilted, rolled, any FOV and aspect",
    "fisheye": "one 150-220 degree fisheye (circular footprint), sometimes straight up or down",
    "multi_camera": "2-6 cameras in any directions: robot head + chest, drone looking down, stereo pairs",
    "partial_pano": "a panorama sweep: a 90-270 degree wide band at some latitude",
    "pano_holes": "a full 360 with holes: nadir / zenith caps, dual-fisheye stitching gaps at +-90, occluders",
    "near_black": "almost nothing visible: a tiny camera or a few specks",
    "no_mask": "everything visible (full first frame, as in the driving50 evaluation)",
}

LANDSCAPE = (16 / 9, 3 / 2, 4 / 3, 1.6)
ANY_ASPECT = (1.0, 4 / 3, 3 / 2, 16 / 9, 3 / 4, 9 / 16)


def mask_seed(base_seed: int, epoch: int, index: int) -> int:
    """Seed of the mask of one (epoch, video index); used by PwPSingleViewTransferDatasetMask."""
    return int(np.random.SeedSequence([int(base_seed), int(epoch), int(index)]).generate_state(1)[0])


def erp_directions() -> np.ndarray:
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)


DIRS = erp_directions()
LON_DEG = (np.arange(W) + 0.5) / W * 360 - 180
LAT_DEG = 90 - (np.arange(H) + 0.5) / H * 180


def direction(yaw: float, pitch: float) -> np.ndarray:
    y, p = np.radians(yaw), np.radians(pitch)
    return np.array([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)])


def camera(yaw: float, pitch: float, roll: float, fov: float, aspect: float) -> np.ndarray:
    """Pixels inside a pinhole camera's rectangular image; fov spans the long side, aspect = width / height."""
    y, r = np.radians(yaw), np.radians(roll)
    forward = direction(yaw, pitch)
    right = np.array([np.cos(y), 0.0, -np.sin(y)])
    up = np.cross(forward, right)
    right, up = np.cos(r) * right + np.sin(r) * up, -np.sin(r) * right + np.cos(r) * up
    z = DIRS @ forward
    tan_long = np.tan(np.radians(fov) / 2)
    tan_h, tan_v = (tan_long, tan_long / aspect) if aspect >= 1 else (tan_long * aspect, tan_long)
    with np.errstate(divide="ignore", invalid="ignore"):
        x, v = (DIRS @ right) / z, (DIRS @ up) / z
    return (z > 1e-6) & (np.abs(x) <= tan_h) & (np.abs(v) <= tan_v)


def cap(yaw: float, pitch: float, radius: float) -> np.ndarray:
    """Pixels within an angular radius of a direction (fisheye footprint, blobs, holes)."""
    return DIRS @ direction(yaw, pitch) >= np.cos(np.radians(radius))


def longitude_strip(center: float, width: float) -> np.ndarray:
    return np.repeat((np.abs((LON_DEG - center + 180) % 360 - 180) <= width / 2)[None, :], H, axis=0)


def driving_front(rng) -> np.ndarray:
    # AV2's front-center camera is portrait; most other front cameras are landscape.
    aspect = rng.choice((3 / 4, 9 / 16)) if rng.random() < 0.3 else rng.choice(LANDSCAPE)
    return camera(rng.uniform(-10, 10), rng.uniform(-5, 5), rng.uniform(-2, 2), rng.uniform(50, 100), aspect)


DRIVING_RIGS = {
    "front3": [0, -55, 55],
    "surround5": [0, -60, 60, -125, 125],
    "nuscenes6": [0, -55, 55, -110, 110, 180],
    "av2_7": [0, -45, 45, -95, 95, -150, 150],
}


def driving_multi(rng) -> np.ndarray:
    yaws = DRIVING_RIGS[rng.choice(list(DRIVING_RIGS))]
    drop = rng.integers(0, min(3, len(yaws) - 2) + 1)
    keep = rng.permutation(len(yaws))[: len(yaws) - drop]
    fov = rng.uniform(50, 100)
    mask = np.zeros((H, W), bool)
    for i in keep:
        aspect = 9 / 16 if (yaws[i] == 0 and rng.random() < 0.3) else rng.choice(LANDSCAPE)
        mask |= camera(yaws[i] + rng.normal(0, 4), rng.uniform(-5, 5), rng.normal(0, 1), fov + rng.normal(0, 5), aspect)
    return mask


def single_camera(rng, fov_range=(30, 120)) -> np.ndarray:
    pitch = float(np.clip(rng.normal(0, 25), -75, 75))
    roll = float(np.clip(rng.normal(0, 7), -15, 15))
    return camera(rng.uniform(-180, 180), pitch, roll, rng.uniform(*fov_range), rng.choice(ANY_ASPECT))


def fisheye(rng) -> np.ndarray:
    if rng.random() < 0.2:
        pitch = rng.choice([-90, 90]) * rng.uniform(0.7, 1)
    else:
        pitch = float(np.clip(rng.normal(0, 25), -60, 60))
    return cap(rng.uniform(-180, 180), pitch, rng.uniform(150, 220) / 2)


def multi_camera(rng) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    kind = rng.choice(["random", "drone", "stereo", "robot"])
    if kind == "drone":  # one camera looking down, plus 0-2 others
        mask |= camera(rng.uniform(-180, 180), rng.uniform(-90, -60), rng.uniform(-180, 180), rng.uniform(60, 110),
                       rng.choice(ANY_ASPECT))
        for _ in range(rng.integers(0, 3)):
            mask |= single_camera(rng, (50, 100))
    elif kind == "stereo":  # 1-3 pairs looking almost the same way
        for _ in range(rng.integers(1, 4)):
            yaw, pitch, fov, aspect = rng.uniform(-180, 180), rng.normal(0, 15), rng.uniform(50, 100), rng.choice(LANDSCAPE)
            for offset in (-rng.uniform(2, 8), rng.uniform(2, 8)):
                mask |= camera(yaw + offset, pitch, 0, fov, aspect)
    elif kind == "robot":  # head camera plus a lower (chest / wrist) camera
        yaw = rng.uniform(-180, 180)
        mask |= camera(yaw, rng.uniform(-30, 10), rng.normal(0, 5), rng.uniform(60, 100), rng.choice(LANDSCAPE))
        mask |= camera(yaw + rng.normal(0, 30), rng.uniform(-75, -35), rng.normal(0, 10), rng.uniform(60, 110),
                       rng.choice(ANY_ASPECT))
    else:
        for _ in range(rng.integers(2, 7)):
            mask |= single_camera(rng, (40, 120))
    return mask


def partial_pano(rng) -> np.ndarray:
    center, width = rng.uniform(-180, 180), rng.uniform(90, 270)
    lat_center, half_height = float(np.clip(rng.normal(0, 12), -30, 30)), rng.uniform(20, 45)
    return longitude_strip(center, width) & (np.abs(LAT_DEG - lat_center) <= half_height)[:, None]


def pano_holes(rng) -> np.ndarray:
    hidden = np.zeros((H, W), bool)
    if rng.random() < 0.7:  # tripod / nadir
        hidden |= cap(0, -90, rng.uniform(10, 40))
    if rng.random() < 0.3:
        hidden |= cap(0, 90, rng.uniform(10, 30))
    for seam in (-90, 90):  # dual-fisheye stitching gaps sit where the two lenses meet, at +-90 degrees
        if rng.random() < 0.4:
            hidden |= longitude_strip(seam + rng.normal(0, 3), rng.uniform(3, 12))
    for _ in range(rng.integers(0, 5)):  # occluders
        hidden |= cap(rng.uniform(-180, 180), rng.uniform(-60, 60), rng.uniform(8, 35))
    if not hidden.any():
        hidden |= cap(0, -90, 25)
    return ~hidden


def near_black(rng) -> np.ndarray:
    if rng.random() < 0.6:
        return single_camera(rng, (10, 25))
    mask = np.zeros((H, W), bool)
    for _ in range(rng.integers(10, 60)):
        mask |= cap(rng.uniform(-180, 180), rng.normal(0, 15), rng.uniform(0.5, 3))
    return mask


def no_mask(rng) -> np.ndarray:
    return np.ones((H, W), bool)


GENERATORS = {
    "driving_front": driving_front,
    "driving_multi": driving_multi,
    "single_camera": single_camera,
    "fisheye": fisheye,
    "multi_camera": multi_camera,
    "partial_pano": partial_pano,
    "pano_holes": pano_holes,
    "near_black": near_black,
    "no_mask": no_mask,
}


def occlude_objects(visible: np.ndarray, rng, fraction_range) -> np.ndarray:
    """Hide 1-6 object-like shapes (ellipses, irregular polygons, vehicle boxes) inside the visible area.

    They cover about ``fraction_range`` of the visible pixels, mostly in its lower half (things on the ground),
    and wrap around the left / right edge like the sphere does.
    """
    rows, cols = np.nonzero(visible)
    if len(rows) < 200:
        return visible
    count = int(rng.integers(1, 7))
    object_area = rng.uniform(*fraction_range) * len(rows) / count
    lower = np.nonzero(rows >= np.median(rows))[0]
    canvas = Image.new("L", (3 * W, H), 0)  # centre copy + one copy on each side, folded back below
    draw = ImageDraw.Draw(canvas)
    for _ in range(count):
        pick = lower[rng.integers(len(lower))] if rng.random() < 0.7 else rng.integers(len(rows))
        cy, cx = float(rows[pick]), float(cols[pick]) + W
        area = object_area * rng.uniform(0.5, 1.5)
        shape = rng.choice(["ellipse", "polygon", "box"])
        if shape == "box":  # vehicles: wider than tall
            aspect = rng.uniform(1.2, 3.0)
            half_w, half_h = np.sqrt(area * aspect) / 2, np.sqrt(area / aspect) / 2
            draw.rectangle([cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=255)
        elif shape == "ellipse":
            aspect = rng.uniform(0.5, 2.0)
            rx, ry = np.sqrt(area * aspect / np.pi), np.sqrt(area / aspect / np.pi)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        else:
            radius = np.sqrt(area / np.pi)
            angles = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(5, 11)))
            radii = radius * rng.uniform(0.6, 1.4, len(angles))
            draw.polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(angles, radii)], fill=255)
    painted = np.asarray(canvas) > 0
    hidden = painted[:, :W] | painted[:, W : 2 * W] | painted[:, 2 * W :]
    return visible & ~hidden


class MaskPoolSampler:
    """Same interface as tools/mask_pool_sampler.py: ``rng`` (random.Random) and ``sample() -> (mask, meta)``."""

    def __init__(
        self,
        pool_root,
        seed: int = 0,
        split: str = "train",
        family_weights: dict | None = None,
        occlusion_probability: float = 0.0,
        occlusion_families: list | None = None,
        occlusion_fraction: list | None = None,
    ) -> None:
        if not family_weights:
            raise ValueError("family_weights is required (see scripts/pwp_geometry/anyview_mask_sampler_config.json)")
        unknown = set(family_weights) - set(FAMILY_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"Unknown mask families {sorted(unknown)}; known: {sorted(FAMILY_DESCRIPTIONS)}")
        self.rng = random.Random(seed)
        self.split = split
        self.families = [family for family, weight in family_weights.items() if float(weight) > 0]
        self.weights = [float(family_weights[family]) for family in self.families]
        self.occlusion_probability = float(occlusion_probability)
        self.occlusion_families = set(occlusion_families or [])
        self.occlusion_fraction = tuple(occlusion_fraction or (0.05, 0.3))
        self.real_files: list[Path] = []
        if "driving_real" in self.families:
            folder = Path(pool_root) / split / "driving_real"
            self.real_files = sorted(folder.glob("*.png"))
            if not self.real_files:
                raise FileNotFoundError(f"driving_real has a weight but {folder} has no PNG files")

    def sample(self) -> tuple[np.ndarray, dict]:
        family = self.rng.choices(self.families, weights=self.weights, k=1)[0]
        rng = np.random.default_rng(self.rng.getrandbits(64))
        meta = {"family": family, "path": None, "occluded": False, "split": self.split}
        if family == "driving_real":
            path = self.real_files[int(rng.integers(len(self.real_files)))]
            visible = np.asarray(Image.open(path).convert("L")) >= 128
            meta["path"] = str(path)
        else:
            visible = GENERATORS[family](rng)
        if family in self.occlusion_families and rng.random() < self.occlusion_probability:
            visible = occlude_objects(visible, rng, self.occlusion_fraction)
            meta["occluded"] = True
        return visible.astype(np.uint8) * 255, meta


def training_mask(sampler: MaskPoolSampler, base_seed: int, epoch: int, index: int) -> tuple[np.ndarray, dict]:
    """The mask PwPSingleViewTransferDatasetMask draws for video ``index`` in ``epoch``."""
    sampler.rng.seed(mask_seed(base_seed, epoch, index))
    return sampler.sample()
