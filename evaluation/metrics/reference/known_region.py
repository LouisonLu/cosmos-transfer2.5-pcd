"""Frame-0 visible-region preservation metrics for hard-lock evaluations."""

from __future__ import annotations

import math

import numpy as np

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext
from evaluation.video import has_opencv, iter_prediction_input_mask, video_dependency_versions


def _erode_known_region(binary_mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Match the runtime mask convention: white is known and shrinks inward."""
    if erode_px == 0:
        return binary_mask
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - availability protects callers
        raise RuntimeError("OpenCV is required for known-region erosion") from error
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), dtype=np.uint8)
    return cv2.erode(binary_mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)


def _known_arrays(context: SceneContext) -> tuple[np.ndarray, np.ndarray]:
    if context.input_rgb is None or context.mask is None:
        raise ValueError("Known-region metrics require input_rgb and mask")
    _frame, prediction, input_rgb, mask = next(iter_prediction_input_mask(context.prediction, context.input_rgb, context.mask, context.policy))
    threshold = float(context.policy.get("mask_white_threshold", 0.5))
    erosion = int(context.policy.get("known_region_mask_erode_px", context.policy.get("mask_erode_px", 0)))
    known = _erode_known_region(mask.mean(axis=2) >= threshold, erosion)
    if not np.any(known):
        raise ValueError("Known-region mask has no retained pixels after erosion")
    return prediction[known], input_rgb[known]


class KnownRegionPSNRMetric:
    spec = MetricSpec(
        name="known_region_psnr",
        version="canonical-known-region-rgb-mse-v1",
        higher_is_better=True,
        requires_gt=False,
        requires_full_360_gt=False,
        requires_mask=True,
        requires_input_frame=True,
        requires_multiple_generated_views=False,
        required_dependencies=("numpy", "opencv-python"),
        expected_input={"range": "RGB float32 [0, 1]", "resolution": "exact match", "frame_count": "frame 0 only"},
        output_metrics=("known_region_psnr",),
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            has_opencv(),
            None if has_opencv() else "Requires opencv-python",
            {"implementation": self.spec.version, **video_dependency_versions()},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        prediction, input_rgb = _known_arrays(context)
        mse = float(np.mean((prediction - input_rgb) ** 2))
        return {"known_region_psnr": math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)}


class KnownRegionSSIMMetric:
    spec = MetricSpec(
        name="known_region_ssim",
        version="not-applicable-without-spatial-window-v1",
        higher_is_better=True,
        requires_gt=False,
        requires_full_360_gt=False,
        requires_mask=True,
        requires_input_frame=True,
        requires_multiple_generated_views=False,
        required_dependencies=("scikit-image", "opencv-python"),
        expected_input={"range": "RGB float32 [0, 1]", "resolution": "exact match", "frame_count": "frame 0 only"},
        output_metrics=("known_region_ssim",),
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            False,
            "Unavailable by design: a spatially masked SSIM requires an explicitly selected, validated masked-SSIM protocol; no approximation is used.",
            {},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        raise RuntimeError(self.availability().reason)
