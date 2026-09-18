"""Exact framewise RGB PSNR without resizing or hidden frame sampling."""

from __future__ import annotations

import math

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext
from evaluation.video import has_opencv, iter_rgb_pairs, video_dependency_versions


class PSNRMetric:
    spec = MetricSpec(
        name="psnr",
        version="canonical-rgb-mse-v1",
        higher_is_better=True,
        requires_gt=True,
        requires_full_360_gt=True,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("numpy", "opencv-python"),
        expected_input={"range": "RGB float32 [0, 1]", "resolution": "exact match", "frame_count": "policy.frame_count"},
        output_metrics=("psnr",),
    )

    def availability(self) -> MetricAvailability:
        if not has_opencv():
            return MetricAvailability(False, "Requires opencv-python for bounded video decoding", {})
        return MetricAvailability(True, dependency_versions={"implementation": self.spec.version, **video_dependency_versions()})

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        if context.reference is None:
            raise ValueError("PSNR requires reference video")
        squared_error = 0.0
        element_count = 0
        for _frame, prediction, reference in iter_rgb_pairs(
            context.prediction, context.reference, context.policy, context.progress
        ):
            delta = prediction - reference
            squared_error += float((delta * delta).sum(dtype=float))
            element_count += delta.size
        mse = squared_error / element_count
        return {"psnr": math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)}
