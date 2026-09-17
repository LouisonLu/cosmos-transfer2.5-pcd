"""ERP cyclic seam metrics reused from the verified seam-inspection tool.

The formulas are intentionally aligned with
``/Users/louisonlu/Desktop/research/360video/tools/inspect_erp_seams.py``:
RGB is normalized to [0, 1], the ERP seam is the cyclic left/right boundary,
and ordinary horizontal jumps provide the per-frame content baseline.
"""

from __future__ import annotations

import numpy as np

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext
from evaluation.video import has_opencv, iter_prediction_frames, video_dependency_versions


def cyclic_seam_metrics(frame_rgb: np.ndarray) -> dict[str, float]:
    seam_l1 = float(np.abs(frame_rgb[:, 0] - frame_rgb[:, -1]).mean())
    interior_jumps = np.abs(frame_rgb[:, 1:] - frame_rgb[:, :-1]).mean(axis=(0, 2))
    median_jump = float(np.median(interior_jumps))
    seam_ratio = seam_l1 / max(median_jump, 1e-6)
    seam_percentile = float(100.0 * np.mean(interior_jumps <= seam_l1))
    left_slope = frame_rgb[:, 1] - frame_rgb[:, 0]
    right_slope = frame_rgb[:, -1] - frame_rgb[:, -2]
    edge_slope_mismatch = float(np.abs(left_slope - right_slope).mean())
    return {
        "seam_l1": seam_l1,
        "seam_ratio": seam_ratio,
        "seam_percentile": seam_percentile,
        "edge_slope_mismatch": edge_slope_mismatch,
    }


class SeamMetric:
    spec = MetricSpec(
        name="seam",
        version="inspect-erp-seams-cyclic-v1",
        higher_is_better=False,
        requires_gt=False,
        requires_full_360_gt=False,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("numpy", "opencv-python"),
        expected_input={"range": "RGB float32 [0, 1]", "resolution": "even-width ERP", "frame_count": "policy.frame_count"},
        output_metrics=("seam_l1", "seam_ratio", "seam_percentile", "temporal_seam_residual", "edge_slope_mismatch"),
    )

    def availability(self) -> MetricAvailability:
        if not has_opencv():
            return MetricAvailability(False, "Requires opencv-python for bounded video decoding", {})
        return MetricAvailability(True, dependency_versions={"implementation": self.spec.version, **video_dependency_versions()})

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        values: dict[str, list[float]] = {name: [] for name in self.spec.output_metrics if name != "temporal_seam_residual"}
        residuals: list[float] = []
        previous_delta: np.ndarray | None = None
        for _frame_index, frame in iter_prediction_frames(context.prediction, context.policy):
            if frame.shape[1] % 2 != 0:
                raise ValueError(f"ERP seam metric requires even width: {context.prediction}")
            frame_values = cyclic_seam_metrics(frame)
            for name, value in frame_values.items():
                values[name].append(value)
            seam_delta = frame[:, 0] - frame[:, -1]
            residuals.append(0.0 if previous_delta is None else float(np.abs(seam_delta - previous_delta).mean()))
            previous_delta = seam_delta
        return {
            **{name: float(np.mean(metric_values)) for name, metric_values in values.items()},
            "temporal_seam_residual": float(np.mean(residuals)),
        }
