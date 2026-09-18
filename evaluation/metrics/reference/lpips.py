"""LPIPS adapter with explicit opt-in for learned-metric dependencies."""

from __future__ import annotations

import importlib.metadata
import os

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext
from evaluation.video import has_opencv, iter_rgb_pairs, video_dependency_versions


class LPIPSMetric:
    spec = MetricSpec(
        name="lpips",
        version="official-lpips-alex-v1",
        higher_is_better=False,
        requires_gt=True,
        requires_full_360_gt=True,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("lpips", "torch", "opencv-python"),
        expected_input={"range": "RGB float32 [-1, 1]", "resolution": "exact match", "frame_count": "policy.frame_count"},
        output_metrics=("lpips",),
    )

    def availability(self) -> MetricAvailability:
        if os.environ.get("EVALUATION_ENABLE_LEARNED_METRICS") != "1":
            return MetricAvailability(
                False,
                "Learned metrics are disabled. Setup: install lpips and set EVALUATION_ENABLE_LEARNED_METRICS=1",
                {},
            )
        if not has_opencv():
            return MetricAvailability(False, "Requires opencv-python for bounded video decoding", {})
        try:
            import lpips  # noqa: F401
            import torch

            return MetricAvailability(
                True,
                dependency_versions={
                    "lpips": importlib.metadata.version("lpips"),
                    "torch": torch.__version__,
                    "implementation": self.spec.version,
                    **video_dependency_versions(),
                },
            )
        except (ImportError, importlib.metadata.PackageNotFoundError):
            return MetricAvailability(False, "Requires LPIPS. Setup: pip install lpips", {})

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        if context.reference is None:
            raise ValueError("LPIPS requires reference video")
        import lpips
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net="alex").to(device).eval()
        values = []
        with torch.inference_mode():
            for _frame, prediction, reference in iter_rgb_pairs(
                context.prediction, context.reference, context.policy, context.progress
            ):
                prediction_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
                reference_tensor = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
                values.append(float(model(prediction_tensor, reference_tensor).item()))
        return {"lpips": sum(values) / len(values)}
