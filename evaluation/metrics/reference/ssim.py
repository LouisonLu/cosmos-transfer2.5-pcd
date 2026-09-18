"""SSIM via scikit-image's documented implementation, never an approximation."""

from __future__ import annotations

import importlib.metadata

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext
from evaluation.video import has_opencv, iter_rgb_pairs, video_dependency_versions


class SSIMMetric:
    spec = MetricSpec(
        name="ssim",
        version="scikit-image-structural_similarity-v1",
        higher_is_better=True,
        requires_gt=True,
        requires_full_360_gt=True,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("scikit-image", "opencv-python"),
        expected_input={
            "range": "RGB float32 [0, 1]",
            "resolution": "exact match",
            "frame_count": "policy.frame_count",
            "implementation": "skimage.metrics.structural_similarity",
        },
        output_metrics=("ssim",),
    )

    def availability(self) -> MetricAvailability:
        if not has_opencv():
            return MetricAvailability(False, "Requires opencv-python for bounded video decoding", {})
        try:
            version = importlib.metadata.version("scikit-image")
            from skimage.metrics import structural_similarity  # noqa: F401
        except (ImportError, importlib.metadata.PackageNotFoundError):
            return MetricAvailability(False, "Requires scikit-image. Setup: pip install scikit-image", {})
        return MetricAvailability(
            True,
            dependency_versions={"scikit-image": version, "implementation": self.spec.version, **video_dependency_versions()},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        if context.reference is None:
            raise ValueError("SSIM requires reference video")
        from skimage.metrics import structural_similarity

        ssim_policy = context.policy.get("ssim", {})
        win_size = int(ssim_policy.get("win_size", 11))
        gaussian_weights = bool(ssim_policy.get("gaussian_weights", True))
        sigma = float(ssim_policy.get("sigma", 1.5))
        use_sample_covariance = bool(ssim_policy.get("use_sample_covariance", False))
        values = []
        for _frame, prediction, reference in iter_rgb_pairs(
            context.prediction, context.reference, context.policy, context.progress
        ):
            values.append(
                float(
                    structural_similarity(
                        prediction,
                        reference,
                        data_range=1.0,
                        channel_axis=2,
                        win_size=win_size,
                        gaussian_weights=gaussian_weights,
                        sigma=sigma,
                        use_sample_covariance=use_sample_covariance,
                    )
                )
            )
        return {"ssim": sum(values) / len(values)}
