"""MEt3R declaration, deliberately unavailable until its official setup is pinned."""

from __future__ import annotations

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext


class MEt3RMetric:
    spec = MetricSpec(
        name="met3r",
        version="official-met3r-required",
        higher_is_better=True,
        requires_gt=False,
        requires_full_360_gt=False,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=True,
        required_dependencies=("official MEt3R", "pinned reconstruction/depth dependencies"),
        expected_input={"range": "official MEt3R input", "resolution": "official MEt3R preprocessing", "frame_count": "multi-view/video protocol"},
        output_metrics=("met3r",),
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            False,
            "Unavailable: install and pin official MEt3R plus its multi-view reconstruction protocol; no geometric proxy is used.",
            {},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        raise RuntimeError(self.availability().reason)
