"""FID declaration.

FID is set-level, not scene-level. It remains unavailable until this project
pins one official feature extractor and preprocessing protocol.
"""

from __future__ import annotations

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext


class FIDMetric:
    spec = MetricSpec(
        name="fid",
        version="unconfigured-official-implementation-required",
        higher_is_better=False,
        requires_gt=True,
        requires_full_360_gt=True,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("pinned official FID implementation", "pinned Inception feature weights"),
        expected_input={"range": "implementation-defined", "resolution": "implementation-defined", "frame_count": "set-level frames"},
        output_metrics=("fid",),
        granularity="dataset",
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            False,
            "Unavailable: pin an official FID implementation, Inception weights, frame sampling, and ERP projection policy before use.",
            {},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        raise RuntimeError(self.availability().reason)
