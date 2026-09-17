"""FVD declaration without an unvalidated substitute."""

from __future__ import annotations

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext


class FVDMetric:
    spec = MetricSpec(
        name="fvd",
        version="unconfigured-official-implementation-required",
        higher_is_better=False,
        requires_gt=True,
        requires_full_360_gt=True,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("pinned official FVD implementation", "pinned video feature weights"),
        expected_input={"range": "implementation-defined", "resolution": "implementation-defined", "frame_count": "implementation-defined clips"},
        output_metrics=("fvd",),
        granularity="dataset",
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            False,
            "Unavailable: pin an official FVD implementation, feature backbone, clip sampler, and ERP projection policy before use.",
            {},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        raise RuntimeError(self.availability().reason)
