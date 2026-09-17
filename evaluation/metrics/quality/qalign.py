"""Q-Align declaration, enabled only with its official checkpoint and protocol."""

from __future__ import annotations

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext


class QAlignMetric:
    spec = MetricSpec(
        name="qalign",
        version="official-qalign-required",
        higher_is_better=True,
        requires_gt=False,
        requires_full_360_gt=False,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("official Q-Align", "pinned Q-Align checkpoint"),
        expected_input={"range": "official Q-Align input", "resolution": "official Q-Align preprocessing", "frame_count": "declared frame sampler"},
        output_metrics=("qalign",),
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            False,
            "Unavailable: install and pin the official Q-Align evaluator/checkpoint and define its video frame-sampling policy.",
            {},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        raise RuntimeError(self.availability().reason)
