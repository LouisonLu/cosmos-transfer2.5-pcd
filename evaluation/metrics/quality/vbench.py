"""VBench declaration, enabled only with its official evaluator."""

from __future__ import annotations

from evaluation.metrics.base import MetricAvailability, MetricSpec, SceneContext


class VBenchMetric:
    spec = MetricSpec(
        name="vbench",
        version="official-vbench-required",
        higher_is_better=True,
        requires_gt=False,
        requires_full_360_gt=False,
        requires_mask=False,
        requires_input_frame=False,
        requires_multiple_generated_views=False,
        required_dependencies=("official VBench",),
        expected_input={"range": "official VBench input", "resolution": "official VBench preprocessing", "frame_count": "official VBench sampler"},
        output_metrics=("vbench",),
    )

    def availability(self) -> MetricAvailability:
        return MetricAvailability(
            False,
            "Unavailable: install and pin the official VBench repository and select its dimensions and video preprocessing protocol.",
            {},
        )

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        raise RuntimeError(self.availability().reason)
