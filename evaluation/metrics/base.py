"""Shared metric-plugin contracts.

Plugins must report unavailable dependencies instead of substituting an
unvalidated approximation.  This lets one evaluation run describe its full
metric plan without silently changing a paper table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MetricSpec:
    name: str
    version: str
    higher_is_better: bool | None
    requires_gt: bool
    requires_full_360_gt: bool
    requires_mask: bool
    requires_input_frame: bool
    requires_multiple_generated_views: bool
    required_dependencies: tuple[str, ...]
    expected_input: dict[str, Any]
    output_metrics: tuple[str, ...]
    granularity: str = "scene"


@dataclass(frozen=True)
class MetricAvailability:
    available: bool
    reason: str | None = None
    dependency_versions: dict[str, str] | None = None


@dataclass(frozen=True)
class SceneContext:
    scene_id: str
    stem: str
    split: str
    cohort: str
    method: str
    prediction: Path
    reference: Path | None
    input_rgb: Path | None
    mask: Path | None
    policy: dict[str, Any]


class MetricPlugin(Protocol):
    spec: MetricSpec

    def availability(self) -> MetricAvailability:
        """Return a concrete setup reason when this plugin cannot run."""

    def evaluate_scene(self, context: SceneContext) -> dict[str, float]:
        """Evaluate one declared scene and return one value per output metric."""


def unavailable_dependency_reason(dependencies: tuple[str, ...], setup: str) -> MetricAvailability:
    return MetricAvailability(
        available=False,
        reason=f"Requires {', '.join(dependencies)}. Setup: {setup}",
        dependency_versions={},
    )
