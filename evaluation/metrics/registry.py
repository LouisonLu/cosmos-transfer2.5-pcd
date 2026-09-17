"""Central registry only; each metric remains implemented in its own module."""

from __future__ import annotations

from evaluation.metrics.base import MetricPlugin
from evaluation.metrics.distribution.fid import FIDMetric
from evaluation.metrics.distribution.fvd import FVDMetric
from evaluation.metrics.geometry.met3r import MEt3RMetric
from evaluation.metrics.panorama.seam import SeamMetric
from evaluation.metrics.quality.qalign import QAlignMetric
from evaluation.metrics.quality.vbench import VBenchMetric
from evaluation.metrics.reference.known_region import KnownRegionPSNRMetric, KnownRegionSSIMMetric
from evaluation.metrics.reference.lpips import LPIPSMetric
from evaluation.metrics.reference.psnr import PSNRMetric
from evaluation.metrics.reference.ssim import SSIMMetric


def plugins() -> dict[str, MetricPlugin]:
    return {
        plugin.spec.name: plugin
        for plugin in (
            PSNRMetric(),
            SSIMMetric(),
            LPIPSMetric(),
            FIDMetric(),
            FVDMetric(),
            VBenchMetric(),
            QAlignMetric(),
            MEt3RMetric(),
            SeamMetric(),
            KnownRegionPSNRMetric(),
            KnownRegionSSIMMetric(),
        )
    }
