# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame size of the PwP pipeline, shared by training and inference.

- ``1280x640``: native 2:1 ERP frame. Training resizes the 2:1 video to exactly 1280x640 (no reflection padding,
  ``padding_mask`` all zero); inference reads the 2:1 inputs at 1280x640 (no stretch).
- ``legacy_720``: the original behaviour. Training puts 2:1 into the 16:9 bucket (1280x640 + 32 px reflection
  padding top and bottom); inference stretches 2:1 to 1280x704.

The native size is a new resolution key whose every aspect-ratio bucket is 1280x640. It is added to the size
tables of this process only when selected; the existing resolutions ("720", ...) are never changed.
"""

from cosmos_transfer2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.transfer2.pwp_geometry.method import NATIVE_FRAME_SIZE, check_choice

LEGACY_RESOLUTION = "720"
NATIVE_RESOLUTION = "pwp_erp_1280x640"
NATIVE_WIDTH, NATIVE_HEIGHT = 1280, 640


def register_native_resolution() -> str:
    """Add the ``pwp_erp_1280x640`` key to the video and image size tables (idempotent)."""
    for table in (VIDEO_RES_SIZE_INFO, IMAGE_RES_SIZE_INFO):
        if NATIVE_RESOLUTION not in table:
            table[NATIVE_RESOLUTION] = {key: (NATIVE_WIDTH, NATIVE_HEIGHT) for key in table[LEGACY_RESOLUTION]}
    return NATIVE_RESOLUTION


def resolution_for_frame_size(frame_size: str) -> str:
    """Resolution key used by the datasets and inference readers; also the OmegaConf resolver of the dataset."""
    if check_choice("frame_size", frame_size) == NATIVE_FRAME_SIZE:
        return register_native_resolution()
    return LEGACY_RESOLUTION


def check_native_erp_input(height: int, width: int, source: str, tolerance: float = 0.01) -> None:
    """The native size resizes without padding, so a non-2:1 input would be stretched; refuse it."""
    if abs(width / height - 2.0) > 2.0 * tolerance:
        raise ValueError(
            f"{source} is {width}x{height}, not 2:1. frame size {NATIVE_FRAME_SIZE} only takes 2:1 ERP videos; "
            "use legacy_720 for other aspect ratios."
        )


__all__ = [
    "LEGACY_RESOLUTION",
    "NATIVE_RESOLUTION",
    "check_native_erp_input",
    "register_native_resolution",
    "resolution_for_frame_size",
]
