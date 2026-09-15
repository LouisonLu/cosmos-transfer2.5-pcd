# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request schema for isolated no-video-path hardlock blended decoding."""

from cosmos_transfer2.config import get_overrides_cls
from cosmos_transfer2.config_no_video_path import InferenceArgumentsNoVideoPath


class InferenceArgumentsNoVideoPathBlendedDecode(InferenceArgumentsNoVideoPath):
    blended_decoding: bool = True
    """Decode the final latent at 0 and 180 degrees and blend them to reduce the ERP seam."""


InferenceOverridesNoVideoPathBlendedDecode = get_overrides_cls(
    InferenceArgumentsNoVideoPathBlendedDecode,
    exclude=[
        "name",
        "edge",
        "depth",
        "vis",
        "seg",
    ],
)
