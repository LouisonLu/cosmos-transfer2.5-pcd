# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Single-view Transfer dataset for local video files with control inputs.

This dataset loader is designed for post-training Cosmos-Transfer2 models with local data.
It uses the full Transfer2 augmentor pipeline including:
- Randomized edge detection thresholds for training diversity
- Automatic resizing with aspect ratio preservation
- Reflection padding
- Text transforms for caption handling
- Control input generation (edge, depth, seg, blur, etc.)

Example usage:
    dataset = SingleViewTransferDatasetMask(
        dataset_dir="datasets/example",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,  # Enable augmentations for training
    )
"""

import base64
import importlib.util
import json
import os
import pickle
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import (
    get_video_augmentor_v2_with_control,
    get_video_augmentor_v2_with_control_and_image_context,
)
from cosmos_transfer2._src.transfer2.utils.input_handling import detect_aspect_ratio


# Mock URL object for augmentor compatibility
class MockUrlMeta:
    """Mock metadata object for WebDataset compatibility."""

    def __init__(self):
        self.opts = {}


class MockUrl:
    """Mock URL object that augmentors expect from WebDataset."""

    def __init__(self, url: str):
        self._url = url
        self.meta = MockUrlMeta()

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"MockUrl({self._url})"


# Mappings between control types and corresponding sub-folder names in the data folder
CTRL_TYPE_INFO = {
    "keypoint": {"folder": "keypoint", "format": "pickle", "data_dict_key": "keypoint"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "seg": {"folder": "seg", "format": "mp4", "data_dict_key": "segmentation"},
    "edge": {"folder": None},  # Canny edge, computed on-the-fly by augmentor
    "vis": {"folder": None},  # Blur, computed on-the-fly by augmentor
}


WAYMO_MASK_SHAPE = (2048, 4096)
WAYMO_MASK_PACKED_ZLIB_B64 = (
    "eNrt3c+La1cBwPGTH5o8i82oiy5cJFJE6qaCWyGpFTroouI/8P6UBBTsrsW/4KFdvYJ04crNBATp8lHcN5UuioimrcJUprkm8+O9"
    "zMxNcu85mZfJzeeD9GGbk5u533vOvTeTmRcCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwP0y7mfPzOyPA1G//PPl6XCpf3bxL5vP"
    "/jv7plnoQYP5P97Mck3PH9ErvsWfh9GPwvSDMIw4CNs/fO9qBXp88q/Hg/JH8U/mIxfeyz45vNij1TN78+zPVhiVPNpuDByc/6+g"
    "/s1NXwytFX6C2s2zV6/gFKiEXuS48cr4lx3qhcq/dDp/7Mn1obN2kf2/eP6PGnmbHp8W/SJeejTMGf/kwM9dRebe0aizvv/vis3g"
    "3LGPCi5cjdT1Z7jm9HUgannH/2YPs41mm+d/f9XYv/1q4yt40Fi77fGm8T9ddwCfHcxVfCfqi69lBWyahd11gzev3es3vnESt5Je"
    "eyWsnkFvT8uv2mV3YtLy0d28/IzLr/xL/ncAl32dyPnXKtg/Gy+upI5inyPqyL229Rzt80vHAuPPyt1F7N/Kv2kHvJN3LTjv+bWs"
    "uD+9387d+C+GWZECOTeC7Q3H7bJ38y5cF1VPio3Pvz2uiE7s/GuV6J9Nkk4g4/wb1G7RrX+afydf9Pi5Ol4qqFVkAmaDW4d/vT7M"
    "SpmNbhf4cfHR7968F28XrrfqGO4WH/3VuD6oZP5asa9/Gj93nvp79OzPfQVHq28aC34FJfpnp9Hvj91r8fvvYVbaNO7YW3EdXvb4"
    "u3kb0C03/C/jw539FxMgoV3+dXjJ0ZOkfsnjH1Uv/zB6/3WyKKXv3FYfgsMsbXz54RX7RMOg3PX79XcD4/JnX6UtIOOra8BexNGz"
    "fAQfhVcTj94K6JeuN728D2pmsZ5OwW9GDf9wcHHk1oZxm59efr+61o8a/s/qvOcTwmtR9erzuTfoRPdfvBPcXLyVE/cUF0twLWr1"
    "f7qEzW9DX49evkYhVOFO4PtxF3BZe3EEfJEluNh7/dj1Y7J4I/bFj+I3H8KTsie+5TNQRT4QcBxZ4K+D+fQZpvRfTOF6I3r4f1KO"
    "nvTxk4p8L6AR23Aafe5dOokcn8QP/3A+PGXzTyJXvqW7wAr0fz326z8LjSzROGX+LlbwLG39eSNl+Fkl+v8o26HPW7vcejY7STx+"
    "Du6NH5b9fs/bNzd/w5+I72RX6Rv+rLuAPd7z7/hLmGa/PxP8oK9g4hXkXvfvCqg/B9vf8m/+oz/6oz/6oz/6oz/6oz/6oz/6oz/6"
    "oz/666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/+2/VQ"
    "QP052P76HXb/oYD6oz/6oz/6oz/6oz/6oz/6oz/6oz/6o7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+/"
    "/vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+/"
    "/vrrr7/+6I/+6I/+6I/+6I/+6I/+6I/+6I/+6I/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/++uuvv/7666+//vrrr7/+"
    "+uuvv/76P0d9AfXH+o/+6I/+6I/rf/RHfyrbvyug/uxD/571/xD698z/KvWvly0yuPzz6PLPxTZGd3Zg6H8P1/+fLf7xYPm9mSfm"
    "/971b16bzqV1bm0phNlA/3ve/9l6X8v+/PiPnaOyRWrZ6GTlxibb7v9QwLua/69czdrUeX/d48fm/z3u38iy07xH9a/N8ZU9imzx"
    "6kahpv896T8oNH2XrgwGNy/mz08TLxTe6H/nj27nnTyWTkTfu7Vd/e+k/7cv9+WDgnP3ePDo8uJw6aLheFpqq9fvEfOOglefPXiq"
    "/12v/7X5GlDo4ePFo9+6/bZArex2RyvPIPlP9evclcD7f9vp/90vCw+YdsLNdu1JxIbHuTecK4+jL3L7m//p/Zvz2VxiwGl/3Fu+"
    "Yvh6dIX64MYSsvkcMuvMRvpvt38jK7sX//DW0in5zVb0th+VunPMuxrQP71/p/ygf8+2dAqOepaJ/rv3xSTuui//MrDcGvLZQP/7"
    "ceHYSn6a38wvH98uO2jSPH/XYqT/Lv12cjRMfpKzqDPI1XeS9N+l6QtbeJJXI7fdXLzv9FCEgzUKPe//HPTyM9P/wK9B9QcAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAZae1hMEn9t/e9w+h"
    "Ezl0Evr2XwX6h6iO08hx3L/+IQwj6utfnf7ljoBPL8foX53+xWuWH8E+9A+htfnxo+XH61+t/mH9zcDZjcfqX73+c7XVi/6R/tXv"
    "31w+FTSy7JOwiv5V7D83uPijHtbTv6L9C9Jff/RHf/RHf/RHf6rf/8T+M//Zab/ODvv/wP7f/fyt7a5/4rbZzvqdsAzPkvKP6qU/"
    "OMgdnL/fiB09Dcm6Guz++q05fP6T/0pDhHtw/R5RYRK2JPb8M3b22OL9Wyd2YLqXo176i24gt3v/XuJUfBa2aRBxFfDxp95A2Pr7"
    "N91d1I/6CaLPvIF0B/3bRT7CPw13o8SrPupdfLhQ/+31H4RekZm4+BB3b9DeZvj5dkchfGNQ9Ah4/5eRiwYb5//aK/JZuGubJ/Q/"
    "wiuDkW8g3E3/NT/MNQ3PR63Aqaen/93336Hhitv91ItG9qP/3HGjyKs0/6vaf/UPFel/oP3N/8Pt39T/kPsPzH/rv/7666+/+3/9"
    "y3/LCP3RH+d/9Ed/9Ndff/31119//fXXX3/99ddff/31119//fXXv5r9x7vc+x/rv/P+CU8yS/wEwnj3n2A49P6LnyjtRo49C7Wk"
    "j6COkj/Ccpa8B/a7f0jPv/g1Eq2osZ9fvITYo+fyr8SrJxT8Mn0K7Hn/kD77z3+6JHb2huij59nv0Omntesn7YHRnvdPOv4n4enf"
    "UdeNzhf3W2wnS+PjDqCnPw/fTdgDYf/FH//j5adpxA+NeA3Xp10tLV30r/CehSrobOnY70ZN/pij57SefBCX/Xu3i30Z+6mxrS8+"
    "ZdHsp51zW1nK+hO1gEx7oSoiLt5z/27aWvldf6X9WsSZP/JC5nQbZ8FRVeKPIq6Bzr/4esQRsBh4dGtos/A6dLqFZezRraHvlF5A"
    "zkJ1tMuugKft9uV9X7mVfONOe6Fo/Zu/z2LQK3wlM97GndAHoXK6qStfb/0hcHmxXG9Gn4rGi5Ujr36JL2JyccDX8xbB7xzi5A/P"
    "VvJh0vn35nLQjd1hjTVvNW3WKnTkNm+2LzMHQlXVks6/29RKmW/9LP4I6h/OdV/EncD0+S9Loy0cPxfVBts4D4bK65a4br7L9PWt"
    "vLsds159654c/rtxnL8IfhAOSu51T/uAdkBrz9/mrq/9v+UOgkkgdh9uQzPydbbrES+6LnTsVdc9OfAGCZsaXL3HwEHq2QUAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAwKX/A5c8BDo="
)


class SingleViewTransferDatasetMask(Dataset):
    """Dataset class for loading single-view video-to-video generation data with control inputs.

    This dataset is designed for post-training Cosmos-Transfer2 models with local video files.
    It supports various control modalities including depth, segmentation, edge, and blur.

    Dataset structure:
        dataset_dir/
        ├── videos/
        │   ├── video1.mp4
        │   └── video2.mp4
        ├── captions/
        │   ├── video1.json  ({"caption": "text description"})
        │   └── video2.json
        └── <control_type>/  (optional for depth/seg, computed on-the-fly for edge/vis)
            ├── video1.mp4  (for depth)
            └── video1.pickle  (for seg/keypoint)

    Args:
        dataset_dir: Base path to the dataset directory
        num_frames: Number of frames to load per sequence
        video_size: Target size (H, W) for video frames
        resolution: Resolution key for augmentor (e.g., "720", "1080")
        hint_key: Control input type (e.g., "control_input_edge", "control_input_depth")
        is_train: Whether this is for training (affects sampling)
        caption_type: Type of caption to load (default: "t2w_qwen2p5_7b")
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        resolution: str = "720",
        hint_key: str = "control_input_edge",
        is_train: bool = True,
        caption_type: str = "t2w_qwen2p5_7b",  # Use Qwen2.5-7B caption type
        input_video_dir: str | None = None,
        target_video_dir: str | None = None,
        input_video_suffix: str = "_vis_pcd_firstframe",
        control_video_dir_override: str | None = None,
        control_video_suffix: str | None = None,
        use_image_context: bool = False,
        image_context_from_rgb_first_frame: bool = False,
        mask_image_context: bool = False,
        mask_depth_control: bool = False,
        mask_mode: str | None = None,
        image_context_keep_ratio: float = 0.25,
        image_context_mask_mode: str = "fixed_vertical_band",
        image_context_mask_fill_value: int = 0,
        image_context_mask_reference_path: str | None = None,
        image_context_mask_reference_threshold: int = 5,
        mask_pool_root: str | None = None,
        mask_pool_use_dynamic: bool = False,
        mask_pool_sampler_py: str | None = None,
        mask_pool_sampler_seed: int = 0,
        mask_pool_sampler_kwargs_json: str | dict[str, Any] | None = None,
        **kwargs,  # Accept extra params for config compatibility (like MultiviewTransferDataset)
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.video_size = video_size
        self.resolution = resolution
        self.is_train = is_train
        self.caption_type = caption_type
        self.input_video_suffix = input_video_suffix
        self.control_video_suffix = control_video_suffix or input_video_suffix
        self.use_image_context = use_image_context
        self.image_context_from_rgb_first_frame = image_context_from_rgb_first_frame
        self.mask_image_context = mask_image_context
        self.mask_depth_control = mask_depth_control
        self.image_context_keep_ratio = image_context_keep_ratio
        self.image_context_mask_mode = mask_mode or image_context_mask_mode
        self.image_context_mask_fill_value = image_context_mask_fill_value
        self.image_context_mask_reference_path = image_context_mask_reference_path
        self.image_context_mask_reference_threshold = image_context_mask_reference_threshold
        self.use_reference_mask = self.image_context_mask_reference_path is not None
        self._image_context_reference_keep_mask: np.ndarray | None = None
        self.mask_pool_root = mask_pool_root
        self.mask_pool_use_dynamic = mask_pool_use_dynamic
        self.mask_pool_sampler_py = mask_pool_sampler_py
        self.mask_pool_sampler_seed = int(mask_pool_sampler_seed)
        self.mask_pool_sampler_kwargs_json = mask_pool_sampler_kwargs_json
        self._dynamic_mask_sampler = None
        self._dynamic_mask_sampler_worker_id: int | None = None
        self._static_mask_pool_paths: list[Path] = []

        if not 0.0 < self.image_context_keep_ratio <= 1.0:
            raise ValueError("image_context_keep_ratio must be in (0, 1].")
        valid_mask_modes = {
            "fixed_vertical_band",
            "fixed_center",
            "fixed_horizontal_band",
            "waymo",
            "front_rear_luma_erp",
            "random_rectangles",
            "random_pixels",
        }
        if not self.use_reference_mask and self.image_context_mask_mode not in valid_mask_modes:
            raise ValueError(
                f"Unsupported image_context_mask_mode: {self.image_context_mask_mode}. "
                f"Supported modes: {sorted(valid_mask_modes)}"
            )
        if self.use_reference_mask and not os.path.exists(self.image_context_mask_reference_path):
            raise FileNotFoundError(
                f"Image context mask reference not found: {self.image_context_mask_reference_path}"
            )
        if self.mask_pool_root:
            if not os.path.isdir(self.mask_pool_root):
                raise FileNotFoundError(f"mask_pool_root not found: {self.mask_pool_root}")
            if self.mask_pool_use_dynamic:
                if not self.mask_pool_sampler_py:
                    raise ValueError("mask_pool_sampler_py must be set when mask_pool_use_dynamic=True")
                if not os.path.exists(self.mask_pool_sampler_py):
                    raise FileNotFoundError(f"mask_pool_sampler_py not found: {self.mask_pool_sampler_py}")
            else:
                self._static_mask_pool_paths = sorted(Path(self.mask_pool_root).rglob("*.png"))
                if not self._static_mask_pool_paths:
                    raise FileNotFoundError(f"No PNG masks found under mask_pool_root: {self.mask_pool_root}")

        # Parse control type from hint_key
        self.hint_key = hint_key
        self.ctrl_type = hint_key.replace("control_input_", "")
        if self.ctrl_type not in CTRL_TYPE_INFO:
            raise ValueError(
                f"Unsupported control type: {self.ctrl_type}. Supported types: {list(CTRL_TYPE_INFO.keys())}"
            )
        self.ctrl_config = CTRL_TYPE_INFO[self.ctrl_type]

        # Set up directories
        target_dir = os.path.join(self.dataset_dir, target_video_dir or "videos")
        self.video_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".mp4")])

        self.input_video_dir = None
        self.input_video_paths: dict[str, str] = {}
        if input_video_dir is not None:
            input_dir = os.path.join(self.dataset_dir, input_video_dir)
            if os.path.isdir(input_dir):
                self.input_video_dir = input_dir
                for video_path in self.video_paths:
                    video_name = os.path.basename(video_path).replace(".mp4", "")
                    input_name = f"{video_name}{self.input_video_suffix}.mp4"
                    self.input_video_paths[video_name] = os.path.join(input_dir, input_name)
            else:
                log.warning(f"Input video dir does not exist: {input_dir}. Falling back to target videos.")

        # Support both "captions/" and "metas/" directories
        self.caption_dir = os.path.join(self.dataset_dir, "captions")
        if not os.path.exists(self.caption_dir):
            self.caption_dir = os.path.join(self.dataset_dir, "metas")

        # Note: We no longer load T5 embeddings - captions are encoded on-the-fly
        # by the model's text encoder (Qwen2.5-VL-7B / reason1p1_7B)
        self.num_failed_loads = 0
        self.bad_video_indices = set()  # Track videos that fail to load (too short, corrupted, etc.)

        # Use proper augmentor pipeline for training quality
        # This includes randomized edge detection, reflection padding, and text transforms
        # Pass embedding_type=None since we're handling T5 embeddings ourselves
        # (if embedding_type is set, the function returns early with only video_parsing)
        if self.use_image_context:
            augmentor_config = get_video_augmentor_v2_with_control_and_image_context(
                resolution=resolution,
                caption_type=caption_type,
                embedding_type=None,  # We handle embeddings ourselves, get full augmentor pipeline
                control_input_type=self.ctrl_type,
                use_random=is_train,  # Enable random augmentations for training
            )
        else:
            augmentor_config = get_video_augmentor_v2_with_control(
                resolution=resolution,
                caption_type=caption_type,
                embedding_type=None,  # We handle embeddings ourselves, get full augmentor pipeline
                control_input_type=self.ctrl_type,
                use_random=is_train,  # Enable random augmentations for training
            )

        # Filter out augmentors that don't apply to local datasets
        # The augmentor pipeline includes augmentors designed for S3/WebDataset that need to be skipped:
        # - video_parsing: Decodes video bytes from S3 → we already load tensors from local MP4 files
        # - depth_parsing: Decodes depth bytes from S3 key "depth_pervideo_video_depth_anything" → we load from local depth/ folder
        # - seg_parsing: Decodes seg bytes from S3 key "segmentation_sam2_color_video_v2" → we load from local seg/ folder
        # - merge_datadict: Merges multiple WebDataset shards → not needed for single local dataset
        # - text_transform: Loads pre-computed T5 embeddings → we pass raw captions for on-the-fly encoding
        skip_augmentors = [
            "video_parsing",
            "video_parsing_with_image_context",
            "merge_datadict",
            "text_transform",
            "depth_parsing",
            "seg_parsing",
        ]
        augmentor_config = {k: v for k, v in augmentor_config.items() if k not in skip_augmentors}

        log.info(f"Filtered augmentors: {list(augmentor_config.keys())}")

        # Instantiate augmentors
        self.augmentor = {k: instantiate(v) for k, v in augmentor_config.items()}

        # Double-check text_transform is not present
        if "text_transform" in self.augmentor:
            raise RuntimeError("text_transform should have been filtered out but is still present!")

        log.info(f"Initialized SingleViewTransferDatasetMask with {len(self.video_paths)} videos")
        log.info(f"  Dataset dir: {self.dataset_dir}")
        log.info(f"  Control type: {self.ctrl_type}")
        log.info(f"  Resolution: {resolution}, Video size: {video_size}")
        log.info(f"  Required frames: {self.sequence_length}")
        if self.use_image_context:
            log.info("  Image context: enabled")
            if self.mask_image_context:
                if self.use_reference_mask:
                    log.info(
                        "  Image context mask: "
                        f"reference_path={self.image_context_mask_reference_path}, "
                        f"threshold={self.image_context_mask_reference_threshold}, "
                        f"fill={self.image_context_mask_fill_value}"
                    )
                else:
                    log.info(
                        "  Image context mask: "
                        f"mode={self.image_context_mask_mode}, "
                        f"keep_ratio={self.image_context_keep_ratio}, "
                        f"fill={self.image_context_mask_fill_value}"
                    )
        if self.mask_depth_control:
            log.info("  Depth / PCD control masking: enabled")
        if self.mask_pool_use_dynamic:
            log.info(
                "  Dynamic mask pool: "
                f"root={self.mask_pool_root}, sampler={self.mask_pool_sampler_py}, seed={self.mask_pool_sampler_seed}"
            )
        elif self._static_mask_pool_paths:
            log.info(
                "  Static mask pool: "
                f"root={self.mask_pool_root}, masks={len(self._static_mask_pool_paths)}, sampling=uniform"
            )
        if self.input_video_dir:
            log.info(f"  Input video dir: {self.input_video_dir}")

        # Optional override for control video directory
        self.control_video_dir_override = None
        if control_video_dir_override is not None:
            override_dir = os.path.join(self.dataset_dir, control_video_dir_override)
            if os.path.isdir(override_dir):
                self.control_video_dir_override = override_dir
            else:
                log.warning(f"Control video dir override does not exist: {override_dir}")
        elif self.ctrl_type == "depth" and self.input_video_dir is not None:
            depth_dir = os.path.join(self.dataset_dir, "depth")
            if not os.path.isdir(depth_dir):
                self.control_video_dir_override = self.input_video_dir

        # Quick validation: check for obviously bad videos (optional, can be slow for large datasets)
        # self._validate_videos()  # Uncomment to pre-filter bad videos at initialization

    def __str__(self) -> str:
        return f"SingleViewTransferDatasetMask: {len(self.video_paths)} videos from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _validate_videos(self) -> None:
        """Validate all videos and pre-mark bad ones (too short, corrupted, etc.).

        This is optional and can be slow for large datasets, but helps identify
        problematic videos upfront. Call this in __init__ if you want pre-filtering.
        """
        log.info("Validating videos for minimum frame count...")
        bad_count = 0

        for idx, video_path in enumerate(self.video_paths):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                del vr

                if total_frames < self.sequence_length:
                    self.bad_video_indices.add(idx)
                    bad_count += 1
                    log.debug(
                        f"Marking video {idx} as bad: {os.path.basename(video_path)} "
                        f"has only {total_frames} frames (need {self.sequence_length})"
                    )
            except Exception as e:
                self.bad_video_indices.add(idx)
                bad_count += 1
                log.debug(f"Marking video {idx} as bad: {os.path.basename(video_path)} - {e}")

        valid_count = len(self.video_paths) - bad_count
        log.info(
            f"Video validation complete: {valid_count} valid, {bad_count} bad "
            f"({bad_count / len(self.video_paths) * 100:.1f}% filtered)"
        )

    def _load_video(self, video_path: str, frame_ids: list[int] | None = None) -> tuple[np.ndarray, float, list[int]]:
        """Load video frames from file.

        Args:
            video_path: Path to video file
            frame_ids: Specific frame indices to load. If None, randomly samples frames.

        Returns:
            Tuple of (frames, fps, frame_ids)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)

        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # Sample frames if not provided
        if frame_ids is None:
            max_start_idx = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start_idx + 1) if self.is_train else 0
            frame_ids = list(range(start_frame, start_frame + self.sequence_length))

        # Load frames
        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # Reset video reader

        # Debug: Log frame loading
        log.info(
            f"Loaded video {os.path.basename(video_path)}: "
            f"total_frames={total_frames}, "
            f"requested={len(frame_ids)}, "
            f"loaded={frame_data.shape[0]}"
        )

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 24  # Default FPS

        del vr
        return frame_data, fps, frame_ids

    def _load_first_frame(self, video_path: str) -> np.ndarray:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        frame = vr.get_batch([0]).asnumpy()[0]
        del vr
        return frame

    def _get_dynamic_mask_sampler(self) -> Any:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        if self._dynamic_mask_sampler is not None and self._dynamic_mask_sampler_worker_id == worker_id:
            return self._dynamic_mask_sampler

        assert self.mask_pool_sampler_py is not None
        assert self.mask_pool_root is not None

        module_name = f"_dynamic_mask_sampler_{worker_id}_{abs(hash(self.mask_pool_sampler_py))}"
        spec = importlib.util.spec_from_file_location(module_name, self.mask_pool_sampler_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load mask sampler module from {self.mask_pool_sampler_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        sampler_cls = getattr(module, "MaskPoolSampler", None) or getattr(module, "MaskPoolSamplerV11", None)
        if sampler_cls is None:
            raise AttributeError(
                f"Mask sampler module must define MaskPoolSampler or MaskPoolSamplerV11: {self.mask_pool_sampler_py}"
            )

        sampler_seed = self.mask_pool_sampler_seed + worker_id
        sampler_kwargs = self._parse_mask_pool_sampler_kwargs()
        self._dynamic_mask_sampler = sampler_cls(self.mask_pool_root, seed=sampler_seed, **sampler_kwargs)
        self._dynamic_mask_sampler_worker_id = worker_id
        return self._dynamic_mask_sampler

    def _parse_mask_pool_sampler_kwargs(self) -> dict[str, Any]:
        raw = self.mask_pool_sampler_kwargs_json
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if not isinstance(raw, str):
            raise TypeError(
                "mask_pool_sampler_kwargs_json must be None, a JSON string, a JSON file path, or a dict."
            )

        text = raw.strip()
        if not text:
            return {}
        if os.path.exists(text):
            with open(text, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("mask_pool_sampler_kwargs_json must resolve to a JSON object/dict.")
        return data

    def _sample_dynamic_mask(self) -> tuple[np.ndarray, dict[str, Any] | None]:
        sampler = self._get_dynamic_mask_sampler()
        sample = sampler.sample()
        if isinstance(sample, tuple) and len(sample) == 2:
            mask, meta = sample
        else:
            mask, meta = sample, None
        mask = np.asarray(mask, dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if mask.ndim != 2 or not values.issubset({0, 255}):
            raise ValueError(
                f"Dynamic mask sampler returned invalid mask: shape={mask.shape}, values={sorted(values)}"
            )
        return mask, meta

    def _sample_static_pool_mask(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._static_mask_pool_paths:
            raise ValueError("Static mask pool is empty.")
        idx = int(np.random.randint(0, len(self._static_mask_pool_paths)))
        path = self._static_mask_pool_paths[idx]
        with Image.open(path) as im:
            mask = np.asarray(im.convert("L"), dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if mask.ndim != 2 or not values.issubset({0, 255}):
            raise ValueError(f"Static mask pool returned invalid mask: path={path}, shape={mask.shape}, values={sorted(values)}")
        return mask, {"path": str(path), "sampling": "uniform"}

    def _resize_keep_mask(self, keep_mask: np.ndarray, height: int, width: int) -> np.ndarray:
        keep_mask = keep_mask.astype(bool, copy=False)
        if keep_mask.shape == (height, width):
            return keep_mask
        keep_img = Image.fromarray(keep_mask.astype(np.uint8) * 255)
        keep_img = keep_img.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(keep_img, dtype=np.uint8) > 0

    def _get_active_keep_mask(self, height: int, width: int, sampled_mask: np.ndarray | None = None) -> np.ndarray:
        if sampled_mask is not None:
            return self._resize_keep_mask(sampled_mask == 255, height, width)
        if self.image_context_mask_reference_path is not None:
            return self._get_reference_keep_mask(height, width)
        keep_ratio = self.image_context_keep_ratio
        if self.image_context_mask_mode == "fixed_vertical_band":
            keep = np.zeros((height, width), dtype=bool)
            keep_w = max(1, int(round(width * keep_ratio)))
            x0 = max(0, (width - keep_w) // 2)
            keep[:, x0 : x0 + keep_w] = True
            return keep
        if self.image_context_mask_mode == "fixed_horizontal_band":
            keep = np.zeros((height, width), dtype=bool)
            keep_h = max(1, int(round(height * keep_ratio)))
            y0 = max(0, (height - keep_h) // 2)
            keep[y0 : y0 + keep_h, :] = True
            return keep
        if self.image_context_mask_mode == "fixed_center":
            keep = np.zeros((height, width), dtype=bool)
            side_ratio = keep_ratio**0.5
            keep_h = max(1, int(round(height * side_ratio)))
            keep_w = max(1, int(round(width * side_ratio)))
            y0 = max(0, (height - keep_h) // 2)
            x0 = max(0, (width - keep_w) // 2)
            keep[y0 : y0 + keep_h, x0 : x0 + keep_w] = True
            return keep
        if self.image_context_mask_mode == "random_pixels":
            return np.random.random((height, width)) < keep_ratio
        if self.image_context_mask_mode in {"waymo", "front_rear_luma_erp"}:
            return self._get_waymo_keep_mask(height, width)
        if self.image_context_mask_mode == "random_rectangles":
            keep = np.zeros((height, width), dtype=bool)
            target_pixels = int(round(height * width * keep_ratio))
            attempts = 0
            while keep.sum() < target_pixels and attempts < 64:
                rect_h = np.random.randint(max(1, height // 8), max(2, height // 2))
                rect_w = np.random.randint(max(1, width // 8), max(2, width // 2))
                y0 = np.random.randint(0, max(1, height - rect_h + 1))
                x0 = np.random.randint(0, max(1, width - rect_w + 1))
                keep[y0 : y0 + rect_h, x0 : x0 + rect_w] = True
                attempts += 1
            if keep.sum() > target_pixels:
                ys, xs = np.where(keep)
                drop_count = int(keep.sum() - target_pixels)
                if drop_count > 0:
                    drop_idx = np.random.choice(len(ys), size=drop_count, replace=False)
                    keep[ys[drop_idx], xs[drop_idx]] = False
            return keep
        raise ValueError(
            "No mask source available. Use a reference mask, a dynamic mask pool, or a supported built-in mask mode."
        )

    def _apply_keep_mask_to_frame(self, frame: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        masked = np.full_like(frame, np.clip(self.image_context_mask_fill_value, 0, 255))
        masked[keep_mask] = frame[keep_mask]
        return masked.astype(np.uint8, copy=False)

    def _apply_keep_mask_to_video_frames(self, frames: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        masked = np.full_like(frames, np.clip(self.image_context_mask_fill_value, 0, 255))
        masked[:, keep_mask] = frames[:, keep_mask]
        return masked.astype(np.uint8, copy=False)

    def _mask_image_context(self, frame: np.ndarray, sampled_mask: np.ndarray | None = None) -> np.ndarray:
        """Mask a first-frame image context while keeping a configured visible region."""
        h, w = frame.shape[:2]
        keep_ratio = self.image_context_keep_ratio

        if self.mask_pool_use_dynamic or self.image_context_mask_reference_path is not None:
            keep = self._get_active_keep_mask(h, w, sampled_mask=sampled_mask)
            return self._apply_keep_mask_to_frame(frame, keep)

        masked = np.full_like(frame, np.clip(self.image_context_mask_fill_value, 0, 255))

        if self.image_context_mask_mode == "fixed_vertical_band":
            keep_w = max(1, int(round(w * keep_ratio)))
            x0 = max(0, (w - keep_w) // 2)
            masked[:, x0 : x0 + keep_w] = frame[:, x0 : x0 + keep_w]
        elif self.image_context_mask_mode == "fixed_horizontal_band":
            keep_h = max(1, int(round(h * keep_ratio)))
            y0 = max(0, (h - keep_h) // 2)
            masked[y0 : y0 + keep_h, :] = frame[y0 : y0 + keep_h, :]
        elif self.image_context_mask_mode == "fixed_center":
            side_ratio = keep_ratio**0.5
            keep_h = max(1, int(round(h * side_ratio)))
            keep_w = max(1, int(round(w * side_ratio)))
            y0 = max(0, (h - keep_h) // 2)
            x0 = max(0, (w - keep_w) // 2)
            masked[y0 : y0 + keep_h, x0 : x0 + keep_w] = frame[y0 : y0 + keep_h, x0 : x0 + keep_w]
        elif self.image_context_mask_mode == "random_pixels":
            keep = np.random.random((h, w)) < keep_ratio
            masked[keep] = frame[keep]
        elif self.image_context_mask_mode in {"waymo", "front_rear_luma_erp"}:
            keep = self._get_waymo_keep_mask(h, w)
            masked[keep] = frame[keep]
        elif self.image_context_mask_mode == "random_rectangles":
            keep = np.zeros((h, w), dtype=bool)
            target_pixels = int(round(h * w * keep_ratio))
            attempts = 0
            while keep.sum() < target_pixels and attempts < 64:
                rect_h = np.random.randint(max(1, h // 8), max(2, h // 2))
                rect_w = np.random.randint(max(1, w // 8), max(2, w // 2))
                y0 = np.random.randint(0, max(1, h - rect_h + 1))
                x0 = np.random.randint(0, max(1, w - rect_w + 1))
                keep[y0 : y0 + rect_h, x0 : x0 + rect_w] = True
                attempts += 1
            if keep.sum() > target_pixels:
                ys, xs = np.where(keep)
                drop_count = int(keep.sum() - target_pixels)
                if drop_count > 0:
                    drop_idx = np.random.choice(len(ys), size=drop_count, replace=False)
                    keep[ys[drop_idx], xs[drop_idx]] = False
            masked[keep] = frame[keep]

        return masked.astype(np.uint8, copy=False)

    def _get_waymo_keep_mask(self, height: int, width: int) -> np.ndarray:
        """Fixed Waymo ERP visibility mask derived exactly from the reference mask image."""
        packed = zlib.decompress(base64.b64decode(WAYMO_MASK_PACKED_ZLIB_B64))
        keep = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        keep = keep[: WAYMO_MASK_SHAPE[0] * WAYMO_MASK_SHAPE[1]].reshape(WAYMO_MASK_SHAPE).astype(bool)
        if keep.shape != (height, width):
            keep_img = Image.fromarray(keep.astype(np.uint8) * 255)
            keep_img = keep_img.resize((width, height), Image.Resampling.NEAREST)
            keep = np.asarray(keep_img) > 0
        return keep

    def _get_reference_keep_mask(self, height: int, width: int) -> np.ndarray:
        """Return a cached keep mask from a non-black reference image resized to the frame size."""
        if self._image_context_reference_keep_mask is not None:
            if self._image_context_reference_keep_mask.shape == (height, width):
                return self._image_context_reference_keep_mask

        assert self.image_context_mask_reference_path is not None
        ref = Image.open(self.image_context_mask_reference_path).convert("RGB")
        if ref.size != (width, height):
            ref = ref.resize((width, height), Image.Resampling.NEAREST)
        ref_np = np.asarray(ref)
        threshold = np.clip(self.image_context_mask_reference_threshold, 0, 255)
        keep = (ref_np > threshold).any(axis=2)
        self._image_context_reference_keep_mask = keep
        return keep

    def _load_caption(self, video_name: str) -> str:
        """Load caption from JSON or text file.

        Args:
            video_name: Video name without extension

        Returns:
            Caption text
        """
        # Try JSON first (Transfer2 format)
        json_path = Path(self.caption_dir) / f"{video_name}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Support various JSON formats
                    if isinstance(data, dict):
                        return data.get("caption", data.get("text", data.get("prompt", "")))
                    return str(data)
            except Exception as e:
                log.warning(f"Failed to load caption from {json_path}: {e}")

        # Fall back to text file (Transfer1 format)
        txt_path = Path(self.caption_dir) / f"{video_name}.txt"
        if txt_path.exists():
            try:
                return txt_path.read_text().strip()
            except Exception as e:
                log.warning(f"Failed to load caption from {txt_path}: {e}")

        log.debug(f"No caption found for {video_name}, using generic caption")
        return "a video"  # Generic fallback caption

    # Captions are now encoded on-the-fly by the model's text encoder (Qwen/reason1p1_7B)

    def _load_control_data(
        self,
        video_name: str,
        frame_ids: list[int],
        sampled_mask: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Load control input data (depth, segmentation, etc.).

        For edge/vis, returns None (computed on-the-fly by augmentor).
        For depth, loads video frames.
        For seg/keypoint, loads pickle data.

        Args:
            video_name: Video name without extension
            frame_ids: Frame indices to load

        Returns:
            Dictionary with control data or None if computed on-the-fly
        """
        # Edge and vis are computed on-the-fly by the augmentor
        if self.ctrl_config["folder"] is None:
            return None

        ctrl_folder = os.path.join(self.dataset_dir, self.ctrl_config["folder"])
        if self.control_video_dir_override is not None:
            ctrl_folder = self.control_video_dir_override
        ctrl_format = self.ctrl_config["format"]
        if self.control_video_dir_override is not None and self.ctrl_type == "depth":
            ctrl_filename = f"{video_name}{self.control_video_suffix}.{ctrl_format}"
        else:
            ctrl_filename = f"{video_name}.{ctrl_format}"
        ctrl_path = os.path.join(ctrl_folder, ctrl_filename)

        if not os.path.exists(ctrl_path):
            raise FileNotFoundError(f"Control input file not found: {ctrl_path}")

        data_dict = {}

        try:
            if self.ctrl_type == "seg":
                # Load segmentation video (same format as depth)
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Seg video has fewer frames than RGB video: {ctrl_path}")

                seg_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                seg_frames = seg_frames.astype(np.uint8)
                # Convert to tensor - augmentor will handle resizing to match video
                seg_t = torch.from_numpy(seg_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                seg_video = seg_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                # Store with the key expected by AddControlInputSeg augmentor
                data_dict["segmentation"] = seg_video
                del vr

            elif self.ctrl_type == "keypoint":
                # Load keypoint pickle
                with open(ctrl_path, "rb") as f:
                    keypoint_data = pickle.load(f)
                data_dict["keypoint"] = keypoint_data

            elif self.ctrl_type == "depth":
                # Load depth video
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Depth video has fewer frames than RGB video: {ctrl_path}")

                depth_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                depth_frames = depth_frames.astype(np.uint8)
                if self.mask_depth_control:
                    keep = self._get_active_keep_mask(
                        depth_frames.shape[1],
                        depth_frames.shape[2],
                        sampled_mask=sampled_mask,
                    )
                    depth_frames = self._apply_keep_mask_to_video_frames(depth_frames, keep)
                # Convert to tensor - augmentor will handle resizing to match video
                depth_t = torch.from_numpy(depth_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                depth_video = depth_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                # Store with the key expected by AddControlInputDepth augmentor
                data_dict["depth"] = depth_video
                del vr

        except Exception as e:
            log.warning(f"Failed to load control data from {ctrl_path}: {e}")
            return None

        return data_dict

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single training sample with full augmentation.

        Returns:
            Dictionary with:
                - video: RGB video tensor [C=3, T, H, W] dtype=uint8, resized and padded by augmentor
                - control_input_<type>: Control input tensor [C=3, T, H, W] dtype=uint8, same shape as video
                  (generated with randomized parameters if is_train=True)
                - fps: Video frame rate (float)
                - aspect_ratio: Aspect ratio string (e.g., "16:9")
                - image_size: Image dimensions [H, W, H, W] tensor (after augmentation)
                - padding_mask: Padding mask [1, H, W] tensor (marks valid vs. padded regions)
                - num_frames: Number of frames (int)
                - chunk_index: Chunk index (int, typically 0)
                - __url__: Dataset directory (str)
                - __key__: Video name (str)

        Note:
            - Augmentor applies randomized edge detection thresholds during training for diversity
            - Video dimensions may differ from input video_size due to augmentor's padding/resizing
            - Model expects uint8 format and performs normalization internally:
              uint8 [0, 255] → float32 [-1, 1]
        """
        max_retries = 10  # Try up to 10 different videos
        original_index = index

        for retry in range(max_retries):
            # Skip known bad videos
            if index in self.bad_video_indices:
                index = (index + 1) % len(self.video_paths)
                continue

            try:
                video_path = self.video_paths[index]
                video_name = os.path.basename(video_path).replace(".mp4", "")

                # Load video frames
                frames, fps, frame_ids = self._load_video(video_path)
                frames = frames.astype(np.uint8)

                # Convert to tensor - augmentor will handle resizing and padding
                # frames: numpy (T, H, W, C) uint8
                frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                # Permute to (C, T, H, W) format expected by augmentors
                video = frames_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8
                aspect_ratio = detect_aspect_ratio((video.shape[3], video.shape[2]))  # (W, H)

                # Build data dictionary
                data = {
                    "video": video,
                    "aspect_ratio": aspect_ratio,
                    "fps": fps,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1] + 1,
                    "num_frames": self.sequence_length,
                    "chunk_index": 0,
                    "frame_indices": frame_ids,
                    "n_orig_video_frames": len(frame_ids),
                }

                sampled_mask = None
                if self.mask_pool_use_dynamic:
                    sampled_mask, _ = self._sample_dynamic_mask()
                elif self._static_mask_pool_paths:
                    sampled_mask, _ = self._sample_static_pool_mask()

                # Optional: load separate input video (e.g., PCD) aligned to target frames
                if self.input_video_dir is not None:
                    input_video_path = self.input_video_paths.get(video_name)
                    if input_video_path is None or not os.path.exists(input_video_path):
                        raise FileNotFoundError(
                            f"Input video not found for '{video_name}' at {input_video_path}"
                        )
                    input_frames, _, _ = self._load_video(input_video_path, frame_ids=frame_ids)
                    input_frames = input_frames.astype(np.uint8)
                    if self.mask_depth_control and self.ctrl_type == "depth":
                        keep = self._get_active_keep_mask(
                            input_frames.shape[1],
                            input_frames.shape[2],
                            sampled_mask=sampled_mask,
                        )
                        input_frames = self._apply_keep_mask_to_video_frames(input_frames, keep)
                    input_frames_t = torch.from_numpy(input_frames).permute(0, 3, 1, 2)  # (T, C, H, W)
                    data["input_video"] = input_frames_t.permute(1, 0, 2, 3)  # (C, T, H, W)

                # Optional: on-the-fly image context from first RGB frame
                if self.use_image_context and self.image_context_from_rgb_first_frame:
                    first_frame = self._load_first_frame(video_path)
                    if self.mask_image_context:
                        first_frame = self._mask_image_context(first_frame, sampled_mask=sampled_mask)
                    data["image_context"] = torch.from_numpy(first_frame).permute(2, 0, 1).to(torch.uint8)

                # Load caption
                caption = self._load_caption(video_name)
                data[self.caption_type] = caption

                # Create metadata structure for augmentor compatibility
                # The augmentor expects "metas" with window information
                # Map caption_type to the expected caption key in window_data
                if self.caption_type == "t2w_qwen2p5_7b":
                    caption_key_in_window = "qwen2p5_7b_caption"  # gitleaks:allow
                else:
                    caption_key_in_window = self.caption_type

                window_data = {
                    "start_frame": frame_ids[0],
                    "end_frame": frame_ids[-1] + 1,
                    caption_key_in_window: caption,
                }
                data["metas"] = {
                    "framerate": fps,
                    "nb_frames": len(frame_ids),
                    # Create a single window spanning the entire video segment
                    # Include both windows and t2w_windows for different caption types
                    "windows": [window_data],
                    "t2w_windows": [window_data],
                    "i2w_windows_later_frames": [window_data],
                }

                # Pass raw caption for on-the-fly encoding by model's text encoder
                # (Like multiview dataset - model will encode with Qwen/reason1 encoder)
                data["ai_caption"] = caption

                # Add URL and key for logging (used by augmentors and training)
                # Use MockUrl object for augmentor compatibility (augmentors expect __url__.meta.opts)
                data["__url__"] = MockUrl(str(self.dataset_dir))
                data["__key__"] = video_name

                # Load control input data (if pre-computed)
                ctrl_data = self._load_control_data(video_name, frame_ids, sampled_mask=sampled_mask)
                if ctrl_data is not None:
                    data.update(ctrl_data)

                # Apply augmentation pipeline
                # This includes: resizing, padding, text transform, and control input generation
                # The augmentor will handle edge detection with randomized thresholds for training
                for aug_name, aug_fn in self.augmentor.items():
                    result = aug_fn(data)
                    # Check if augmentor returned None (e.g., filtering)
                    if result is None:
                        raise ValueError(f"Augmentor {aug_name} filtered out the sample")
                    data = result

                # Ensure input_video matches augmented video spatial size if provided.
                if "input_video" in data:
                    _, _, h, w = data["video"].shape
                    input_video = data["input_video"]
                    if input_video.shape[-2:] != (h, w):
                        input_video_tchw = input_video.permute(1, 0, 2, 3).float()
                        resized = F.interpolate(
                            input_video_tchw,
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        data["input_video"] = resized.permute(1, 0, 2, 3).clamp(0, 255).to(torch.uint8)

                # Convert MockUrl back to string for DataLoader collate compatibility
                # (PyTorch's collate function can't handle custom objects)
                if isinstance(data.get("__url__"), MockUrl):
                    data["__url__"] = str(data["__url__"])

                # Add final metadata (after augmentation)
                c, t, h, w = data["video"].shape
                if "image_size" not in data:
                    data["image_size"] = torch.tensor([h, w, h, w])
                if "padding_mask" not in data:
                    data["padding_mask"] = torch.ones(1, h, w)  # All valid (no padding)

                # Validate output format after augmentation
                assert data["video"].dtype == torch.uint8, f"Video dtype is {data['video'].dtype}, expected uint8"
                assert data["video"].shape[0] == 3, f"Video should have 3 channels, got {data['video'].shape[0]}"
                assert data["video"].shape[1] == self.sequence_length, (
                    f"Video should have {self.sequence_length} frames, got {data['video'].shape[1]}"
                )

                # Check control input exists and has correct format
                ctrl_key = f"control_input_{self.ctrl_type}"
                assert ctrl_key in data, f"Control input key '{ctrl_key}' not found in data"
                assert data[ctrl_key].dtype == torch.uint8, (
                    f"Control input dtype is {data[ctrl_key].dtype}, expected uint8"
                )
                assert data[ctrl_key].shape == data["video"].shape, (
                    f"Control input shape {data[ctrl_key].shape} doesn't match video shape {data['video'].shape}"
                )

                log.debug(
                    f"Dataset sample ready: video={data['video'].shape} {data['video'].dtype}, "
                    f"{ctrl_key}={data[ctrl_key].shape} {data[ctrl_key].dtype}, "
                )

                return data

            except Exception as e:
                self.num_failed_loads += 1
                # Mark this video as bad so we skip it in the future
                self.bad_video_indices.add(index)

                log.warning(
                    f"Failed to load video {self.video_paths[index]} (index {index}): {e}. "
                    f"Marking as bad and trying next video. "
                    f"(attempt {retry + 1}/{max_retries}, total bad videos: {len(self.bad_video_indices)})",
                    rank0_only=False,
                )

                if retry == max_retries - 1:
                    log.error(
                        f"Failed to load data after {max_retries} attempts starting from index {original_index}. "
                        f"Total bad videos: {len(self.bad_video_indices)}/{len(self.video_paths)}"
                    )
                    raise RuntimeError(
                        f"Failed to load data after {max_retries} attempts. "
                        f"Original index: {original_index}, last tried: {video_path}"
                    )

                # Try the next video in sequence (wraps around at end)
                index = (index + 1) % len(self.video_paths)

        raise RuntimeError("Should not reach here")


if __name__ == "__main__":
    """
    Sanity check for the dataset.

    Usage:
        PYTHONPATH=. python cosmos_transfer2/_src/transfer2/datasets/local_datasets/singleview_dataset_mask.py
    """
    import sys

    # Example dataset with edge control (computed on-the-fly)
    dataset = SingleViewTransferDatasetMask(
        dataset_dir="datasets/hdvila",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,
        use_image_context=True,
        image_context_from_rgb_first_frame=True,
        mask_image_context=True,
    )

    log.info(f"Dataset: {dataset}")
    log.info(f"Number of videos: {len(dataset)}")

    # Test loading a few samples
    indices = [0] if len(dataset) > 0 else []
    for idx in indices:
        log.info(f"\nTesting sample {idx}:")
        try:
            data = dataset[idx]
            log.info(f"  Video shape: {data['video'].shape}")
            log.info(f"  Control input shape: {data['control_input_edge'].shape}")
            log.info(f"  Caption: {data.get('ai_caption', 'N/A')[:100]}...")
            log.info(f"  FPS: {data['fps']}")
            log.info(f"  Aspect ratio: {data['aspect_ratio']}")
            log.info("  ✅ Sample loaded successfully")
        except Exception as e:
            log.error(f"  ❌ Failed to load sample: {e}")
            sys.exit(1)

    log.info("\n✅ All tests passed!")
