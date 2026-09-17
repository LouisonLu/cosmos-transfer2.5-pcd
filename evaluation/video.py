"""Bounded video readers shared by pixel-level evaluation plugins."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled by plugin availability
    cv2 = None


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    reported_frames: int


def has_opencv() -> bool:
    return cv2 is not None


def video_dependency_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__}
    if cv2 is not None:
        versions["opencv-python"] = cv2.__version__
    return versions


@contextmanager
def open_video(path: Path) -> Iterator[tuple[object, VideoInfo]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; install opencv-python for video metrics")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty video: {path}")
    capture = cv2.VideoCapture(str(path))
    info = VideoInfo(
        path=path,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        reported_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.width < 2 or info.height < 1 or info.fps <= 0 or info.reported_frames < 1:
        capture.release()
        raise ValueError(f"Cannot read video properties: {path}")
    try:
        yield capture, info
    finally:
        capture.release()


def _read_rgb(capture: object, path: Path, frame_index: int) -> np.ndarray:
    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def frame_count_from_policy(policy: dict[str, object]) -> int:
    frame_count = int(policy.get("frame_count", 93))
    if frame_count < 1:
        raise ValueError("evaluation.frame_count must be positive")
    return frame_count


def require_exact_match(first: VideoInfo, second: VideoInfo, label: str) -> None:
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError(
            f"{label} requires exact spatial match, got "
            f"{first.path}={first.width}x{first.height}, {second.path}={second.width}x{second.height}"
        )


def iter_rgb_pairs(prediction: Path, reference: Path, policy: dict[str, object]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    requested = frame_count_from_policy(policy)
    with open_video(prediction) as (pred_capture, pred_info), open_video(reference) as (ref_capture, ref_info):
        require_exact_match(pred_info, ref_info, "Reference metric")
        if pred_info.reported_frames < requested or ref_info.reported_frames < requested:
            raise ValueError(
                f"Reference metric requires {requested} frames, got "
                f"prediction={pred_info.reported_frames}, reference={ref_info.reported_frames}"
            )
        for frame_index in range(requested):
            yield frame_index, _read_rgb(pred_capture, prediction, frame_index), _read_rgb(ref_capture, reference, frame_index)


def iter_prediction_frames(prediction: Path, policy: dict[str, object]) -> Iterator[tuple[int, np.ndarray]]:
    requested = frame_count_from_policy(policy)
    with open_video(prediction) as (capture, info):
        if info.reported_frames < requested:
            raise ValueError(f"Metric requires {requested} frames, got {info.reported_frames}: {prediction}")
        for frame_index in range(requested):
            yield frame_index, _read_rgb(capture, prediction, frame_index)


def iter_prediction_input_mask(
    prediction: Path, input_rgb: Path, mask: Path, policy: dict[str, object]
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield first-frame RGB/mask tuples for known-region preservation metrics."""
    with open_video(prediction) as (pred_capture, pred_info), open_video(input_rgb) as (input_capture, input_info), open_video(
        mask
    ) as (mask_capture, mask_info):
        require_exact_match(pred_info, input_info, "Known-region metric")
        require_exact_match(pred_info, mask_info, "Known-region metric")
        if min(pred_info.reported_frames, input_info.reported_frames, mask_info.reported_frames) < 1:
            raise ValueError("Known-region metric needs frame 0 in prediction, input RGB, and mask")
        yield 0, _read_rgb(pred_capture, prediction, 0), _read_rgb(input_capture, input_rgb, 0), _read_rgb(mask_capture, mask, 0)
