"""Video loading + normalization.

Handles iPhone clips (.mov/.mp4), corrects rotation metadata that OpenCV
otherwise ignores, and exposes frames as upright RGB arrays with timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int          # upright (after rotation) dimensions
    height: int
    rotation: int       # degrees applied to make frames upright (0/90/180/270)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


def _read_rotation(cap: cv2.VideoCapture) -> int:
    """Read rotation metadata (degrees). iPhone clips carry this; OpenCV applies
    it on read only for some builds, so we track and apply it ourselves."""
    try:
        # CAP_PROP_ORIENTATION_META exists on newer OpenCV (>=4.5 w/ FFmpeg).
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        if meta and not np.isnan(meta):
            return int(meta) % 360
    except Exception:
        pass
    return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def probe(path: str | Path) -> VideoInfo:
    """Open a clip and return its (rotation-corrected) properties."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video (unsupported codec?): {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        rotation = _read_rotation(cap)
        # If rotated 90/270, upright dimensions are swapped.
        if rotation in (90, 270):
            width, height = raw_h, raw_w
        else:
            width, height = raw_w, raw_h
    finally:
        cap.release()

    return VideoInfo(
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        rotation=rotation,
    )


def iter_frames(info: VideoInfo) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, timestamp_seconds, rgb_frame) for each upright frame."""
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {info.path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _apply_rotation(frame, info.rotation)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = idx / info.fps if info.fps else 0.0
            yield idx, ts, rgb
            idx += 1
    finally:
        cap.release()
