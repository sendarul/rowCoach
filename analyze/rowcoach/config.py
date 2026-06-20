"""Configuration + the shared stroke-target definitions.

Targets live in ../../shared/stroke_targets.json so Tier 1 (PWA) and Tier 2
(this package) grade strokes identically. Everything tunable lives there or here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# analyze/rowcoach/config.py -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = _REPO_ROOT / "shared" / "stroke_targets.json"

# MediaPipe Tasks pose model (auto-downloaded on first run if missing).
_ANALYZE_ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL_PATH = _ANALYZE_ROOT / "models" / "pose_landmarker_full.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def ensure_pose_model(path: Path = POSE_MODEL_PATH) -> Path:
    """Download the pose model on first use if it isn't already present."""
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model -> {path} …")
    urllib.request.urlretrieve(POSE_MODEL_URL, path)
    return path

Grade = Literal["good", "minor", "needs_work"]

# --- Tunable analysis constants (Tier 2 only) -------------------------------

# Plausible stroke-rate window, used to reject noise in segmentation.
MIN_SPM = 16
MAX_SPM = 40

# A frame is "low confidence" below this mean landmark visibility (0..1).
MIN_FRAME_CONFIDENCE = 0.4
# If more than this fraction of frames are low-confidence, warn about footage.
MAX_LOW_CONF_FRACTION = 0.25

# Savitzky-Golay smoothing window (frames) for landmark trajectories. Odd number.
SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2

# Shortest clip we'll attempt to segment.
MIN_CLIP_SECONDS = 5.0

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass(frozen=True)
class MetricTarget:
    key: str
    label: str
    describe: str
    unit: str
    target_min: float
    target_max: float
    minor_tolerance: float
    cue_low: str
    cue_high: str
    priority: int

    def grade(self, value: float) -> Grade:
        """Grade a measured value against this metric's target window."""
        if self.target_min <= value <= self.target_max:
            return "good"
        if value < self.target_min:
            dist = self.target_min - value
        else:
            dist = value - self.target_max
        return "minor" if dist <= self.minor_tolerance else "needs_work"

    def cue_for(self, value: float) -> str:
        """The short corrective cue for an out-of-range value ('' if in range)."""
        if value < self.target_min:
            return self.cue_low
        if value > self.target_max:
            return self.cue_high
        return ""


def load_targets(path: Path = TARGETS_PATH) -> dict[str, MetricTarget]:
    """Load the shared metric targets keyed by metric name."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, MetricTarget] = {}
    for key, m in data["metrics"].items():
        out[key] = MetricTarget(
            key=key,
            label=m["label"],
            describe=m["describe"],
            unit=m["unit"],
            target_min=float(m["target_min"]),
            target_max=float(m["target_max"]),
            minor_tolerance=float(m["minor_tolerance"]),
            cue_low=m.get("cue_low", ""),
            cue_high=m.get("cue_high", ""),
            priority=int(m["priority"]),
        )
    return out
