"""Compute the 8 stroke metrics from landmarks + segmented strokes.

Per-stroke raw values are computed first, then aggregated (mean over strokes,
plus rhythm consistency across strokes) and graded against the shared targets.

All metrics are INTRINSIC to the rower's own stroke (no external reference),
which is what lets us turn them into specific, actionable cues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .config import MetricTarget
from .geometry import Facing
from .segmentation import Stroke

# Sequencing compares when each joint reaches the halfway point of its own drive
# range. Good ordering: legs reach 50% before the back, and the back before the
# arms. A negative offset means that joint led the legs (the classic fault).
_SEQ_PROGRESS = 0.5


def _pt(landmarks: np.ndarray, frame: int, idx: int) -> np.ndarray:
    return landmarks[frame, idx, :]


def _knee_angle(landmarks, frame, side) -> float:
    return geometry.joint_angle(_pt(landmarks, frame, side["hip"]),
                                _pt(landmarks, frame, side["knee"]),
                                _pt(landmarks, frame, side["ankle"]))


def _elbow_angle(landmarks, frame, side) -> float:
    return geometry.joint_angle(_pt(landmarks, frame, side["shoulder"]),
                                _pt(landmarks, frame, side["elbow"]),
                                _pt(landmarks, frame, side["wrist"]))


def _torso_lean(landmarks, frame, side, facing) -> float:
    return geometry.signed_lean_from_vertical(_pt(landmarks, frame, side["hip"]),
                                              _pt(landmarks, frame, side["shoulder"]),
                                              facing)


def _progress(values: np.ndarray, rising: bool = True) -> np.ndarray:
    """Min-max normalize a drive trajectory to 0..1 (0 at catch, 1 at finish)."""
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    rng = vmax - vmin
    if rng <= 1e-6:
        return np.zeros_like(values)
    p = (values - vmin) / rng
    return p if rising else 1.0 - p


def _first_cross(values: np.ndarray, threshold: float, rising: bool) -> int | None:
    """Index of first value crossing threshold (rising or falling); None if never."""
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if (rising and v >= threshold) or (not rising and v <= threshold):
            return i
    return None


@dataclass
class StrokeMetrics:
    catch_shin_angle: float = float("nan")
    catch_back_angle: float = float("nan")
    legs_back_sequence: float = float("nan")   # ms
    arm_break_timing: float = float("nan")     # ms
    finish_layback_angle: float = float("nan")
    drive_recovery_ratio: float = float("nan")
    stroke_rate: float = float("nan")


def compute_stroke(landmarks: np.ndarray, stroke: Stroke, facing: Facing) -> StrokeMetrics:
    side = geometry.side_indices(facing)
    fps = stroke.fps
    drive = list(stroke.drive_frames)
    m = StrokeMetrics()

    # --- catch-frame angles ---
    c = stroke.catch
    m.catch_shin_angle = geometry.angle_from_vertical(
        _pt(landmarks, c, side["ankle"]), _pt(landmarks, c, side["knee"]))
    m.catch_back_angle = _torso_lean(landmarks, c, side, facing)  # +forward

    # --- finish-frame layback (degrees past vertical = backward lean) ---
    f = stroke.finish
    m.finish_layback_angle = -_torso_lean(landmarks, f, side, facing)

    # --- drive-phase sequences (50%-progress crossings) ---
    if len(drive) >= 3:
        knee = np.array([_knee_angle(landmarks, fr, side) for fr in drive])
        elbow = np.array([_elbow_angle(landmarks, fr, side) for fr in drive])
        lean = np.array([_torso_lean(landmarks, fr, side, facing) for fr in drive])

        # Normalized 0..1 progress of each joint over the drive.
        leg_prog = _progress(knee, rising=True)      # knee straightens (increases)
        back_prog = _progress(-lean, rising=True)    # torso swings back (lean falls)
        arm_prog = _progress(-elbow, rising=True)    # elbow bends (decreases)

        t_legs = _first_cross(leg_prog, _SEQ_PROGRESS, rising=True)
        t_back = _first_cross(back_prog, _SEQ_PROGRESS, rising=True)
        t_arm = _first_cross(arm_prog, _SEQ_PROGRESS, rising=True)

        ms_per_frame = 1000.0 / fps if fps else 0.0
        if t_legs is not None and t_back is not None:
            m.legs_back_sequence = (t_back - t_legs) * ms_per_frame
        if t_legs is not None and t_arm is not None:
            m.arm_break_timing = (t_arm - t_legs) * ms_per_frame

    # --- timing metrics ---
    if stroke.drive_s > 0:
        m.drive_recovery_ratio = stroke.recovery_s / stroke.drive_s
    m.stroke_rate = stroke.spm
    return m


@dataclass
class GradedMetric:
    key: str
    label: str
    value: float
    grade: str
    cue: str
    unit: str


@dataclass
class MetricsReport:
    per_stroke: list[StrokeMetrics]
    aggregate: dict[str, GradedMetric]
    n_strokes: int
    facing: Facing
    notes: list[str] = field(default_factory=list)


def _nanmean(vals: list[float]) -> float:
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def aggregate(per_stroke: list[StrokeMetrics], targets: dict[str, MetricTarget],
              facing: Facing) -> MetricsReport:
    """Average per-stroke metrics, add rhythm consistency, and grade everything."""
    keys = ["catch_shin_angle", "catch_back_angle", "legs_back_sequence",
            "arm_break_timing", "finish_layback_angle", "drive_recovery_ratio",
            "stroke_rate"]
    means = {k: _nanmean([getattr(s, k) for s in per_stroke]) for k in keys}

    # rhythm consistency: CV of stroke period across strokes
    periods = [s.stroke_rate for s in per_stroke]  # spm proxy; use 60/spm for period
    period_s = [60.0 / p for p in periods if p and not np.isnan(p)]
    if len(period_s) >= 2:
        arr = np.array(period_s)
        means["rhythm_consistency"] = float(arr.std() / arr.mean()) if arr.mean() else float("nan")
    else:
        means["rhythm_consistency"] = float("nan")

    graded: dict[str, GradedMetric] = {}
    for key, value in means.items():
        t = targets[key]
        if np.isnan(value):
            graded[key] = GradedMetric(key, t.label, value, "unknown", "", t.unit)
        else:
            graded[key] = GradedMetric(key, t.label, value, t.grade(value),
                                       t.cue_for(value), t.unit)

    return MetricsReport(per_stroke=per_stroke, aggregate=graded,
                         n_strokes=len(per_stroke), facing=facing)
