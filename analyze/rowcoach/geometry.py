"""Geometry helpers: angles, side detection, and trajectory smoothing.

All inputs are landmark arrays from pose.py with image coordinates (y DOWN).
Angle helpers convert to a math-friendly frame (y UP) internally so that
"degrees from vertical" reads the way a human expects.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import savgol_filter

from . import pose
from .config import SMOOTH_POLYORDER, SMOOTH_WINDOW

Facing = Literal["left", "right"]


def smooth_landmarks(landmarks: np.ndarray,
                     window: int = SMOOTH_WINDOW,
                     polyorder: int = SMOOTH_POLYORDER) -> np.ndarray:
    """Savitzky-Golay smoothing of x,y trajectories over time (axis 0).

    Visibility channel is left untouched. No-op if the clip is shorter than the
    smoothing window.
    """
    n = landmarks.shape[0]
    if n < window or window < 3:
        return landmarks.copy()
    win = window if window % 2 == 1 else window + 1
    win = min(win, n if n % 2 == 1 else n - 1)
    out = landmarks.copy()
    out[:, :, 0] = savgol_filter(landmarks[:, :, 0], win, polyorder, axis=0)
    out[:, :, 1] = savgol_filter(landmarks[:, :, 1], win, polyorder, axis=0)
    return out


def detect_facing(landmarks: np.ndarray) -> Facing:
    """Decide whether the rower faces left or right in the frame.

    Heuristic: on an erg the handle/wrists travel toward the flywheel in front of
    the rower. The nose sits ahead of the hips horizontally in the facing
    direction. Average nose-vs-hip x over high-confidence frames.
    """
    nose_x = landmarks[:, pose.NOSE, 0]
    hip_x = (landmarks[:, pose.L_HIP, 0] + landmarks[:, pose.R_HIP, 0]) / 2.0
    # weight by hip visibility to ignore junk frames
    w = (landmarks[:, pose.L_HIP, 2] + landmarks[:, pose.R_HIP, 2]) / 2.0
    if w.sum() <= 0:
        return "right"
    delta = float(np.average(nose_x - hip_x, weights=w))
    # nose to the right of hips (larger x) -> facing right
    return "right" if delta >= 0 else "left"


def side_indices(facing: Facing) -> dict[str, int]:
    """Landmark indices for the camera-facing side of the body."""
    if facing == "left":
        return {
            "shoulder": pose.L_SHOULDER, "elbow": pose.L_ELBOW,
            "wrist": pose.L_WRIST, "hip": pose.L_HIP,
            "knee": pose.L_KNEE, "ankle": pose.L_ANKLE,
        }
    return {
        "shoulder": pose.R_SHOULDER, "elbow": pose.R_ELBOW,
        "wrist": pose.R_WRIST, "hip": pose.R_HIP,
        "knee": pose.R_KNEE, "ankle": pose.R_ANKLE,
    }


def _to_xy_up(p: np.ndarray) -> np.ndarray:
    """Convert an [..., (x,y,...)] point from image coords (y down) to y-up."""
    q = p[..., :2].astype(np.float64).copy()
    q[..., 1] = -q[..., 1]
    return q


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle (degrees) at vertex b formed by points a-b-c."""
    a, b, c = _to_xy_up(a), _to_xy_up(b), _to_xy_up(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def angle_from_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (degrees) of segment a->b away from vertical (0 = vertical).

    Always non-negative; pair with a sign helper when lean direction matters.
    """
    a, b = _to_xy_up(a), _to_xy_up(b)
    v = b - a
    if np.linalg.norm(v) == 0:
        return float("nan")
    # angle vs the +y (up) axis
    cosang = np.clip(v[1] / np.linalg.norm(v), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cosang))))


def signed_lean_from_vertical(hip: np.ndarray, shoulder: np.ndarray,
                              facing: Facing) -> float:
    """Torso lean from vertical in degrees, signed so that:

      positive  = leaning toward the catch (forward / shoulders ahead of hips)
      negative  = leaning toward the finish (back / shoulders behind hips)

    'Forward' is the facing direction. Used for both catch lean and finish layback.
    """
    h, s = _to_xy_up(hip), _to_xy_up(shoulder)
    v = s - h
    if np.linalg.norm(v) == 0:
        return float("nan")
    mag = np.degrees(np.arccos(np.clip(v[1] / np.linalg.norm(v), -1.0, 1.0)))
    # horizontal component: shoulder ahead of hip in facing direction => forward
    dx = v[0]
    forward = dx >= 0 if facing == "right" else dx <= 0
    return float(mag if forward else -mag)
