"""Stroke segmentation.

Builds a 1D "reach" signal from the camera-side wrist (how far the hands are
extended toward the catch), then finds catches (reach maxima) and finishes
(reach minima between catches). Each stroke = catch -> finish (drive) ->
next catch (recovery). This intrinsic, per-rower segmentation is also our
"alignment" -- we never compare against an external reference clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from . import geometry
from .config import MAX_SPM, MIN_SPM
from .geometry import Facing


@dataclass
class Stroke:
    catch: int          # frame index of the catch
    finish: int         # frame index of the finish
    next_catch: int     # frame index of the following catch
    fps: float

    @property
    def drive_frames(self) -> range:
        return range(self.catch, self.finish + 1)

    @property
    def recovery_frames(self) -> range:
        return range(self.finish, self.next_catch + 1)

    @property
    def drive_s(self) -> float:
        return (self.finish - self.catch) / self.fps if self.fps else 0.0

    @property
    def recovery_s(self) -> float:
        return (self.next_catch - self.finish) / self.fps if self.fps else 0.0

    @property
    def period_s(self) -> float:
        return (self.next_catch - self.catch) / self.fps if self.fps else 0.0

    @property
    def spm(self) -> float:
        return 60.0 / self.period_s if self.period_s else 0.0


def reach_signal(landmarks: np.ndarray, facing: Facing) -> np.ndarray:
    """Per-frame hand reach toward the catch (higher = more extended forward).

    Normalized so larger values always mean 'closer to the catch' regardless of
    which way the rower faces.
    """
    idx = geometry.side_indices(facing)
    wrist_x = landmarks[:, idx["wrist"], 0]
    return wrist_x if facing == "right" else (1.0 - wrist_x)


def _min_stroke_frames(fps: float) -> int:
    # fastest plausible stroke sets the minimum spacing between catches
    return max(1, int(round(fps * 60.0 / MAX_SPM)))


def segment(landmarks: np.ndarray, facing: Facing, fps: float) -> list[Stroke]:
    """Split a clip into strokes. Returns full strokes (each needs a following
    catch); partial leading/trailing strokes are excluded."""
    if landmarks.shape[0] < 3 or not fps:
        return []

    reach = reach_signal(landmarks, facing)
    # normalize to 0..1 for stable peak prominence
    rng = np.ptp(reach)
    norm = (reach - reach.min()) / rng if rng > 0 else reach * 0.0

    min_dist = _min_stroke_frames(fps)
    prominence = 0.10  # fraction of full reach range; tuned later on real clips

    catches, _ = find_peaks(norm, distance=min_dist, prominence=prominence)
    if len(catches) < 2:
        return []

    strokes: list[Stroke] = []
    for c0, c1 in zip(catches[:-1], catches[1:]):
        # finish = deepest reach minimum between consecutive catches
        seg = norm[c0:c1 + 1]
        if seg.size < 3:
            continue
        finish = c0 + int(np.argmin(seg))
        if finish <= c0 or finish >= c1:
            continue
        strokes.append(Stroke(catch=int(c0), finish=int(finish),
                              next_catch=int(c1), fps=fps))

    return _reject_outliers(strokes)


# Loose physiological bounds on a single stroke period (seconds). These only
# reject degenerate detections; real cadence is handled by the median band below.
_ABS_MIN_PERIOD = 0.8     # ~75 spm ceiling
_ABS_MAX_PERIOD = 12.0    # accommodates slow demo/teaching strokes


def _reject_outliers(strokes: list[Stroke]) -> list[Stroke]:
    """Keep strokes within loose absolute bounds AND near the median cadence.

    Adaptive to whatever pace the rower holds (slow demo or racing), while
    dropping spurious half- or double-length detections.
    """
    if not strokes:
        return []
    periods = np.array([s.period_s for s in strokes], dtype=float)
    med = float(np.median(periods[periods > 0])) if np.any(periods > 0) else 0.0
    out: list[Stroke] = []
    for s, p in zip(strokes, periods):
        if not (_ABS_MIN_PERIOD <= p <= _ABS_MAX_PERIOD):
            continue
        if med > 0 and not (0.45 * med <= p <= 2.2 * med):
            continue
        out.append(s)
    return out
