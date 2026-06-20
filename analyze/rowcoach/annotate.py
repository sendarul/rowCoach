"""Annotated video: skeleton overlay + phase label + per-stroke HUD + fault callouts.

Re-renders the (rotation-corrected) clip with OpenCV. Fault callouts are drawn
during the drive of strokes whose sequencing graded poorly, at the moment the
fault is visible.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import geometry
from .geometry import Facing
from .metrics import StrokeMetrics
from .pose import PoseSequence
from .segmentation import Stroke
from .video import VideoInfo, _apply_rotation

# skeleton bone connections (camera-side + spine), using MediaPipe indices
from . import pose as _pose

_BONES_BY_SIDE = {
    "right": [(_pose.R_SHOULDER, _pose.R_ELBOW), (_pose.R_ELBOW, _pose.R_WRIST),
              (_pose.R_SHOULDER, _pose.R_HIP), (_pose.R_HIP, _pose.R_KNEE),
              (_pose.R_KNEE, _pose.R_ANKLE)],
    "left": [(_pose.L_SHOULDER, _pose.L_ELBOW), (_pose.L_ELBOW, _pose.L_WRIST),
             (_pose.L_SHOULDER, _pose.L_HIP), (_pose.L_HIP, _pose.L_KNEE),
             (_pose.L_KNEE, _pose.L_ANKLE)],
}


def _phase_at(frame: int, strokes: list[Stroke]) -> tuple[str, int]:
    """Return (phase_label, stroke_number_1based) for a frame."""
    for i, s in enumerate(strokes, 1):
        if s.catch <= frame <= s.finish:
            return "DRIVE", i
        if s.finish < frame <= s.next_catch:
            return "RECOVERY", i
    return "—", 0


def render(info: VideoInfo, seq: PoseSequence, strokes: list[Stroke],
           per_stroke: list[StrokeMetrics], facing: Facing,
           bad_sequence_strokes: set[int], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, info.fps or 30.0,
                             (info.width, info.height))
    bones = _BONES_BY_SIDE[facing]
    lm = seq.landmarks

    cap = cv2.VideoCapture(str(info.path))
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _apply_rotation(frame, info.rotation)
            h, w = frame.shape[:2]

            if idx < lm.shape[0]:
                pts = lm[idx]
                for a, b in bones:
                    if pts[a, 2] > 0.3 and pts[b, 2] > 0.3:
                        pa = (int(pts[a, 0] * w), int(pts[a, 1] * h))
                        pb = (int(pts[b, 0] * w), int(pts[b, 1] * h))
                        cv2.line(frame, pa, pb, (0, 255, 0), 2)
                        cv2.circle(frame, pa, 4, (0, 200, 255), -1)
                        cv2.circle(frame, pb, 4, (0, 200, 255), -1)

            phase, snum = _phase_at(idx, strokes)
            cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.putText(frame, f"stroke {snum}  |  {phase}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # fault callout during the drive of bad-sequencing strokes
            if phase == "DRIVE" and snum in bad_sequence_strokes:
                cv2.putText(frame, "LEGS FIRST!", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path
