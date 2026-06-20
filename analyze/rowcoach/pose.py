"""MediaPipe Pose wrapper.

Extracts 33 body landmarks per frame as an array of (x, y, visibility), plus a
per-frame mean confidence used for footage-quality warnings.

Coordinate convention: x, y are normalized to [0, 1] in image space, with y
increasing DOWNWARD (standard image coords). Downstream geometry accounts for this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .video import VideoInfo, iter_frames

# Landmark indices we care about for a side-view erg stroke (MediaPipe Pose).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

NUM_LANDMARKS = 33


@dataclass
class PoseSequence:
    # [n_frames, 33, 3] -> (x, y, visibility)
    landmarks: np.ndarray
    # [n_frames] mean visibility of the body landmarks (0..1)
    frame_confidence: np.ndarray
    fps: float

    @property
    def n_frames(self) -> int:
        return self.landmarks.shape[0]


def extract(info: VideoInfo) -> PoseSequence:
    """Run MediaPipe Pose (Tasks API, VIDEO mode) over every frame of the clip."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    from .config import ensure_pose_model

    model_path = ensure_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )

    frames: list[np.ndarray] = []
    confs: list[float] = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for idx, ts, rgb in iter_frames(info):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # timestamps must be monotonically increasing integer ms
            ts_ms = int(round(ts * 1000)) if info.fps else idx
            result = landmarker.detect_for_video(mp_image, ts_ms)
            frame = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            if result.pose_landmarks:
                for i, lm in enumerate(result.pose_landmarks[0]):
                    frame[i] = (lm.x, lm.y, lm.visibility)
                confs.append(float(frame[:, 2].mean()))
            else:
                confs.append(0.0)
            frames.append(frame)

    landmarks = (np.stack(frames, axis=0) if frames
                 else np.zeros((0, NUM_LANDMARKS, 3), dtype=np.float32))
    return PoseSequence(
        landmarks=landmarks,
        frame_confidence=np.asarray(confs, dtype=np.float32),
        fps=info.fps,
    )
