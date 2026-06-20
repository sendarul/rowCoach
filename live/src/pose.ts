// MoveNet (TensorFlow.js) in-browser pose detector. Runs real-time on phones.
import * as poseDetection from "@tensorflow-models/pose-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

import type { KP } from "./geometry";

let detector: poseDetection.PoseDetector | null = null;

export async function loadDetector(): Promise<void> {
  await tf.setBackend("webgl");
  await tf.ready();
  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING },
  );
}

// Returns the 17 keypoints (pixel coords) for the single detected pose, or null.
export async function estimate(video: HTMLVideoElement): Promise<KP[] | null> {
  if (!detector) return null;
  const poses = await detector.estimatePoses(video, { flipHorizontal: false });
  if (!poses.length) return null;
  return poses[0].keypoints.map((k) => ({
    x: k.x,
    y: k.y,
    score: k.score ?? 0,
  }));
}
