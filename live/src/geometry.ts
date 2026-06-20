// Geometry helpers mirroring the Python Tier 2 conventions, but for MoveNet's
// 17 COCO keypoints in pixel coords (y DOWN). Angles convert to y-up internally.

export interface KP {
  x: number;
  y: number;
  score: number;
}

// MoveNet / COCO-17 keypoint indices
export const KEYPOINT = {
  nose: 0,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
} as const;

export type Facing = "left" | "right";

export interface SideIdx {
  shoulder: number;
  elbow: number;
  wrist: number;
  hip: number;
  knee: number;
  ankle: number;
}

export function sideIndices(facing: Facing): SideIdx {
  return facing === "left"
    ? {
        shoulder: KEYPOINT.leftShoulder,
        elbow: KEYPOINT.leftElbow,
        wrist: KEYPOINT.leftWrist,
        hip: KEYPOINT.leftHip,
        knee: KEYPOINT.leftKnee,
        ankle: KEYPOINT.leftAnkle,
      }
    : {
        shoulder: KEYPOINT.rightShoulder,
        elbow: KEYPOINT.rightElbow,
        wrist: KEYPOINT.rightWrist,
        hip: KEYPOINT.rightHip,
        knee: KEYPOINT.rightKnee,
        ankle: KEYPOINT.rightAnkle,
      };
}

const up = (p: KP): [number, number] => [p.x, -p.y];

export function jointAngle(a: KP, b: KP, c: KP): number {
  const [ax, ay] = up(a), [bx, by] = up(b), [cx, cy] = up(c);
  const bax = ax - bx, bay = ay - by;
  const bcx = cx - bx, bcy = cy - by;
  const denom = Math.hypot(bax, bay) * Math.hypot(bcx, bcy);
  if (denom === 0) return NaN;
  const cos = Math.max(-1, Math.min(1, (bax * bcx + bay * bcy) / denom));
  return (Math.acos(cos) * 180) / Math.PI;
}

// Angle of segment a->b away from vertical (0 = vertical), always >= 0.
export function angleFromVertical(a: KP, b: KP): number {
  const [ax, ay] = up(a), [bx, by] = up(b);
  const vx = bx - ax, vy = by - ay;
  const mag = Math.hypot(vx, vy);
  if (mag === 0) return NaN;
  const cos = Math.max(-1, Math.min(1, Math.abs(vy) / mag));
  return (Math.acos(cos) * 180) / Math.PI;
}

// Torso lean from vertical, signed: + = forward (toward the catch), - = back.
export function signedLean(hip: KP, shoulder: KP, facing: Facing): number {
  const [hx, hy] = up(hip), [sx, sy] = up(shoulder);
  const vx = sx - hx, vy = sy - hy;
  const mag = Math.hypot(vx, vy);
  if (mag === 0) return NaN;
  const deg = (Math.acos(Math.max(-1, Math.min(1, vy / mag))) * 180) / Math.PI;
  const forward = facing === "right" ? vx >= 0 : vx <= 0;
  return forward ? deg : -deg;
}

// Decide facing from averaged nose-vs-hip horizontal offset (confidence-weighted).
export function detectFacing(frames: KP[][]): Facing {
  let num = 0, den = 0;
  for (const f of frames) {
    const nose = f[KEYPOINT.nose];
    const lh = f[KEYPOINT.leftHip], rh = f[KEYPOINT.rightHip];
    if (!nose || !lh || !rh) continue;
    const w = (lh.score + rh.score) / 2;
    const hipX = (lh.x + rh.x) / 2;
    num += (nose.x - hipX) * w;
    den += w;
  }
  if (den === 0) return "right";
  return num / den >= 0 ? "right" : "left";
}
