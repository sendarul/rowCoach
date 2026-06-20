// Synthetic validation of the online StrokeTracker + live metrics.
// Interpolates between a CATCH pose and a FINISH pose to drive a realistic
// reach oscillation, then checks stroke count, SPM, and metric sanity.
import type { KP } from "../src/geometry";
import { StrokeTracker } from "../src/segmentation";
import { computeStroke } from "../src/metrics";

// 17 COCO keypoints; only the ones we use need to be meaningful.
function pose(vals: Record<number, [number, number]>): KP[] {
  const kp: KP[] = [];
  for (let i = 0; i < 17; i++) {
    const v = vals[i] ?? [0, 0];
    kp.push({ x: v[0], y: v[1], score: 1 });
  }
  return kp;
}

// facing right: x increases toward the catch (front)
const CATCH = pose({
  0: [190, 220], 6: [180, 250], 8: [220, 255], 10: [300, 260],
  12: [150, 300], 14: [210, 270], 16: [250, 380],
});
const FINISH = pose({
  0: [140, 215], 6: [150, 240], 8: [120, 250], 10: [170, 255],
  12: [155, 300], 14: [215, 360], 16: [250, 380],
});

function lerp(a: KP[], b: KP[], t: number): KP[] {
  return a.map((p, i) => ({
    x: p.x + (b[i].x - p.x) * t,
    y: p.y + (b[i].y - p.y) * t,
    score: 1,
  }));
}

const fps = 30, periodFrames = 60, strokes = 6; // 30 spm
const driveFrac = 0.4;
const strokesOut: any[] = [];
const tracker = new StrokeTracker((s) => {
  strokesOut.push(computeStroke(s, tracker.facing!));
});

for (let f = 0; f < periodFrames * strokes; f++) {
  const phase = (f % periodFrames) / periodFrames; // 0=catch
  const dp = phase < driveFrac ? phase / driveFrac : 1 - (phase - driveFrac) / (1 - driveFrac);
  const kp = lerp(CATCH, FINISH, dp);
  tracker.push((f / fps) * 1000, kp);
}

console.log("facing:", tracker.facing);
console.log("strokes emitted:", strokesOut.length, "(expect ~4-5)");
for (const m of strokesOut) {
  console.log(
    `  spm=${m.stroke_rate.toFixed(1)} dr=${m.drive_recovery_ratio.toFixed(2)} ` +
    `shin=${m.catch_shin_angle.toFixed(0)} layback=${m.finish_layback_angle.toFixed(0)} ` +
    `legs_back=${m.legs_back_sequence.toFixed(0)}ms`,
  );
}
