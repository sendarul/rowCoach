// Per-stroke metrics for the live tier. Mirrors Tier 2's definitions so both
// grade against the same shared targets. Uses real timestamps for sequencing.

import { type Facing, angleFromVertical, sideIndices } from "./geometry";
import type { StrokeData } from "./segmentation";

export interface StrokeMetrics {
  catch_shin_angle: number;
  catch_back_angle: number;
  legs_back_sequence: number; // ms
  arm_break_timing: number; // ms
  finish_layback_angle: number;
  drive_recovery_ratio: number;
  stroke_rate: number;
}

// Min-max normalize then return the interpolated time at which progress hits 0.5.
function crossTime(values: number[], times: number[], rising = true): number {
  const v = rising ? values : values.map((x) => -x);
  let lo = Infinity, hi = -Infinity;
  for (const x of v) {
    if (Number.isNaN(x)) continue;
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  const range = hi - lo;
  if (!(range > 1e-6)) return NaN;
  const target = lo + 0.5 * range;
  for (let i = 1; i < v.length; i++) {
    if (v[i - 1] <= target && v[i] >= target) {
      const f = (target - v[i - 1]) / (v[i] - v[i - 1] || 1);
      return times[i - 1] + f * (times[i] - times[i - 1]);
    }
  }
  return NaN;
}

export function computeStroke(s: StrokeData, facing: Facing): StrokeMetrics {
  const idx = sideIndices(facing);
  const c = s.catch.kp;

  const tLeg = crossTime(s.driveKnee, s.driveT, true); // knee straightens (rising)
  const tBack = crossTime(s.driveLean, s.driveT, false); // lean falls -> use -lean
  const tArm = crossTime(s.driveElbow, s.driveT, false); // elbow bends -> use -elbow

  return {
    catch_shin_angle: angleFromVertical(c[idx.ankle], c[idx.knee]),
    catch_back_angle: s.catch.lean,
    legs_back_sequence: Number.isNaN(tLeg) || Number.isNaN(tBack) ? NaN : tBack - tLeg,
    arm_break_timing: Number.isNaN(tLeg) || Number.isNaN(tArm) ? NaN : tArm - tLeg,
    finish_layback_angle: -s.finish.lean,
    drive_recovery_ratio: s.driveMs > 0 ? s.recoveryMs / s.driveMs : NaN,
    stroke_rate: s.periodMs > 0 ? 60000 / s.periodMs : NaN,
  };
}
