// Real-time stroke segmentation: an online peak detector on the hand-reach
// signal. Emits a completed stroke (with the drive trajectories needed for
// sequencing) at each catch. No offline find_peaks — this runs live.

import {
  type Facing,
  type KP,
  detectFacing,
  jointAngle,
  signedLean,
  sideIndices,
} from "./geometry";

export interface Sample {
  t: number; // ms
  kp: KP[];
  reach: number;
  knee: number;
  lean: number;
  elbow: number;
}

export interface StrokeData {
  catch: Sample;
  finish: Sample;
  driveT: number[];
  driveKnee: number[];
  driveLean: number[];
  driveElbow: number[];
  driveMs: number;
  recoveryMs: number;
  periodMs: number;
}

export type Phase = "init" | "drive" | "recovery";

const WARMUP_FRAMES = 20; // collect before locking facing
const RANGE_WINDOW = 90; // ~3s at 30fps for adaptive range
const HYST_FRAC = 0.18; // peak hysteresis as a fraction of reach range
const MIN_RANGE = 8; // px; below this there's no real rowing motion

export class StrokeTracker {
  facing: Facing | null = null;
  phase: Phase = "init";
  private warmup: KP[][] = [];
  private recentReach: number[] = [];
  private driveSamples: Sample[] = [];
  private prevCatch: Sample | null = null;
  private curExtreme: Sample | null = null; // running max (recovery/init) or min (drive)
  private finishPos = 0; // index in driveSamples of the running min
  private lastFinish: Sample | null = null;
  private emitted = 0; // count of completed strokes

  constructor(private onStroke: (s: StrokeData) => void) {}

  get currentPhase(): Phase {
    return this.phase;
  }

  private reachOf(kp: KP[]): number {
    const idx = sideIndices(this.facing!);
    const wx = kp[idx.wrist].x;
    return this.facing === "right" ? wx : -wx;
  }

  private makeSample(t: number, kp: KP[]): Sample {
    const idx = sideIndices(this.facing!);
    return {
      t,
      kp,
      reach: this.reachOf(kp),
      knee: jointAngle(kp[idx.hip], kp[idx.knee], kp[idx.ankle]),
      lean: signedLean(kp[idx.hip], kp[idx.shoulder], this.facing!),
      elbow: jointAngle(kp[idx.shoulder], kp[idx.elbow], kp[idx.wrist]),
    };
  }

  private hyst(): number {
    if (this.recentReach.length < 5) return MIN_RANGE;
    const range = Math.max(...this.recentReach) - Math.min(...this.recentReach);
    return Math.max(MIN_RANGE, range * HYST_FRAC);
  }

  /** Feed one frame. */
  push(t: number, kp: KP[] | null): void {
    if (!kp) return;

    if (this.facing === null) {
      this.warmup.push(kp);
      if (this.warmup.length >= WARMUP_FRAMES) {
        this.facing = detectFacing(this.warmup);
      }
      return;
    }

    const s = this.makeSample(t, kp);
    this.recentReach.push(s.reach);
    if (this.recentReach.length > RANGE_WINDOW) this.recentReach.shift();
    const h = this.hyst();

    if (this.phase === "init") {
      if (!this.curExtreme || s.reach > this.curExtreme.reach) this.curExtreme = s;
      if (this.curExtreme && s.reach < this.curExtreme.reach - h) {
        this.prevCatch = this.curExtreme;
        this.phase = "drive";
        this.driveSamples = [this.curExtreme, s];
        this.curExtreme = s; // now track min
        this.finishPos = 1;
      }
      return;
    }

    if (this.phase === "drive") {
      this.driveSamples.push(s);
      if (s.reach < this.curExtreme!.reach) {
        this.curExtreme = s;
        this.finishPos = this.driveSamples.length - 1;
      }
      if (s.reach > this.curExtreme!.reach + h) {
        this.lastFinish = this.curExtreme;
        this.phase = "recovery";
        this.curExtreme = s; // track max for next catch
      }
      return;
    }

    // recovery: look for the next catch
    if (s.reach > this.curExtreme!.reach) this.curExtreme = s;
    if (s.reach < this.curExtreme!.reach - h) {
      const newCatch = this.curExtreme!;
      if (this.prevCatch && this.lastFinish) {
        this.emit(this.prevCatch, this.lastFinish, newCatch);
      }
      this.prevCatch = newCatch;
      this.phase = "drive";
      this.driveSamples = [newCatch, s];
      this.curExtreme = s;
      this.finishPos = 1;
    }
  }

  private emit(c: Sample, f: Sample, nextCatch: Sample): void {
    const drive = this.driveSamples.slice(0, this.finishPos + 1);
    if (drive.length < 3) return;
    // The first completed stroke spans the warm-up/init boundary and is
    // unreliable (partial drive, unsettled cadence) — skip it.
    this.emitted += 1;
    if (this.emitted === 1) return;
    this.onStroke({
      catch: c,
      finish: f,
      driveT: drive.map((d) => d.t),
      driveKnee: drive.map((d) => d.knee),
      driveLean: drive.map((d) => d.lean),
      driveElbow: drive.map((d) => d.elbow),
      driveMs: f.t - c.t,
      recoveryMs: nextCatch.t - f.t,
      periodMs: nextCatch.t - c.t,
    });
  }
}
