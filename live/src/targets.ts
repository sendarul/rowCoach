// Loads the SHARED stroke targets (same file Tier 2 uses) so the live coach and
// the deep-dive grade strokes identically.
import targetsData from "../../shared/stroke_targets.json";

export type Grade = "good" | "minor" | "needs_work" | "unknown";

export interface MetricTarget {
  key: string;
  label: string;
  unit: string;
  targetMin: number;
  targetMax: number;
  minorTolerance: number;
  cueLow: string;
  cueHigh: string;
  priority: number;
}

function grade(t: MetricTarget, value: number): Grade {
  if (Number.isNaN(value)) return "unknown";
  if (value >= t.targetMin && value <= t.targetMax) return "good";
  const dist = value < t.targetMin ? t.targetMin - value : value - t.targetMax;
  return dist <= t.minorTolerance ? "minor" : "needs_work";
}

function cueFor(t: MetricTarget, value: number): string {
  if (value < t.targetMin) return t.cueLow;
  if (value > t.targetMax) return t.cueHigh;
  return "";
}

const raw = (targetsData as any).metrics as Record<string, any>;

export const TARGETS: Record<string, MetricTarget> = {};
for (const [key, m] of Object.entries(raw)) {
  TARGETS[key] = {
    key,
    label: m.label,
    unit: m.unit,
    targetMin: Number(m.target_min),
    targetMax: Number(m.target_max),
    minorTolerance: Number(m.minor_tolerance),
    cueLow: m.cue_low ?? "",
    cueHigh: m.cue_high ?? "",
    priority: Number(m.priority),
  };
}

export function gradeMetric(key: string, value: number): { grade: Grade; cue: string } {
  const t = TARGETS[key];
  return { grade: grade(t, value), cue: cueFor(t, value) };
}
