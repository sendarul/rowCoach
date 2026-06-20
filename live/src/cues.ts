// Cue arbiter: from a stroke's graded metrics, pick the single most important
// correction and speak it — with debouncing so we don't nag every stroke.

import type { StrokeMetrics } from "./metrics";
import { TARGETS, gradeMetric } from "./targets";

const SEVERITY: Record<string, number> = {
  needs_work: 0,
  minor: 1,
  good: 2,
  unknown: 3,
};

// Don't repeat the same cue within this window, and never speak more often than
// the min gap (so cues stay sparse and "urgent", not chatty).
const REPEAT_COOLDOWN_MS = 8000;
const MIN_GAP_MS = 3000;
// Only cue clear faults by default; minors are noted on-screen, not spoken.
const SPEAK_MINORS = false;

export interface PickedCue {
  key: string;
  text: string;
  grade: string;
}

export class CueArbiter {
  private lastSpokenAt = 0;
  private lastTextAt: Record<string, number> = {};
  private voiceReady = false;

  /** Must be called from a user gesture (iOS unlocks audio only then). */
  unlockAudio(): void {
    try {
      const u = new SpeechSynthesisUtterance("");
      u.volume = 0;
      window.speechSynthesis.speak(u);
      this.voiceReady = true;
    } catch {
      /* speech not available */
    }
  }

  /** Choose the worst fault for this stroke (or null if all good). */
  pick(m: StrokeMetrics): PickedCue | null {
    const faults: PickedCue[] = [];
    for (const [key, value] of Object.entries(m)) {
      const { grade, cue } = gradeMetric(key, value as number);
      if (grade === "needs_work" || (SPEAK_MINORS && grade === "minor")) {
        if (cue) faults.push({ key, text: cue, grade });
      }
    }
    if (!faults.length) return null;
    faults.sort(
      (a, b) =>
        SEVERITY[a.grade] - SEVERITY[b.grade] ||
        TARGETS[a.key].priority - TARGETS[b.key].priority,
    );
    return faults[0];
  }

  /** Pick + speak with debouncing. Returns the cue actually spoken (or null). */
  coach(m: StrokeMetrics, now: number): PickedCue | null {
    const cue = this.pick(m);
    if (!cue) return null;
    if (now - this.lastSpokenAt < MIN_GAP_MS) return null;
    if (now - (this.lastTextAt[cue.text] ?? -Infinity) < REPEAT_COOLDOWN_MS) return null;
    this.speak(cue.text);
    this.lastSpokenAt = now;
    this.lastTextAt[cue.text] = now;
    return cue;
  }

  private speak(text: string): void {
    if (!this.voiceReady || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    u.pitch = 1.0;
    window.speechSynthesis.cancel(); // interrupt any queued cue; freshest wins
    window.speechSynthesis.speak(u);
  }
}
