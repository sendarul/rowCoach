import "./style.css";
import { type KP } from "./geometry";
import { computeStroke } from "./metrics";
import { estimate, loadDetector } from "./pose";
import { StrokeTracker } from "./segmentation";
import { CueArbiter } from "./cues";

const video = document.getElementById("cam") as HTMLVideoElement;
const canvas = document.getElementById("overlay") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const hudPhase = document.getElementById("hud-phase") as HTMLSpanElement;
const hudSpm = document.getElementById("hud-spm") as HTMLSpanElement;
const hudCue = document.getElementById("hud-cue") as HTMLSpanElement;

const arbiter = new CueArbiter();
let lastSpm = NaN;

const tracker = new StrokeTracker((stroke) => {
  const m = computeStroke(stroke, tracker.facing!);
  lastSpm = m.stroke_rate;
  const cue = arbiter.coach(m, performance.now());
  if (cue) {
    hudCue.textContent = cue.text;
    hudCue.classList.add("flash");
    setTimeout(() => hudCue.classList.remove("flash"), 1200);
  }
});

// COCO-17 skeleton connections for drawing.
const BONES: [number, number][] = [
  [5, 7], [7, 9], [6, 8], [8, 10], // arms
  [5, 6], [5, 11], [6, 12], [11, 12], // torso
  [11, 13], [13, 15], [12, 14], [14, 16], // legs
];

function draw(kp: KP[] | null): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!kp) return;
  ctx.strokeStyle = "#3ddc97";
  ctx.fillStyle = "#ffcc33";
  ctx.lineWidth = 4;
  for (const [a, b] of BONES) {
    if (kp[a]?.score > 0.3 && kp[b]?.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(kp[a].x, kp[a].y);
      ctx.lineTo(kp[b].x, kp[b].y);
      ctx.stroke();
    }
  }
  for (const k of kp) {
    if (k.score > 0.3) {
      ctx.beginPath();
      ctx.arc(k.x, k.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

async function loop(): Promise<void> {
  const kp = await estimate(video);
  tracker.push(performance.now(), kp);
  draw(kp);
  hudPhase.textContent = tracker.phase.toUpperCase();
  hudSpm.textContent = Number.isNaN(lastSpm) ? "-- spm" : `${lastSpm.toFixed(0)} spm`;
  requestAnimationFrame(loop);
}

async function start(): Promise<void> {
  startBtn.disabled = true;
  arbiter.unlockAudio(); // must happen in the user-gesture (iOS audio unlock)
  try {
    statusEl.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    statusEl.textContent = "Loading pose model…";
    await loadDetector();

    try {
      await (navigator as any).wakeLock?.request("screen");
    } catch {
      /* wake lock optional */
    }

    statusEl.textContent = "Coaching — side-on, full body in frame.";
    startBtn.style.display = "none";
    loop();
  } catch (e) {
    statusEl.textContent = `Could not start: ${(e as Error).message}`;
    startBtn.disabled = false;
  }
}

startBtn.addEventListener("click", start);
