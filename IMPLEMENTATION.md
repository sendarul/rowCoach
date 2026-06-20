# RowCoach — Implementation Plan

An AI rowing-technique coach for erg (indoor rowing) machines. It watches
**side-view** footage of you rowing and gives feedback in two tiers:

- **Tier 1 — Live coach (PWA):** real-time spoken cues to your earbuds *while you
  row*, computed entirely on the phone.
- **Tier 2 — Deep dive (Python):** after the session, a written coaching report,
  an annotated video, and per-stroke metrics + charts, with Claude as the coach brain.

Both tiers share the same biomechanical model of a good stroke — the difference is
latency budget and output format, not what counts as "good rowing."

---

## 1. Design philosophy

**Measure specific biomechanics, don't score similarity.** We extract concrete,
named metrics (shin angle at the catch, leg-vs-back sequencing, layback angle) and
grade each against a target range. This is what makes feedback *actionable*
("legs first!") instead of opaque ("you're 0.82 similar to a pro").

> **Why not compare to a reference video?** A prior project (Kapica, 2024) tried
> exactly that for rowing: pose-estimate the user + a pro, then score them with
> cosine similarity. Two problems we deliberately avoid:
> 1. **Weak, un-actionable signal** — correct vs. deliberately-wrong form differed
>    by only ~0.015 cosine similarity, and a single score can't say *what* to fix.
> 2. **The alignment problem** — comparing two videos requires time-aligning them
>    (he did it by hand + Dynamic Time Warping and called full automation "a topic
>    for a separate article"). We never have this problem: we measure each stroke
>    against *itself*, so our stroke segmentation IS our alignment.
>
> That project did validate one thing we rely on: lightweight pose estimation runs
> **in the phone browser at 30+ FPS** (MoveNet) — which is what makes Tier 1 feasible.

**The LLM is never in the real-time loop.** A stroke lasts ~2s; live cues must fire
in <1s and work with no connectivity. So Tier 1 is 100% deterministic rules on the
phone. Claude only does the reflective, between/after-session coaching in Tier 2.

**Always produces output.** Tier 2's Claude call has a deterministic rule-based
fallback so the report works offline / without an API key.

**2D side-view, angles & timing.** We grade relative angles and sequencing (which a
side view handles well), not absolute distances.

---

## 2. Hardware & session setup (decided)

- **Capture:** iPhone on a **small tripod**, side-on, full body in frame.
- **Audio:** cues spoken to **Bluetooth earbuds** (paired to the phone).
- **Cue style:** **short urgent corrections only** ("legs first!", "slow the
  slide", "ease the layback") — no chatter mid-piece.
- **Tier 1 runs on the phone**, in the browser (laptop not needed during workout).
- **Tier 2 runs on the laptop** afterward, on the recorded clip.

---

## 3. The shared stroke model

The stroke cycle: **catch → drive → finish → recovery**. Drive sequence is
*legs → back → arms*; recovery reverses it. Both tiers compute from these 8 metrics
(targets in a shared config; tune on real clips):

| # | Metric | What it catches | Live cue (Tier 1) |
|---|---|---|---|
| 1 | Catch shin angle | Over/under-compression | "reach a bit more" / "too far" |
| 2 | Catch back angle | Hunching / overreach | "flatten your back" |
| 3 | Legs-vs-back sequencing | Opening the back too early | **"legs first!"** |
| 4 | Arm-break timing | Pulling with arms too early | "legs then arms" |
| 5 | Finish layback angle | Too much / too little layback | "ease the layback" |
| 6 | Drive:recovery ratio | Rushing the slide (target ≈ 1:2) | **"slow the slide"** |
| 7 | Stroke rate (SPM) | Pacing | (shown, rarely cued) |
| 8 | Rhythm consistency | Choppiness stroke-to-stroke | "smooth it out" |

Live tier fires the **single highest-priority** cue per stroke (never stacks).
Confidence-weight all angles by per-keypoint confidence; suppress cues when
tracking confidence is low.

---

## 4. Tier 1 — Live coach (PWA)

A Progressive Web App you open in iPhone Safari and add to your home screen.
No app store, no install, no laptop.

### Stack
| Concern | Choice |
|---|---|
| Pose (in-browser) | TensorFlow.js **MoveNet** (Lightning) — 17 keypoints, 30+ FPS on phone |
| Camera | `getUserMedia` (rear camera, landscape) |
| Audio cues | Web Speech API (`SpeechSynthesis`) → Bluetooth earbuds |
| Language | TypeScript |
| Build | Vite (PWA plugin for manifest + service worker) |
| Recording | `MediaRecorder` saves the raw clip for Tier 2 |

### Real-time pipeline (per frame, on phone)
1. Grab camera frame → MoveNet → 17 keypoints + confidences.
2. Smooth keypoints (short rolling window) and auto-detect facing side.
3. Update the **drive signal** (wrist horizontal position) → live stroke-phase
   state machine (catch / drive / finish / recovery), detecting catch & finish in
   real time (lightweight peak/zero-cross detection, not offline `find_peaks`).
4. At key phase transitions, compute the relevant metrics for the just-finished
   stroke and compare to targets.
5. **Cue arbiter:** pick the worst single fault, debounce (don't repeat the same
   cue every stroke), and speak it. Stay silent if everything's in range.

### Live UX
- Big "Start / Stop" + a minimal on-screen HUD (current SPM, phase, last cue) —
  but the phone's on a tripod, so **audio is the primary channel**.
- Screen-wake lock during a session.
- On Stop: offer to **download the recorded clip** (and a JSON of live metrics) to
  move to the laptop for Tier 2.

### iPhone/Safari specifics to plan around
- `getUserMedia` needs **HTTPS** (serve over https / localhost; for phone testing
  use a tunneled https URL).
- iOS Safari requires a **user gesture** before audio/`SpeechSynthesis` will play —
  trigger a silent utterance on the Start tap to unlock it.
- Keep work off the main thread where possible; cap inference resolution to hold FPS.

---

## 5. Tier 2 — Deep dive (Python, laptop)

The post-session analyst: precise metrics + Claude coaching + annotated video.
(This is the original plan, now positioned as Tier 2.)

### Stack
| Concern | Choice |
|---|---|
| Language | Python 3.10+ |
| Pose | MediaPipe Pose (33 landmarks, CPU, per-landmark confidence) |
| Video I/O + annotation | OpenCV |
| Signal processing | NumPy + SciPy (`find_peaks`, Savitzky–Golay) |
| Charts | Matplotlib |
| Coaching LLM | Anthropic SDK, model `claude-opus-4-8` (configurable) |
| Env | `python-dotenv` for `ANTHROPIC_API_KEY` |

### Project structure
```
rowCoach/
├── IMPLEMENTATION.md
├── README.md
├── shared/
│   └── stroke_targets.json     # single source of truth for metric target ranges
│                               # (Tier 1 imports as JSON, Tier 2 loads in config)
├── live/                       # Tier 1 PWA
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.ts             # camera + render loop + start/stop + screen lock
│       ├── pose.ts             # MoveNet load + per-frame keypoints
│       ├── segmentation.ts     # live phase state machine, catch/finish detection
│       ├── metrics.ts          # the 8 metrics from keypoints
│       ├── cues.ts             # cue arbiter + debounce + SpeechSynthesis
│       └── targets.ts          # loads ../shared/stroke_targets.json
└── analyze/                    # Tier 2 Python
    ├── requirements.txt
    ├── .env.example
    ├── rowcoach/
    │   ├── __main__.py         # CLI: python -m rowcoach analyze clip.mp4
    │   ├── config.py           # loads ../../shared/stroke_targets.json + thresholds
    │   ├── video.py            # load/normalize (rotation, fps), iphone .mov/.mp4
    │   ├── pose.py             # MediaPipe wrapper -> landmarks + confidence
    │   ├── geometry.py         # angle helpers, side detection, smoothing
    │   ├── segmentation.py     # offline stroke split + catch/finish (find_peaks)
    │   ├── metrics.py          # 8 metrics per stroke + aggregates + grading
    │   ├── coach.py            # Claude call + rule-based fallback
    │   ├── annotate.py         # skeleton/phase/callout overlay video
    │   ├── charts.py           # matplotlib per-stroke charts
    │   └── report.py           # assemble Markdown + write outputs
    ├── samples/                # downloaded test clips (gitignored)
    │   ├── good/  └── flawed/
    └── outputs/                # timestamped session folders (gitignored)
        └── 2026-06-19_1530_clip/
            ├── report.md  ├── annotated.mp4
            ├── metrics.json  ├── metrics.csv  └── charts/
```

### CLI
```
python -m rowcoach analyze <video_path> [options]
  --out DIR            Output base dir (default: ./outputs)
  --no-video           Skip annotated video (faster)
  --no-llm             Force rule-based report (skip Claude)
  --model NAME         Claude model (default: claude-opus-4-8)
  --facing {auto,left,right}
  --debug              Save intermediate signals/plots for tuning
```

### Stage notes (the parts most likely to need tuning)
- **video.py** — fix iPhone rotation metadata OpenCV ignores; handle `.mov`/`.mp4`,
  portrait or landscape; warn on unknown fps or <5s clips.
- **segmentation.py** — drive signal = smoothed wrist horizontal position;
  `find_peaks` with min-distance from plausible SPM (16–40); catch = furthest
  forward, finish = furthest back (orientation-normalized); drop partial first/last
  strokes from grading. **This is the technical core — validate on real clips early.**
- **coach.py** — build a compact numeric summary → Claude for prioritized top-3
  coaching (why + a cue/drill each); deterministic templated fallback on no
  key/offline/error; never crashes the run.
- **annotate.py** — skeleton overlay + phase label + fault callouts at the exact
  offending frames + small HUD (stroke #, SPM, drive:recovery).

---

## 6. Shared targets (`shared/stroke_targets.json`)
Single source of truth so both tiers agree on "good." Initial values from standard
Concept2/coaching guidance, tuned once we see real clips. Each metric: target range
+ thresholds for `good` / `minor` / `needs work`. Examples: layback ~30–45°, shin
near vertical at catch, drive:recovery ~1:1.8–1:2.2.

---

## 7. Build order

**Phase A — Tier 2 first (gets the biomechanics right on real data):**
1. Scaffold `analyze/`, requirements, CLI that prints fps/frame count.
2. MediaPipe landmarks + debug overlay on a downloaded side-view clip.
3. **Stroke segmentation** — verify catch/finish/stroke count on good + flawed clips.
4. The 8 metrics + grading; confirm good clips grade well, flawed clips get flagged.
5. Rule-based report → then Claude integration in `coach.py`.
6. Annotated video + charts + timestamped output assembly.
7. Lock `shared/stroke_targets.json` from what we learned.

**Phase B — Tier 1 live PWA (reuses the validated model):**
8. Vite + PWA scaffold; camera preview on iPhone over https; audio-unlock on Start.
9. MoveNet in-browser; confirm 30+ FPS and stable keypoints on the phone.
10. Port the stroke state machine + metrics to TS (reading shared targets).
11. Cue arbiter + Web Speech to earbuds; debounce; silence-when-good.
12. `MediaRecorder` clip download → hand off to Tier 2. Field-test at the gym.

Doing Tier 2 first means Tier 1 inherits an already-correct definition of a good
stroke instead of re-deriving it under real-time pressure.

---

## 8. Test assets
Genuine **side views, full body in frame**, in `analyze/samples/`:
- `good/` — 1–2 strong-technique clips ("Olympic rower erg side view").
- `flawed/` — 1–2 beginner clips to confirm fault detection.

Acceptance: good clips score mostly `good`; flawed clips surface expected faults;
stroke count + SPM match a manual count; Tier 1 fires the right cue on flawed clips.

---

## 9. Explicitly out of scope for v1 (maybe-later)
- **"Compare to a pro" mode** — deliberately skipped (weak, un-actionable signal +
  the alignment problem). If ever added, DTW is the tool, and it could also score
  rhythm consistency (metric #8).
- Native iOS app (only if the PWA can't hold FPS).
- Cloud compute (rejected — latency + gym-wifi reliability).
- Multi-session progress dashboards / history trends.
```
