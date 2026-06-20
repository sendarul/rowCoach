# RowCoach 🚣

An AI rowing-technique coach for the erg. Film yourself rowing side-on, and get
back specific, actionable coaching — not a vague "you're 82% similar to a pro."

Two tiers (see `IMPLEMENTATION.md` for the full design):

- **Tier 1 — Live coach (PWA):** real-time spoken cues to your earbuds while you
  row, on the phone (`live/`). **Built.**
- **Tier 2 — Deep dive (Python):** post-session report + annotated video +
  per-stroke charts, with Claude as the coach brain (`analyze/`). **Built & calibrated.**

---

## Tier 2 quickstart

### 1. Install
```powershell
cd analyze
python -m pip install --user -r requirements.txt
```
The pose model (~9 MB) auto-downloads to `analyze/models/` on first run.

### 2. (Optional) Add a Claude API key
Richer coaching when present; works without it (rule-based fallback).
```powershell
copy .env.example .env   # then edit .env and paste your key
```

### 3. Record a clip
- Phone on a **tripod, side-on**, full body in frame.
- ~30 seconds of steady rowing. `.mp4` or `.mov` (iPhone), portrait or landscape.

### 4. Analyze
```powershell
python -m rowcoach analyze path\to\clip.mp4
```

Output lands in `analyze/outputs/<date>_<clip>/`:
- `report.md` — coaching report (top fixes + full metric table)
- `annotated.mp4` — skeleton overlay, phase labels, fault callouts
- `metrics.json` / `metrics.csv` — per-stroke + aggregate metrics
- `charts/per_stroke.png` — metrics vs. target bands

### Options
```
--no-video    skip the annotated video (faster)
--no-llm      force the offline rule-based report
--model NAME  Claude model (default: claude-opus-4-8)
--facing {auto,left,right}   override side detection
--debug       save intermediate signals for tuning
```

---

## Tier 1 — live coach on your iPhone

**Deployed: https://sendarul.github.io/rowCoach/** — no laptop needed at the gym.
Auto-redeploys on every push to `main` (GitHub Actions → Pages).

On your iPhone:

1. Open **https://sendarul.github.io/rowCoach/** in Safari (real HTTPS, so camera
   works with no cert warnings). Optional: Share → **Add to Home Screen**.
2. Pair your **Bluetooth earbuds**.
3. Phone on a **tripod, side-on**, full body in frame.
4. Tap **Start coaching** (this also unlocks audio — required by iOS).
5. Row. You'll hear short cues ("legs first!", "slow the slide") for clear faults.

> First load needs internet (MoveNet model downloads from Google, then caches).

Tier 1 uses **MoveNet** (TensorFlow.js) for in-browser pose and the **Web Speech
API** for cues. It grades against the same `shared/stroke_targets.json` as Tier 2.

### Local dev
```powershell
cd live
npm install
npm run dev   # HTTPS dev server on your LAN IP for quick iteration
```

## What it measures
Catch shin angle · catch back angle · legs-before-back sequencing · arm-break
timing · finish layback · drive:recovery ratio · stroke rate · rhythm consistency.

Targets live in `shared/stroke_targets.json` (shared with Tier 1). They're
provisional starting values — **tune them against real clips.**

## Status
**Tier 2 is built, validated, and calibrated** on real side-view footage:
- Stroke rate verified against the Concept2 monitor (30→30.0, 20→~20 SPM).
- Gold-standard steady-state rowing grades all-good; flawed clips correctly flag
  sequencing, excessive layback, and over-reach.
- Targets in `shared/stroke_targets.json` softened from elite to recreational ranges.

**Known limitations:** back-rounding (lumbar flexion) isn't detectable with
BlazePose (torso is modeled as a straight line); arm-break timing isn't fully
validated; the pipeline assumes a single continuous shot (no hard cuts).

**Best next calibration:** a continuous clip of *you* rowing — rate/ratio targets
are somewhat individual. **Next feature:** Tier 1 (live PWA).
