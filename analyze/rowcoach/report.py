"""Pipeline orchestration + output assembly.

Runs the full Tier 2 analysis and writes a timestamped session folder:
report.md, annotated.mp4, metrics.json, metrics.csv, charts/.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import datetime
from pathlib import Path

from . import charts, coach, config, geometry, metrics, pose, segmentation, video
from .config import MAX_LOW_CONF_FRACTION, MIN_FRAME_CONFIDENCE


def _session_dir(base: Path, clip: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    d = base / f"{stamp}_{clip.stem}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _footage_notes(seq: pose.PoseSequence, info: video.VideoInfo) -> list[str]:
    notes: list[str] = []
    if seq.n_frames:
        low = float((seq.frame_confidence < MIN_FRAME_CONFIDENCE).mean())
        if low > MAX_LOW_CONF_FRACTION:
            notes.append(
                f"{low*100:.0f}% of frames had low pose confidence — camera angle, "
                "framing, or lighting may be off. Treat metrics with caution.")
    if info.duration_s and info.duration_s < config.MIN_CLIP_SECONDS:
        notes.append("Clip is very short; few strokes to average over.")
    return notes


def _write_metrics_files(report: metrics.MetricsReport, out_dir: Path) -> None:
    # JSON: aggregate + per-stroke
    agg = {k: dataclasses.asdict(g) for k, g in report.aggregate.items()}
    per = [dataclasses.asdict(s) for s in report.per_stroke]
    (out_dir / "metrics.json").write_text(
        json.dumps({"n_strokes": report.n_strokes, "facing": report.facing,
                    "notes": report.notes, "aggregate": agg, "per_stroke": per},
                   indent=2), encoding="utf-8")
    # CSV: one row per stroke
    if per:
        fields = list(per[0].keys())
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stroke"] + fields)
            w.writeheader()
            for i, row in enumerate(per, 1):
                w.writerow({"stroke": i, **row})


def run(video_path: Path, out_base: Path, *, make_video: bool = True,
        use_llm: bool = True, model: str = config.DEFAULT_MODEL,
        facing_override: str = "auto", debug: bool = False) -> Path:
    targets = config.load_targets()

    info = video.probe(video_path)
    seq = pose.extract(info)
    smoothed = geometry.smooth_landmarks(seq.landmarks)

    facing = (facing_override if facing_override in ("left", "right")
              else geometry.detect_facing(smoothed))

    strokes = segmentation.segment(smoothed, facing, info.fps)
    per_stroke = [metrics.compute_stroke(smoothed, s, facing) for s in strokes]
    report = metrics.aggregate(per_stroke, targets, facing)
    report.notes = _footage_notes(seq, info)

    out_dir = _session_dir(out_base, Path(video_path))

    # coaching report
    report_md, used_llm = coach.coach(report, targets, use_llm=use_llm, model=model)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    # metrics + charts
    _write_metrics_files(report, out_dir)
    chart_paths = charts.render(report, targets, out_dir / "charts")

    # annotated video
    if make_video and strokes:
        from . import annotate
        bad_seq = {i for i, s in enumerate(per_stroke, 1)
                   if not _is_nan(s.legs_back_sequence)
                   and targets["legs_back_sequence"].grade(s.legs_back_sequence)
                   != "good"}
        annotate.render(info, pose.PoseSequence(smoothed, seq.frame_confidence,
                                                info.fps),
                        strokes, per_stroke, facing, bad_seq,
                        out_dir / "annotated.mp4")

    _print_summary(report, out_dir, used_llm, len(chart_paths), make_video and bool(strokes))
    return out_dir


def _is_nan(x: float) -> bool:
    return x != x


def _print_summary(report: metrics.MetricsReport, out_dir: Path, used_llm: bool,
                   n_charts: int, made_video: bool) -> None:
    print(f"\n[OK] Analysis complete - {report.n_strokes} strokes "
          f"(facing {report.facing}).")
    for n in report.notes:
        print(f"   [!] {n}")
    print(f"   coaching : {'Claude' if used_llm else 'rule-based (offline)'}")
    print(f"   outputs  : {out_dir}")
    print(f"     - report.md")
    print(f"     - metrics.json / metrics.csv")
    if n_charts:
        print(f"     - charts/per_stroke.png")
    if made_video:
        print(f"     - annotated.mp4")
