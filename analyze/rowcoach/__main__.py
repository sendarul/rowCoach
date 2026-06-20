"""RowCoach Tier 2 CLI.

    python -m rowcoach analyze <video_path> [options]

Currently implements the scaffold step: probe a clip and print its properties,
with footage sanity warnings. Subsequent build steps add pose, segmentation,
metrics, coaching, annotation, and charts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config


def _cmd_analyze(args: argparse.Namespace) -> int:
    from . import report, video
    from dotenv import load_dotenv

    load_dotenv()  # pick up ANTHROPIC_API_KEY from analyze/.env if present

    try:
        info = video.probe(args.video_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"RowCoach {__version__} — analyzing: {info.path.name}")
    print(f"  {info.width}x{info.height} "
          f"({'portrait' if info.is_portrait else 'landscape'}), "
          f"{info.fps:.1f} fps, {info.duration_s:.1f}s, "
          f"rotation {info.rotation}° (auto-corrected)")
    if not info.fps:
        print("  WARNING: could not read frame rate; timing metrics unreliable.",
              file=sys.stderr)
    print("  running pose estimation… (this can take a bit)")

    try:
        report.run(
            video_path=args.video_path,
            out_base=args.out,
            make_video=not args.no_video,
            use_llm=not args.no_llm,
            model=args.model,
            facing_override=args.facing,
            debug=args.debug,
        )
    except Exception as e:
        print(f"error during analysis: {e}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rowcoach",
        description="AI erg-rowing technique coach (Tier 2, post-session).",
    )
    p.add_argument("--version", action="version", version=f"rowcoach {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Analyze a side-view rowing clip.")
    a.add_argument("video_path", type=Path, help="Path to the .mp4/.mov clip.")
    a.add_argument("--out", type=Path, default=Path("./outputs"),
                   help="Output base directory (default: ./outputs).")
    a.add_argument("--no-video", action="store_true",
                   help="Skip annotated video (faster).")
    a.add_argument("--no-llm", action="store_true",
                   help="Force rule-based report (skip Claude).")
    a.add_argument("--model", default=config.DEFAULT_MODEL,
                   help=f"Claude model (default: {config.DEFAULT_MODEL}).")
    a.add_argument("--facing", choices=["auto", "left", "right"], default="auto",
                   help="Rower facing direction (default: auto-detect).")
    a.add_argument("--debug", action="store_true",
                   help="Save intermediate signals/plots for tuning.")
    a.set_defaults(func=_cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; make our output UTF-8 safe.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
