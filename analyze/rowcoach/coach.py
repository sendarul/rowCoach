"""The coaching brain.

Turns graded metrics into prioritized, plain-English coaching. Uses Claude when
an API key is available; always falls back to a deterministic rule-based report
so the tool works offline / without a key. The LLM never sees video -- only the
compact numeric summary built here.
"""

from __future__ import annotations

import os

from .config import DEFAULT_MODEL, MetricTarget
from .metrics import GradedMetric, MetricsReport

_SEVERITY = {"needs_work": 0, "minor": 1, "good": 2, "unknown": 3}


def _fmt(g: GradedMetric) -> str:
    if g.grade == "unknown":
        return f"- {g.label}: not measurable"
    return f"- {g.label}: {g.value:.1f} {g.unit} -> {g.grade}" + (
        f" (cue: {g.cue})" if g.cue else "")


def build_summary(report: MetricsReport, targets: dict[str, MetricTarget]) -> str:
    """Compact, model-friendly numeric summary of the session."""
    lines = [
        f"Strokes analyzed: {report.n_strokes}",
        f"Rower facing: {report.facing}",
        "",
        "Metrics (value -> grade):",
    ]
    # order by priority so the most important metrics lead
    ordered = sorted(report.aggregate.values(),
                     key=lambda g: targets[g.key].priority)
    for g in ordered:
        lines.append(_fmt(g))
    for note in report.notes:
        lines.append(f"NOTE: {note}")
    return "\n".join(lines)


def _priority_faults(report: MetricsReport,
                     targets: dict[str, MetricTarget]) -> list[GradedMetric]:
    faults = [g for g in report.aggregate.values()
              if g.grade in ("needs_work", "minor")]
    faults.sort(key=lambda g: (_SEVERITY[g.grade], targets[g.key].priority))
    return faults


def rule_based_report(report: MetricsReport,
                      targets: dict[str, MetricTarget]) -> str:
    """Deterministic coaching report (offline floor)."""
    faults = _priority_faults(report, targets)
    out = ["# Your rowing session\n",
           f"Analyzed **{report.n_strokes} strokes** (facing {report.facing}).\n"]
    for note in report.notes:
        out.append(f"> ⚠️ {note}\n")

    if not faults:
        out.append("## ✅ Looking solid\n")
        out.append("No clear technique faults stood out this session — your key "
                   "angles and sequencing are within target ranges. Keep it up.\n")
    else:
        out.append("## Top things to work on\n")
        for i, g in enumerate(faults[:3], 1):
            t = targets[g.key]
            direction = "low" if g.value < t.target_min else "high"
            out.append(
                f"**{i}. {g.label}** — {g.value:.1f} {g.unit} "
                f"(target {t.target_min:g}–{t.target_max:g}). "
                f"_{g.grade.replace('_', ' ')}_.\n\n"
                f"   - What it means: {t.describe}\n"
                f"   - Focus cue: **{g.cue or t.cue_low or t.cue_high}** "
                f"(running a touch {direction}).\n")

    out.append("\n## All metrics\n")
    out.append("| Metric | Value | Target | Grade |")
    out.append("|---|---|---|---|")
    ordered = sorted(report.aggregate.values(),
                     key=lambda g: targets[g.key].priority)
    for g in ordered:
        t = targets[g.key]
        val = "n/a" if g.grade == "unknown" else f"{g.value:.1f} {g.unit}"
        out.append(f"| {g.label} | {val} | {t.target_min:g}–{t.target_max:g} | {g.grade} |")
    return "\n".join(out) + "\n"


_SYSTEM = (
    "You are an experienced, encouraging indoor-rowing (erg) coach. You are given "
    "objective biomechanical metrics extracted from a side-view video of one "
    "rower, already graded against target ranges. The rower is strong but new to "
    "technique. Give focused, actionable coaching:\n"
    "1. Open with one genuine encouraging sentence about what's working.\n"
    "2. Give the TOP 1-3 priorities to fix, most important first. For each: name "
    "the fault, explain briefly why it matters, and give ONE concrete cue or drill.\n"
    "3. Keep it concise and practical. Use the metric values to be specific. Do "
    "not invent metrics you weren't given. Output Markdown."
)


def claude_report(summary: str, model: str = DEFAULT_MODEL) -> str | None:
    """Ask Claude for coaching. Returns None on any failure (caller falls back)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Here are my rowing metrics for this session:\n\n{summary}",
            }],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
        text = "\n".join(parts).strip()
        return text or None
    except Exception:
        return None


def coach(report: MetricsReport, targets: dict[str, MetricTarget],
          use_llm: bool = True, model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    """Return (markdown_report, used_llm)."""
    summary = build_summary(report, targets)
    if use_llm:
        llm = claude_report(summary, model)
        if llm:
            header = (f"# Your rowing session\n\n_Analyzed {report.n_strokes} "
                      f"strokes (facing {report.facing}). Coaching by Claude._\n\n")
            notes = "".join(f"> ⚠️ {n}\n\n" for n in report.notes)
            table = "\n## All metrics\n" + _metrics_table(report, targets)
            return header + notes + llm + "\n" + table, True
    return rule_based_report(report, targets), False


def _metrics_table(report: MetricsReport,
                   targets: dict[str, MetricTarget]) -> str:
    rows = ["| Metric | Value | Target | Grade |", "|---|---|---|---|"]
    ordered = sorted(report.aggregate.values(),
                     key=lambda g: targets[g.key].priority)
    for g in ordered:
        t = targets[g.key]
        val = "n/a" if g.grade == "unknown" else f"{g.value:.1f} {g.unit}"
        rows.append(f"| {g.label} | {val} | {t.target_min:g}–{t.target_max:g} | {g.grade} |")
    return "\n".join(rows) + "\n"
