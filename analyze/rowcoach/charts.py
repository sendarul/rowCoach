"""Per-stroke charts (matplotlib, no GUI backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import MetricTarget  # noqa: E402
from .metrics import MetricsReport  # noqa: E402

# per-stroke series to plot with their target bands
_SERIES = [
    ("finish_layback_angle", "Finish layback (deg)"),
    ("drive_recovery_ratio", "Drive:recovery ratio"),
    ("catch_shin_angle", "Catch shin angle (deg)"),
    ("legs_back_sequence", "Legs-before-back (ms)"),
]


def _target_band(ax, t: MetricTarget):
    ax.axhspan(t.target_min, t.target_max, color="tab:green", alpha=0.12,
               label="target")


def render(report: MetricsReport, targets: dict[str, MetricTarget],
           out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    n = report.n_strokes
    if n == 0:
        return paths
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, _SERIES):
        vals = np.array([getattr(s, key) for s in report.per_stroke], dtype=float)
        t = targets[key]
        _target_band(ax, t)
        ax.plot(x, vals, "o-", color="tab:blue")
        ax.set_title(title)
        ax.set_xlabel("stroke #")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Per-stroke metrics vs target bands", fontsize=13)
    fig.tight_layout()
    p = out_dir / "per_stroke.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths
