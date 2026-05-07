"""Per-dataset F1 vs tokens scaling: 4 plots, line per W, distinct marker per D.

Each panel = one dataset.
- x-axis: avg tokens per question (k)
- y-axis: F1 on the 100q grid
- 3 lines: W=1, W=2, W=4 (color)
- 4 markers along each line: D=2, D=4, D=6, D=8 (shape)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "grid"


def _load() -> list[dict]:
    rows = []
    for ds_dir in sorted(GRID_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for cfg_dir in sorted(ds_dir.iterdir()):
            sj = cfg_dir / "summary.json"
            if not sj.exists():
                continue
            try:
                s = json.loads(sj.read_text())
            except Exception:
                continue
            cfg = s.get("config", {})
            rows.append({
                "dataset": ds_dir.name,
                "W": int(cfg.get("width", 0)),
                "D": int(cfg.get("max_depth", 0)),
                "f1": float(s.get("f1", 0.0)),
                "tokens": float(s.get("avg_tokens", 0.0)),
            })
    return rows


W_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
D_MARKERS = {2: "o", 4: "s", 6: "^", 8: "D"}


def _plot(rows: list[dict]) -> None:
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    datasets = sorted(by_ds.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.flatten()
    for ax, ds in zip(axes, datasets):
        for w in (1, 2, 4):
            pts = sorted([r for r in by_ds[ds] if r["W"] == w], key=lambda x: x["D"])
            if not pts:
                continue
            xs = [p["tokens"] / 1000.0 for p in pts]
            ys = [p["f1"] for p in pts]
            ax.plot(xs, ys, color=W_COLORS[w], linewidth=1.6, alpha=0.85,
                    label=f"W={w}", zorder=2)
            for p, x, y in zip(pts, xs, ys):
                ax.scatter(x, y, marker=D_MARKERS[p["D"]], color=W_COLORS[w],
                           s=80, edgecolor="black", linewidth=0.8, zorder=3)
                ax.annotate(f"W{p['W']}D{p['D']}", (x, y), fontsize=7,
                            xytext=(4, 4), textcoords="offset points",
                            color=W_COLORS[w], alpha=0.85)
        ax.set_xlabel("Avg tokens / question (k)")
        ax.set_ylabel("F1 (100q grid)")
        ax.set_title(ds)
        ax.grid(True, alpha=0.3)
        # Build legend entries: width colors + depth markers
        from matplotlib.lines import Line2D
        w_handles = [Line2D([0], [0], color=W_COLORS[w], lw=2, label=f"W={w}") for w in (1, 2, 4)]
        d_handles = [Line2D([0], [0], marker=D_MARKERS[d], color="black",
                             markerfacecolor="white", markeredgecolor="black",
                             linestyle="None", markersize=7, label=f"D={d}")
                     for d in (2, 4, 6, 8)]
        leg1 = ax.legend(handles=w_handles, loc="lower right", fontsize=8, title="Width")
        ax.add_artist(leg1)
        ax.legend(handles=d_handles, loc="upper left", fontsize=8, title="Depth")
    fig.suptitle("SPARC-RAG W×D scaling curves (Qwen3-14B, dense wiki18)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = GRID_DIR / "scaling_per_dataset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    _plot(_load())


if __name__ == "__main__":
    main()
