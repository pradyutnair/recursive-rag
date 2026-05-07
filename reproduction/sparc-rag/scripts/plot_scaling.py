"""Plot per-dataset F1 vs avg_tokens scaling curves (paper Figure 3 layout).

Usage:
    python scripts/plot_scaling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "grid"


def _load_rows() -> list[dict]:
    rows: list[dict] = []
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
                "em": float(s.get("em", 0.0)),
                "tokens": float(s.get("avg_tokens", 0.0)),
            })
    return rows


def _plot(rows: list[dict], metric: str, fname: str) -> None:
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    datasets = sorted(by_ds.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.6))
    if n == 1:
        axes = [axes]
    color_map = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
    marker_map = {1: "o", 2: "s", 4: "^"}
    for ax, ds in zip(axes, datasets):
        for w in (1, 2, 4):
            pts = sorted([r for r in by_ds[ds] if r["W"] == w], key=lambda x: x["D"])
            if not pts:
                continue
            xs = [p["tokens"] / 1000.0 for p in pts]
            ys = [p[metric] for p in pts]
            ax.plot(xs, ys, marker=marker_map[w], color=color_map[w],
                    label=f"W={w}", linewidth=1.6, markersize=6)
            for p, x, y in zip(pts, xs, ys):
                ax.annotate(f"D={p['D']}", (x, y), fontsize=7, alpha=0.6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Tokens per question (k)")
        ax.set_ylabel(metric.upper())
        ax.set_title(ds)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = GRID_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    rows = _load_rows()
    if not rows:
        print("no rows")
        return
    _plot(rows, "f1", "scaling_f1.png")
    _plot(rows, "em", "scaling_em.png")


if __name__ == "__main__":
    main()
