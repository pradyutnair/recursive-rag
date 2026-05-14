"""Aggregate grid summaries into a CSV and Markdown table.

Reads results/grid/<dataset>/W<W>_D<D>/summary.json and produces:
  results/grid/grid.csv
  results/grid/grid.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "grid"


def _rows() -> list[dict]:
    rows: list[dict] = []
    for ds_dir in sorted(GRID_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for cfg_dir in sorted(ds_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
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
                "n": int(s.get("n", 0)),
                "em": float(s.get("em", 0.0)),
                "f1": float(s.get("f1", 0.0)),
                "acc": float(s.get("acc", 0.0)),
                "avg_tokens": float(s.get("avg_tokens", 0.0)),
                "avg_rounds": float(s.get("avg_rounds", 0.0)),
                "avg_wall_s": float(s.get("avg_wall_s", 0.0)),
                "wall_clock_s": float(s.get("wall_clock_s", 0.0)),
            })
    return rows


def main() -> None:
    rows = _rows()
    if not rows:
        print("no summaries found", file=sys.stderr)
        return
    csv_path = GRID_DIR / "grid.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md_lines = ["| dataset | W | D |  n  |  EM   |  F1   |  Acc  | tokens | rounds | wall(s) |",
                "|---------|---|---|-----|-------|-------|-------|--------|--------|---------|"]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['W']} | {r['D']} | {r['n']:3d} |"
            f" {r['em']:.3f} | {r['f1']:.3f} | {r['acc']:.3f} |"
            f" {r['avg_tokens']:6.0f} | {r['avg_rounds']:5.2f} | {r['avg_wall_s']:5.1f} |"
        )
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    md_lines.append("\n## Best (max F1) per dataset\n")
    md_lines.append("| dataset | W | D |  F1   |  EM   | tokens |")
    md_lines.append("|---------|---|---|-------|-------|--------|")
    best_per_ds: dict[str, dict] = {}
    for ds, rs in by_ds.items():
        best = max(rs, key=lambda x: (x["f1"], -x["avg_tokens"]))
        best_per_ds[ds] = best
        md_lines.append(
            f"| {ds} | {best['W']} | {best['D']} | {best['f1']:.3f} | {best['em']:.3f} |"
            f" {best['avg_tokens']:6.0f} |"
        )
    md_path = GRID_DIR / "grid.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    (GRID_DIR / "best.json").write_text(json.dumps(best_per_ds, indent=2) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {GRID_DIR / 'best.json'}")


if __name__ == "__main__":
    main()
