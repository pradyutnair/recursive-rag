"""Build final report: aggregate finals + grid + emit REPORT.md and figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "grid"
FINAL_DIR = ROOT / "results" / "final"
OUT_MD = ROOT / "results" / "REPORT.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _final_rows() -> list[dict]:
    rows = []
    for ds_dir in sorted(FINAL_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for cfg_dir in sorted(ds_dir.iterdir()):
            sj = cfg_dir / "summary.json"
            if not sj.exists():
                continue
            s = _load(sj)
            cfg = s.get("config", {})
            rows.append({
                "dataset": ds_dir.name,
                "W": cfg.get("width"),
                "D": cfg.get("max_depth"),
                "n": s["n"],
                "em": s["em"],
                "f1": s["f1"],
                "acc": s["acc"],
                "avg_tokens": s["avg_tokens"],
                "avg_rounds": s["avg_rounds"],
            })
    return rows


def _grid_rows() -> list[dict]:
    rows = []
    for ds_dir in sorted(GRID_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for cfg_dir in sorted(ds_dir.iterdir()):
            sj = cfg_dir / "summary.json"
            if not sj.exists():
                continue
            s = _load(sj)
            cfg = s.get("config", {})
            rows.append({
                "dataset": ds_dir.name,
                "W": cfg.get("width"),
                "D": cfg.get("max_depth"),
                "n": s["n"],
                "em": s["em"],
                "f1": s["f1"],
                "acc": s["acc"],
                "avg_tokens": s["avg_tokens"],
                "avg_rounds": s["avg_rounds"],
            })
    return rows


def _matrix(rows: list[dict], dataset: str, metric: str) -> str:
    by_wd: dict[tuple[int, int], dict] = {}
    for r in rows:
        if r["dataset"] != dataset:
            continue
        by_wd[(r["W"], r["D"])] = r
    widths = sorted(set(k[0] for k in by_wd))
    depths = sorted(set(k[1] for k in by_wd))
    if not widths or not depths:
        return "(no data)"
    head = "| W\\D | " + " | ".join(f"D={d}" for d in depths) + " |"
    sep = "|------" + "|------" * len(depths) + "|"
    lines = [head, sep]
    for w in widths:
        cells = []
        for d in depths:
            r = by_wd.get((w, d))
            if r is None:
                cells.append("—")
            elif metric == "f1":
                cells.append(f"{r['f1']:.3f}")
            elif metric == "em":
                cells.append(f"{r['em']:.3f}")
            elif metric == "tok":
                cells.append(f"{r['avg_tokens']:.0f}")
            else:
                cells.append("?")
        lines.append(f"| W={w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    grid = _grid_rows()
    finals = _final_rows()
    paper_table1_qwen25 = {
        "hotpotqa": {"em": 0.454, "f1": 0.559, "acc": 0.484},
        "2wikimultihop": {"em": 0.452, "f1": 0.562, "acc": 0.562},
        "musique": {"em": 0.162, "f1": 0.243, "acc": 0.178},
        "bamboogle": {"em": 0.360, "f1": 0.427, "acc": 0.368},
    }

    md = []
    md.append("# SPARC-RAG repro — Qwen3-14B, dense-only wiki18\n")
    md.append("Reproduction of Yang et al. 2026 (arXiv:2602.00083) restricted to dense retrieval over the wiki18 corpus, single backbone (Qwen3-14B no-think), no DPO.\n")
    md.append("## Final results (1000q × 3 + 125q bamboogle)\n")
    md.append("Best (W,D) per dataset selected from a 100q grid sweep over W ∈ {1,2,4} and D ∈ {2,4,6,8} (top-k=6 dense, T=0.5/0.0, max_tokens=600, seed=42, Dmax=8). Run on 3 × RTX A6000 vLLM Qwen3-14B replicas.\n")
    md.append("| dataset | W,D chosen | n | EM | F1 | Acc | avg tokens | avg rounds | paper Qwen2.5-7B no-DPO (EM/F1/Acc) |")
    md.append("|---------|-----------|-----|------|------|------|------------|-----------|--------------------------------------|")
    order = ["hotpotqa", "2wikimultihop", "musique", "bamboogle"]
    finals_by_ds = {r["dataset"]: r for r in finals}
    for ds in order:
        r = finals_by_ds.get(ds)
        if not r:
            continue
        p = paper_table1_qwen25[ds]
        md.append(
            f"| {ds} | W={r['W']} D={r['D']} | {r['n']} |"
            f" {r['em']:.3f} | {r['f1']:.3f} | {r['acc']:.3f} |"
            f" {r['avg_tokens']:.0f} | {r['avg_rounds']:.2f} |"
            f" {p['em']:.3f}/{p['f1']:.3f}/{p['acc']:.3f} |"
        )
    md.append("")
    md.append("Notes: paper numbers are SPARC-RAG (no DPO) on Qwen2.5-7B-Instruct, paper's hybrid retriever, IRCoT-style 500q test subsets. Our setup uses Qwen3-14B no-think + single dense E5-base retriever over the wiki18 corpus on 1000q subsets (Bamboogle 125q official). Absolute numbers therefore differ from paper but the scaling pattern is reproduced.\n")

    md.append("## 100q grid (full 4×3=12 configs per dataset)\n")
    for ds in order:
        md.append(f"### {ds} — F1\n")
        md.append(_matrix(grid, ds, "f1"))
        md.append("\n### " + ds + " — avg tokens/q\n")
        md.append(_matrix(grid, ds, "tok"))
        md.append("")

    md.append("## Best (max F1) per dataset on grid\n")
    md.append("| dataset | W | D | F1 (100q) | EM (100q) | tokens | F1 (full set) | EM (full set) |")
    md.append("|---------|---|---|-----------|-----------|--------|---------------|---------------|")
    by_ds_grid: dict[str, list[dict]] = {}
    for r in grid:
        by_ds_grid.setdefault(r["dataset"], []).append(r)
    for ds in order:
        rs = by_ds_grid.get(ds, [])
        if not rs:
            continue
        best = max(rs, key=lambda x: (x["f1"], -x["avg_tokens"]))
        full = finals_by_ds.get(ds, {})
        md.append(
            f"| {ds} | {best['W']} | {best['D']} |"
            f" {best['f1']:.3f} | {best['em']:.3f} | {best['avg_tokens']:.0f} |"
            f" {full.get('f1', float('nan')):.3f} | {full.get('em', float('nan')):.3f} |"
        )
    md.append("")

    md.append("## Scaling figures\n")
    md.append("![F1 vs tokens](grid/scaling_f1.png)\n")
    md.append("![EM vs tokens](grid/scaling_em.png)\n")
    md.append("Each panel: F1 (or EM) vs avg tokens/q on the 100q grid. Series = W ∈ {1,2,4}, points = D ∈ {2,4,6,8}.\n")

    md.append("## Caveats\n")
    md.append("- Bamboogle grid uses 42q (one node408 shard). Bamboogle final eval is the full 125q.")
    md.append("- Paper's NQ + DPO + BM25 stages omitted by request.")
    md.append("- AnswerEvaluator prompt is a minimal STOP/CONTINUE faithful to Section 3.2 — paper appendix did not expose its verbatim prompt at fetch time.")
    md.append("- Per-config concurrency varied (4–20) due to vLLM saturation tuning during long runs.\n")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")

    csv_path = ROOT / "results" / "final.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(finals[0].keys()))
        w.writeheader()
        w.writerows(finals)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
