"""Build thesis analysis tables from completed evaluation runs.

This is intentionally read-only: it consumes prediction JSONL/summary JSON files
and writes compact JSON/Markdown reports under results/analysis.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["musique", "2wikimultihop", "hotpotqa", "bamboogle"]


DEFAULT_RUNS = {
    "adaptive_default": "results/runs/test_v4_default_20260504",
    "force_easy": "results/runs/test_forceeasy_20260504",
    "force_hard": "results/runs/test_forcehard_20260504",
    "no_critic": "results/runs/test_nocritic_20260504",
    "force_hard_no_critic": "results/runs/test_forcehard_nocritic_20260504",
    "random_route": "results/runs/test_randomroute_20260504",
    "no_oracle_gepa": "results/runs/test_nooracle_gepa_20260504",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def em(row: dict[str, Any]) -> float:
    return float(row.get("reward", {}).get("em", 0.0))


def tok(row: dict[str, Any]) -> float:
    return float(row.get("metadata", {}).get("total_tokens", 0.0))


def profile(row: dict[str, Any]) -> str:
    return str(row.get("source_profile") or row.get("metadata", {}).get("profile") or "unknown")


def collect_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_path = root / "results/diagnostics/baselines_rescored.json"
    if baseline_path.exists():
        for r in read_json(baseline_path):
            rows.append({
                "dataset": r["dataset"],
                "method": r["method"],
                "n": r["n"],
                "em": r["norm_em"],
                "f1": r["token_f1"],
                "contain": r["contain"],
                "mean_tokens": r["mean_tokens"],
                "source": str(baseline_path),
            })
    for label, rel in DEFAULT_RUNS.items():
        run_root = root / rel
        for ds in DATASETS:
            summary = run_root / ds / "summary.json"
            if not summary.exists():
                continue
            s = read_json(summary)
            rows.append({
                "dataset": ds,
                "method": label,
                "n": s.get("n", 0),
                "em": s.get("norm_em", s.get("em", 0.0)),
                "f1": s.get("token_f1", s.get("f1", 0.0)),
                "contain": s.get("contain", 0.0),
                "mean_tokens": s.get("mean_tokens", 0.0),
                "easy_route_fraction": s.get("easy_route_fraction", 0.0),
                "source": str(summary),
            })
    mark_pareto(rows)
    return rows


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    for ds in DATASETS:
        subset = [r for r in rows if r["dataset"] == ds]
        for r in subset:
            r["pareto"] = not any(
                q["em"] >= r["em"]
                and q["mean_tokens"] <= r["mean_tokens"]
                and (q["em"] > r["em"] or q["mean_tokens"] < r["mean_tokens"])
                for q in subset
            )


def scaling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in DATASETS:
        pts = [
            (math.log(float(r["mean_tokens"])), math.log(max(float(r["em"]), 1e-6)), r["method"])
            for r in rows
            if r["dataset"] == ds and float(r["mean_tokens"]) > 0
        ]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xbar = mean(xs)
        ybar = mean(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else 0.0
        intercept = ybar - slope * xbar
        out[ds] = {
            "slope_log_em_vs_log_tokens": round(slope, 4),
            "intercept": round(intercept, 4),
            "points": [{"method": m, "log_tokens": round(x, 4), "log_em": round(y, 4)} for x, y, m in pts],
        }
    return out


def route_impact(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    adaptive_root = root / DEFAULT_RUNS["adaptive_default"]
    easy_root = root / DEFAULT_RUNS["force_easy"]
    hard_root = root / DEFAULT_RUNS["force_hard"]
    for ds in DATASETS:
        ap = adaptive_root / ds / "predictions.jsonl"
        ep = easy_root / ds / "predictions.jsonl"
        hp = hard_root / ds / "predictions.jsonl"
        if not (ap.exists() and ep.exists() and hp.exists()):
            continue
        a = {str(r.get("id")): r for r in read_jsonl(ap)}
        e = {str(r.get("id")): r for r in read_jsonl(ep)}
        h = {str(r.get("id")): r for r in read_jsonl(hp)}
        buckets: dict[str, list[str]] = defaultdict(list)
        for qid, row in a.items():
            if qid in e and qid in h:
                buckets[str(row.get("metadata", {}).get("route", "unknown"))].append(qid)
        out[ds] = {}
        for route, ids in sorted(buckets.items()):
            out[ds][route] = {
                "n": len(ids),
                "adaptive_em": round(mean(em(a[i]) for i in ids), 4),
                "force_easy_em": round(mean(em(e[i]) for i in ids), 4),
                "force_hard_em": round(mean(em(h[i]) for i in ids), 4),
                "adaptive_tokens": round(mean(tok(a[i]) for i in ids), 1),
                "force_easy_tokens": round(mean(tok(e[i]) for i in ids), 1),
                "force_hard_tokens": round(mean(tok(h[i]) for i in ids), 1),
            }
    return out


def profile_table(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, rel in DEFAULT_RUNS.items():
        run_root = root / rel
        for ds in DATASETS:
            pred = run_root / ds / "predictions.jsonl"
            if not pred.exists():
                continue
            rows = read_jsonl(pred)
            by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_profile[profile(r)].append(r)
            out.setdefault(ds, {})[label] = {
                p: {
                    "n": len(rs),
                    "em": round(mean(em(r) for r in rs), 4),
                    "mean_tokens": round(mean(tok(r) for r in rs), 1),
                    "easy_route_fraction": round(sum(1 for r in rs if r.get("metadata", {}).get("route") == "easy") / len(rs), 4),
                }
                for p, rs in sorted(by_profile.items())
            }
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], route: dict[str, Any], scale: dict[str, Any]) -> None:
    lines = ["# Thesis Results Snapshot", ""]
    baseline_methods = {"naive", "ircot", "opera", "ma-rag"}
    ours_methods = {"adaptive_default", "no_critic", "force_hard_no_critic", "force_hard", "force_easy"}
    lines += ["## Best-vs-baseline summary", "", "| dataset | best ours | EM/tokens | best external | EM/tokens | result |", "|---|---|---:|---|---:|---|"]
    for ds in DATASETS:
        ours = [r for r in rows if r["dataset"] == ds and r["method"] in ours_methods]
        baselines = [r for r in rows if r["dataset"] == ds and r["method"] in baseline_methods]
        if not ours or not baselines:
            continue
        best_ours = max(ours, key=lambda r: (float(r["em"]), -float(r["mean_tokens"])))
        best_base = max(baselines, key=lambda r: (float(r["em"]), -float(r["mean_tokens"])))
        beats = float(best_ours["em"]) > float(best_base["em"]) and float(best_ours["mean_tokens"]) <= float(best_base["mean_tokens"])
        result = "beats at fewer/equal tokens" if beats else "not dominant"
        lines.append(
            f"| {ds} | {best_ours['method']} | {float(best_ours['em']):.3f}/{float(best_ours['mean_tokens']):.0f} | "
            f"{best_base['method']} | {float(best_base['em']):.3f}/{float(best_base['mean_tokens']):.0f} | {result} |"
        )
    lines.append("")
    for ds in DATASETS:
        lines += [f"## {ds}", "", "| method | EM | tokens | Pareto |", "|---|---:|---:|---|"]
        for r in sorted([x for x in rows if x["dataset"] == ds], key=lambda x: (x["mean_tokens"], x["method"])):
            lines.append(f"| {r['method']} | {float(r['em']):.3f} | {float(r['mean_tokens']):.1f} | {str(bool(r.get('pareto'))).lower()} |")
        if ds in route:
            lines += ["", "| adaptive route | n | adaptive EM | force-easy EM | force-hard EM | adaptive tokens |", "|---|---:|---:|---:|---:|---:|"]
            for route_name, r in route[ds].items():
                lines.append(
                    f"| {route_name} | {r['n']} | {r['adaptive_em']:.3f} | {r['force_easy_em']:.3f} | "
                    f"{r['force_hard_em']:.3f} | {r['adaptive_tokens']:.1f} |"
                )
        if ds in scale:
            lines += ["", f"Scaling slope log(EM)~log(tokens): `{scale[ds]['slope_log_em_vs_log_tokens']}`"]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _xy(v: float, lo: float, hi: float, size: int, pad: int, invert: bool = False) -> float:
    if hi <= lo:
        frac = 0.5
    else:
        frac = (v - lo) / (hi - lo)
    if invert:
        frac = 1.0 - frac
    return pad + frac * (size - 2 * pad)


def write_pareto_svgs(root: Path, rows: list[dict[str, Any]], out_dir: Path) -> None:
    colors = {
        "adaptive_default": "#d62728",
        "force_easy": "#2ca02c",
        "force_hard": "#9467bd",
        "ma-rag": "#1f77b4",
        "naive": "#111111",
        "ircot": "#ff7f0e",
        "opera": "#17becf",
        "random_route": "#8c564b",
        "no_critic": "#e377c2",
        "force_hard_no_critic": "#bcbd22",
        "no_oracle_gepa": "#7f7f7f",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        pts = [r for r in rows if r["dataset"] == ds]
        if not pts:
            continue
        width, height, pad = 760, 460, 70
        max_tok = max(float(r["mean_tokens"]) for r in pts) * 1.08
        max_em = max(float(r["em"]) for r in pts) * 1.15
        elems = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{ds}: EM vs mean tokens</text>',
            f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#222"/>',
            f'<line x1="{pad}" y1="{height-pad}" x2="{pad}" y2="{pad}" stroke="#222"/>',
            f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13">mean tokens</text>',
            f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">EM</text>',
        ]
        for tick in range(6):
            tx = max_tok * tick / 5
            x = _xy(tx, 0, max_tok, width, pad)
            elems.append(f'<line x1="{x:.1f}" y1="{height-pad}" x2="{x:.1f}" y2="{height-pad+5}" stroke="#222"/>')
            elems.append(f'<text x="{x:.1f}" y="{height-pad+20}" text-anchor="middle" font-family="Arial" font-size="10">{tx:.0f}</text>')
            ey = max_em * tick / 5
            y = _xy(ey, 0, max_em, height, pad, invert=True)
            elems.append(f'<line x1="{pad-5}" y1="{y:.1f}" x2="{pad}" y2="{y:.1f}" stroke="#222"/>')
            elems.append(f'<text x="{pad-9}" y="{y+3:.1f}" text-anchor="end" font-family="Arial" font-size="10">{ey:.2f}</text>')
        for r in pts:
            x = _xy(float(r["mean_tokens"]), 0, max_tok, width, pad)
            y = _xy(float(r["em"]), 0, max_em, height, pad, invert=True)
            c = colors.get(str(r["method"]), "#444444")
            stroke = "#111111" if r.get("pareto") else "none"
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{c}" stroke="{stroke}" stroke-width="2"/>')
            elems.append(f'<text x="{x+8:.1f}" y="{y-8:.1f}" font-family="Arial" font-size="11">{r["method"]}</text>')
        elems.append("</svg>")
        (out_dir / f"pareto_{ds}.svg").write_text("\n".join(elems), encoding="utf-8")


def write_scaling_svg(root: Path, rows: list[dict[str, Any]], out_path: Path) -> None:
    pts = [
        r for r in rows
        if float(r.get("mean_tokens", 0.0)) > 0 and float(r.get("em", 0.0)) > 0
    ]
    if not pts:
        return
    width, height, pad = 900, 520, 75
    xs = [math.log(float(r["mean_tokens"])) for r in pts]
    ys = [math.log(float(r["em"])) for r in pts]
    xmin, xmax = min(xs) * 0.98, max(xs) * 1.02
    ymin, ymax = min(ys) * 1.08, max(ys) * 0.92
    colors = {"musique": "#d62728", "2wikimultihop": "#1f77b4", "hotpotqa": "#2ca02c", "bamboogle": "#9467bd"}
    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">Inference-time scaling snapshot</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#222"/>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{pad}" y2="{pad}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13">log(mean tokens)</text>',
        f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">log(EM)</text>',
    ]
    for r in pts:
        x = _xy(math.log(float(r["mean_tokens"])), xmin, xmax, width, pad)
        y = _xy(math.log(float(r["em"])), ymin, ymax, height, pad, invert=True)
        c = colors.get(str(r["dataset"]), "#444444")
        elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c}"/>')
        if str(r["method"]) in {"adaptive_default", "force_easy", "force_hard", "ma-rag", "naive"}:
            elems.append(f'<text x="{x+6:.1f}" y="{y-6:.1f}" font-family="Arial" font-size="10">{r["dataset"]}:{r["method"]}</text>')
    elems.append("</svg>")
    out_path.write_text("\n".join(elems), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-json", default="results/analysis/thesis_results_snapshot.json")
    ap.add_argument("--out-md", default="results/analysis/thesis_results_snapshot.md")
    ap.add_argument("--plot-dir", default="results/analysis/plots")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect_runs(root)
    route = route_impact(root)
    profiles = profile_table(root)
    scale = scaling(rows)
    out = {"runs": rows, "route_impact": route, "profiles": profiles, "scaling": scale}
    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(out_md, rows, route, scale)
    plot_dir = root / args.plot_dir
    write_pareto_svgs(root, rows, plot_dir)
    write_scaling_svg(root, rows, plot_dir / "scaling_snapshot.svg")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
