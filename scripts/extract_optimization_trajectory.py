"""Extract GEPA/GRPO optimization trajectory summaries from local logs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean


def main() -> None:
    root = Path(".")
    outdir = root / "results/analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = root / "compiled/gepa_v4_logs/run.log"
    log = log_path.read_text(errors="ignore") if log_path.exists() else ""
    gepa = [
        {"iteration": int(m.group(1)), "val_score": float(m.group(2))}
        for m in re.finditer(r"Iteration (\d+): Valset score for new program: ([\-0-9.]+)", log)
    ]
    base_match = re.search(r"Iteration 0: Base program full valset score: ([\-0-9.]+)", log)
    base = float(base_match.group(1)) if base_match else None
    best = [
        {"iteration": int(m.group(1)), "best_score": float(m.group(2))}
        for m in re.finditer(r"Iteration (\d+): Best score on valset: ([\-0-9.]+)", log)
    ]
    url = "https://wandb.ai/msc-thesis-pradyut/recrag-gepa/runs/38a33711" if "38a33711" in log else ""

    trace = root / "results/grpo_logs/v4_cand13/compile_trace.jsonl"
    rollouts = []
    if trace.exists():
        for line in trace.read_text().splitlines():
            if line.strip():
                try:
                    rollouts.append(json.loads(line))
                except Exception:
                    pass

    groups: dict[tuple[object, object], list[dict]] = {}
    for row in rollouts:
        groups.setdefault((row.get("epoch"), row.get("id")), []).append(row)

    group_rows = []
    for (epoch, qid), rows in groups.items():
        rewards = [
            float(r.get("reward", {}).get("composite", 0.0)) + float(r.get("oracle_bonus", 0.0))
            for r in rows
        ]
        ems = [float(r.get("reward", {}).get("em", 0.0)) for r in rows]
        group_rows.append(
            {
                "epoch": epoch,
                "question_id": qid,
                "n": len(rows),
                "profile": rows[0].get("profile"),
                "difficulty": rows[0].get("difficulty"),
                "mean_reward": round(mean(rewards), 4),
                "best_reward": round(max(rewards), 4),
                "worst_reward": round(min(rewards), 4),
                "mixed_em": len(set(ems)) > 1,
            }
        )

    ckpts = []
    for path in sorted((root / "compiled/grpo_v4_cand13_checkpoints").glob("epoch_0_batch_*.json")):
        obj = json.loads(path.read_text())
        ckpts.append({"checkpoint": str(path), "library_size": len(obj.get("entries", [])), "next_num": obj.get("next_num")})

    summary = {
        "gepa": {
            "wandb_run": url,
            "base_val_score": base,
            "num_val_evaluations": len(gepa),
            "best_logged_score": max([x["val_score"] for x in gepa], default=None),
            "last_best_score": best[-1]["best_score"] if best else None,
            "trajectory": gepa,
            "best_score_trace": best,
            "artifact_note": "W&B metrics synced; artifact upload failed because WANDB_DATA_DIR was not set for artifact staging in this run.",
        },
        "grpo": {
            "num_rollouts": len(rollouts),
            "num_question_groups": len(group_rows),
            "groups_with_mixed_em": sum(1 for g in group_rows if g["mixed_em"]),
            "mean_group_best_reward": round(mean([g["best_reward"] for g in group_rows]), 4) if group_rows else None,
            "library_checkpoints": ckpts,
            "group_rows": group_rows,
            "stopped_note": "Full cand13 GRPO stopped at epoch_0_batch_3 after OpenAI reflection stalled; partial library matched val EM but increased tokens.",
        },
    }
    (outdir / "optimization_trajectory_20260504.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Optimization Trajectory Snapshot",
        "",
        f"GEPA W&B run: {url}",
        f"GEPA base val score: {base}; best logged new-program val score: {summary['gepa']['best_logged_score']}; final best score trace: {summary['gepa']['last_best_score']}.",
        "",
        "| GEPA iter | val score | best-so-far trace |",
        "|---:|---:|---:|",
    ]
    best_by_iter = {x["iteration"]: x["best_score"] for x in best}
    step = max(1, len(gepa) // 12) if gepa else 1
    seen: set[int] = set()
    for item in gepa[::step] + ([gepa[-1]] if gepa else []):
        if item["iteration"] in seen:
            continue
        seen.add(item["iteration"])
        best_val = best_by_iter.get(item["iteration"])
        best_str = f"{best_val:.4f}" if best_val is not None else ""
        lines.append(f"| {item['iteration']} | {item['val_score']:.4f} | {best_str} |")
    lines += [
        "",
        f"GRPO rollouts: {len(rollouts)} over {len(group_rows)} question groups; mixed-EM groups: {summary['grpo']['groups_with_mixed_em']}.",
        "| GRPO checkpoint | library size |",
        "|---|---:|",
    ]
    for ckpt in ckpts:
        lines.append(f"| {ckpt['checkpoint']} | {ckpt['library_size']} |")
    (outdir / "optimization_trajectory_20260504.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gepa_points": len(gepa), "grpo_rollouts": len(rollouts), "checkpoints": ckpts}, indent=2))


if __name__ == "__main__":
    main()
