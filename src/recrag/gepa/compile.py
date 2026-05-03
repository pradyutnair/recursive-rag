"""GEPA optimizer over the AdaptiveProgram (planner + synthesizer modules)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dspy

from recrag.gepa.metric import make_metric, metric as default_metric
from recrag.lm import make_lm
from recrag.oracle import OracleLookup
from recrag.pipeline import PipelineConfig, ReactRagPipeline
from recrag.program import AdaptiveProgram, ReactProgram
from recrag.retriever import Retriever
from recrag.sync_pipeline import AdaptiveConfig, SyncAdaptivePipeline


def _load_oracle(args: argparse.Namespace) -> OracleLookup | None:
    if not args.oracle_naive_dir:
        return None
    base = Path(args.oracle_naive_dir)
    paths: list[Path] = []
    for ds in args.oracle_datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        # Try fresh SAS-mode oracle first, then the legacy FlashRAG naive baselines
        for candidate in (base / f"{ds}_fresh_naive" / "predictions.jsonl",
                          base / f"naive_{ds}" / "predictions.jsonl",
                          base / f"naive_{ {'2wikimultihop': '2wiki'}.get(ds, ds) }" / "predictions.jsonl"):
            if candidate.exists():
                paths.append(candidate)
                break
    if not paths:
        return None
    return OracleLookup.from_paths(paths)


def _load_prompts(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    raw_prompts = obj.get("prompts")
    if isinstance(raw_prompts, dict):
        return {str(k): str(v) for k, v in raw_prompts.items()}
    prompts: dict[str, str] = {}
    for name, value in obj.items():
        if isinstance(value, str):
            prompts[str(name)] = value
        elif isinstance(value, dict):
            sig = value.get("signature")
            if isinstance(sig, dict) and isinstance(sig.get("instructions"), str):
                prompts[str(name)] = sig["instructions"]
    return prompts


def compile_gepa(args: argparse.Namespace) -> None:
    train_rows = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    val_rows = json.loads(Path(args.valset).read_text(encoding="utf-8")) if args.valset else None
    if args.n_train and args.n_train > 0:
        train_rows = train_rows[: args.n_train]
    if val_rows is not None and args.n_val and args.n_val > 0:
        val_rows = val_rows[: args.n_val]

    root_lm = make_lm(args.root_lm, max_tokens=args.root_max_tokens)
    sub_lm = make_lm(args.sub_lm, max_tokens=args.sub_max_tokens)
    retriever = Retriever(args.retriever_url)
    seed_prompts = _load_prompts(args.seed_program)

    if args.program == "react":
        pipeline = ReactRagPipeline(root_lm, sub_lm, retriever, PipelineConfig(max_iters=args.max_iters, experience_library=args.experience_library))
        program = ReactProgram(pipeline)
    else:
        pipeline = SyncAdaptivePipeline(
            root_lm, sub_lm, retriever,
            AdaptiveConfig(
                max_nodes=args.max_nodes,
                max_recursion_depth=args.max_recursion,
                tau_recurse=args.tau_recurse,
                experience_library=args.experience_library,
                budget_hint=args.budget_hint,
                max_critic_retries=args.max_critic_retries,
                max_searches=args.max_searches,
                router_instructions=seed_prompts.get("router", AdaptiveConfig.router_instructions),
                planner_instructions=seed_prompts.get("planner", AdaptiveConfig.planner_instructions),
                synth_instructions=seed_prompts.get("synthesizer", AdaptiveConfig.synth_instructions),
                critic_instructions=seed_prompts.get("critic", AdaptiveConfig.critic_instructions),
            ),
        )
        program = AdaptiveProgram(pipeline)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(dspy, "GEPA"):
        out.write_text(json.dumps({"status": "skipped", "reason": "dspy.GEPA unavailable"}, indent=2), encoding="utf-8")
        return

    def _to_examples(rows: list[dict]) -> list[dspy.Example]:
        out = []
        for q in rows:
            ex = dspy.Example(
                question=str(q.get("question", "")),
                answer=str(q.get("answer", "")),
                id=str(q.get("id", "")),
                dataset=str(q.get("dataset", "")),
                budget_hint=str(q.get("budget_hint", args.budget_hint)),
            ).with_inputs("question", "budget_hint")
            out.append(ex)
        return out

    examples = _to_examples(train_rows)
    val_examples = _to_examples(val_rows) if val_rows is not None else None

    oracle = _load_oracle(args)
    if oracle:
        print(f"[oracle] loaded {len(oracle)} entries; stats={oracle.stats()}")
    metric_fn = make_metric(oracle=oracle) if oracle else default_metric

    gepa_kwargs = {"metric": metric_fn, "reflection_lm": make_lm(args.reflection_lm, max_tokens=args.reflection_max_tokens), "num_threads": args.num_threads, "track_stats": True}
    if args.log_dir:
        log_path = Path(args.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        gepa_kwargs["log_dir"] = str(log_path)
    if args.max_metric_calls and args.max_metric_calls > 0:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["auto"] = args.auto
    optimizer = dspy.GEPA(**gepa_kwargs)
    compiled = optimizer.compile(program, trainset=examples, valset=val_examples)
    try:
        compiled.save(str(out))
    except Exception:
        out.write_text(json.dumps({"status": "compiled", "note": "compiled object has no save()"}, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--program", choices=["adaptive", "react"], default="adaptive")
    p.add_argument("--questions", default="data/multidataset/train_v1.json")
    p.add_argument("--valset", default="data/multidataset/val_v1.json")
    p.add_argument("--n-train", type=int, default=0, help="0 = use all rows in --questions")
    p.add_argument("--n-val", type=int, default=0, help="0 = use all rows in --valset")
    p.add_argument("--oracle-naive-dir", default="compiled/oracle")
    p.add_argument("--oracle-datasets", default="musique,2wikimultihop,hotpotqa")
    p.add_argument("--root-lm", default="qwen14b-nothink")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=768)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--reflection-lm", default="qwen14b-nothink")
    p.add_argument("--reflection-max-tokens", type=int, default=2048)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--experience-library")
    p.add_argument("--seed-program", default="", help="JSON prompts file used to initialize router/planner/synthesizer/critic")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=0)
    p.add_argument("--max-critic-retries", type=int, default=0)
    p.add_argument("--max-searches", type=int, default=3)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--budget-hint", choices=["tight", "normal", "rich"], default="normal")
    p.add_argument("--auto", choices=["light", "medium", "heavy"], default="medium")
    p.add_argument("--max-metric-calls", type=int, default=0, help="If >0 overrides auto")
    p.add_argument("--log-dir", default="", help="If set, GEPA writes detailed logs and supports resume from this dir")
    p.add_argument("--num-threads", type=int, default=6)
    p.add_argument("--out", default="compiled/gepa_adaptive_v1.json")
    return p


def main() -> None:
    compile_gepa(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
