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
        # Map our dataset names to naive_<name> directories
        alias = {"2wikimultihop": "2wiki", "hotpotqa": "hotpotqa", "musique": "musique", "bamboogle": "bamboogle"}.get(ds, ds)
        p = base / f"naive_{alias}" / "predictions.jsonl"
        if p.exists():
            paths.append(p)
    if not paths:
        return None
    return OracleLookup.from_paths(paths)


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
            ).with_inputs("question")
            out.append(ex)
        return out

    examples = _to_examples(train_rows)
    val_examples = _to_examples(val_rows) if val_rows is not None else None

    oracle = _load_oracle(args)
    if oracle:
        print(f"[oracle] loaded {len(oracle)} entries; stats={oracle.stats()}")
    metric_fn = make_metric(oracle=oracle) if oracle else default_metric

    gepa_kwargs = {"metric": metric_fn, "reflection_lm": make_lm(args.reflection_lm, max_tokens=args.reflection_max_tokens), "num_threads": args.num_threads}
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
    p.add_argument("--oracle-naive-dir", default="results/baselines/wiki18-corpus/qwen3-14b-no-think/qwen3_14b_nothink_top5_node408")
    p.add_argument("--oracle-datasets", default="musique,2wikimultihop,hotpotqa")
    p.add_argument("--root-lm", default="qwen14b-think")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=4096)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--reflection-lm", default="qwen14b-think")
    p.add_argument("--reflection-max-tokens", type=int, default=2048)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--experience-library")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=1)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--auto", choices=["light", "medium", "heavy"], default="medium")
    p.add_argument("--max-metric-calls", type=int, default=0, help="If >0 overrides auto")
    p.add_argument("--num-threads", type=int, default=6)
    p.add_argument("--out", default="compiled/gepa_adaptive_v1.json")
    return p


def main() -> None:
    compile_gepa(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
