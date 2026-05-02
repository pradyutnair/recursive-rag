"""GEPA metric: composite Pareto reward + oracle-routing bonus + targeted feedback."""
from __future__ import annotations

import json
from typing import Any, Callable

import dspy

from recrag.metric import composite_reward_with_oracle, feedback_text
from recrag.oracle import OracleEntry, OracleLookup


def _oracle_feedback(entry: OracleEntry | None, em: float, topology: str, total_tokens: int, reason: str) -> str:
    if entry is None:
        return ""
    sas = "solved" if entry.em == 1 else "failed"
    return (
        f"[ORACLE] naive_rag (Qwen3-14B no-think SAS lane) {sas} this question in "
        f"{entry.tokens} tokens. Yours: topology={topology or 'unknown'}, "
        f"tokens={total_tokens}, em={em}. {reason}"
    )


def make_metric(oracle: OracleLookup | None = None) -> Callable:
    """Build a GEPA metric closure with optional oracle-routing reward shaping."""

    def metric(gold: Any, pred: Any, trace: Any = None, pred_name: str | None = None, pred_trace: Any = None):
        gold_answer = str(getattr(gold, "answer", "") if not isinstance(gold, dict) else gold.get("answer", ""))
        gold_id = str(getattr(gold, "id", "") if not isinstance(gold, dict) else gold.get("id", ""))
        gold_dataset = str(getattr(gold, "dataset", "") if not isinstance(gold, dict) else gold.get("dataset", ""))
        pred_answer = str(getattr(pred, "answer", "") if not isinstance(pred, dict) else pred.get("answer", ""))
        meta = getattr(pred, "metadata", None) if not isinstance(pred, dict) else pred.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        findings = meta.get("findings", []) if isinstance(meta, dict) else []
        total_tokens = int(meta.get("total_tokens", 0))
        expected_type = meta.get("expected_type", "auto")
        topology = meta.get("topology", "")

        oracle_entry = oracle.get(gold_id) if oracle and gold_id else None
        oracle_easy = (oracle_entry.em == 1) if oracle_entry else None
        naive_tokens = oracle_entry.tokens if oracle_entry else 0

        rb, score, oracle_reason = composite_reward_with_oracle(
            pred_answer, gold_answer, findings, total_tokens, expected_type,
            topology=topology, oracle_easy=oracle_easy, naive_tokens=naive_tokens,
        )

        # Build feedback
        fb_parts: list[str] = [feedback_text(rb)]
        if pred_name == "planner":
            n_nodes = int(meta.get("n_nodes", 0))
            expected = int(meta.get("expected_hops_for_profile", 2))
            if n_nodes > expected + 1:
                fb_parts.append(f"Planner over-decomposed: {n_nodes} nodes for a profile expecting ~{expected} hops.")
            if n_nodes < expected:
                fb_parts.append(f"Planner under-decomposed: only {n_nodes} node(s) for a profile expecting ~{expected} hops.")
            if topology == "single_hop" and expected >= 2:
                fb_parts.append("Single-hop chosen for a multi-hop question.")
        elif pred_name == "synthesizer":
            if rb.grounded < 0.5:
                fb_parts.append("Synthesizer is not grounding the final span in cited findings.")
            if rb.shape < 0.5:
                fb_parts.append("Final answer shape does not match expected_type.")

        oracle_str = _oracle_feedback(oracle_entry, rb.em, topology, total_tokens, oracle_reason)
        if oracle_str:
            fb_parts.append(oracle_str)
        if gold_dataset:
            fb_parts.append(f"[dataset={gold_dataset}]")

        return dspy.Prediction(score=float(score), feedback=" ".join(fb_parts))

    return metric


# Default no-oracle metric for backwards compat
metric = make_metric(oracle=None)
