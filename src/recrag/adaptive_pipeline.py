"""Force-hard no-critic recursive RAG base pipeline.

This is the frozen base method for future experiments:
profile -> planner -> DAG execution -> synthesizer -> citation gate -> final
answer cleanup.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import dspy

from .aio import to_thread as aio_to_thread
from .contracts import normalize_answer
from .dag import AdaptiveDAGRunner, PlanDAGSig, PlanRun, PlannedNode, SynthesizeFinalSig, execute_plan, parse_plan
from .pipeline import _clean_final_answer
from .profile import classify, expected_hops
from .retriever import Retriever
from .tools import ToolRuntime


DEFAULT_PLANNER_INSTRUCTIONS = (
    "You are a multi-hop QA planner. Decompose the question into the smallest "
    "DAG of atomic sub-questions that fully resolves the final answer.\n"
    "Rules:\n"
    "- Use one node only for genuinely single-hop questions.\n"
    "- Bridge questions need sequential nodes: resolve the bridge entity, then "
    "ask the final target question using <A1.1>-style parent-answer tags.\n"
    "- Comparisons or intersections use parallel root nodes plus a final child.\n"
    "- Use ids Q1.1, Q1.2 for roots; Q2.1, Q2.2 for children.\n"
    "- Maximum 6 nodes. Maximum 3 depth levels.\n"
    "- Each node has id, question, expected_type, depends_on.\n"
    "- expected_type is one of: person, place, date, number, title, "
    "organization, yes_no, entity.\n"
    "- Output strict JSON only: {\"nodes\": [...], \"final_node\": \"QX.Y\"}."
)

DEFAULT_SYNTH_INSTRUCTIONS = (
    "Read the original question and resolved DAG trace. Return one concise "
    "final answer span that directly answers the original question, not an "
    "intermediate bridge. Preserve full names, full dates, official titles, "
    "and precise units when supported. Also return support chunk_ids as CSV."
)


@dataclass
class AdaptiveConfig:
    max_nodes: int = 6
    max_recursion_depth: int = 0
    tau_recurse: float = 0.5
    experience_library: str | None = None
    use_dag: bool = True
    planner_instructions: str = DEFAULT_PLANNER_INSTRUCTIONS
    synth_instructions: str = DEFAULT_SYNTH_INSTRUCTIONS
    budget_hint: str = "normal"
    max_searches: int = 5


def _tokens_since(lm: dspy.LM, start_idx: int) -> int:
    total = 0
    for item in getattr(lm, "history", [])[start_idx:]:
        usage = (item or {}).get("usage") or {}
        try:
            total += int(usage.get("total_tokens", 0))
        except Exception:
            pass
    return total


class AdaptiveRecursivePipeline:
    """Hard-only DAG RAG pipeline.

    The class name is kept for CLI compatibility, but the runtime behavior is
    intentionally fixed to the force-hard no-critic base method.
    """

    def __init__(
        self,
        root_lm: dspy.LM,
        sub_lm: dspy.LM,
        retriever: Retriever,
        config: AdaptiveConfig,
        *,
        synth_lm: dspy.LM | None = None,
    ) -> None:
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        self.synth_lm = synth_lm or sub_lm
        self.retriever = retriever
        self.config = config
        self.tool_state = ToolRuntime()
        self.plan_sig = PlanDAGSig.with_instructions(config.planner_instructions)
        self.synth_sig = SynthesizeFinalSig.with_instructions(config.synth_instructions)
        self.plan_predict = dspy.Predict(self.plan_sig)
        self.synth_predict = dspy.Predict(self.synth_sig)

    def _norm_budget_hint(self, budget_hint: str | None) -> str:
        hint = str(budget_hint or self.config.budget_hint or "normal").strip().lower()
        return hint if hint in {"tight", "normal", "rich"} else "normal"

    async def _plan_one(self, question: str, expected_type: str = "auto", note: str = "", budget_hint: str = "normal") -> PlanRun | None:
        prof = classify(question)
        try:
            def _call() -> Any:
                with dspy.context(lm=self.root_lm):
                    return self.plan_predict(
                        question=question,
                        profile=prof,
                        experience=note,
                        budget_hint=budget_hint,
                    )

            pred = await aio_to_thread(_call)
            return parse_plan(getattr(pred, "plan_json", ""))
        except Exception as exc:
            self.tool_state.tool_errors.append(f"plan error: {exc}")
            return None

    def _trace_json(self, plan: PlanRun) -> str:
        trace = []
        for nid in sorted(plan.nodes):
            n = plan.nodes[nid]
            trace.append({
                "id": n.id,
                "question": n.question,
                "answer": n.answer,
                "confidence": round(n.confidence, 3),
                "chunk_id": n.chunk_id,
                "expected_type": n.expected_type,
                "depth": n.depth,
                "retrieval_query": n.retrieval_query,
            })
        return json.dumps(trace, ensure_ascii=False)

    async def _synth(self, question: str, expected_type: str, plan: PlanRun, budget_hint: str) -> tuple[str, str]:
        try:
            def _call() -> Any:
                with dspy.context(lm=self.synth_lm):
                    return self.synth_predict(
                        question=question,
                        expected_type=expected_type,
                        trace_json=self._trace_json(plan),
                        budget_hint=budget_hint,
                    )

            pred = await aio_to_thread(_call)
            return str(getattr(pred, "final_answer", "")).strip(), str(getattr(pred, "support_ids", "")).strip()
        except Exception as exc:
            self.tool_state.tool_errors.append(f"synth error: {exc}")
            return "", ""

    async def _execute(self, question: str, expected_type: str, budget_hint: str) -> PlanRun:
        plan = await self._plan_one(question, expected_type, budget_hint=budget_hint) if self.config.use_dag else None
        if plan is None or not plan.nodes:
            plan = PlanRun(nodes={}, final_node="Q1.1")
            plan.nodes["Q1.1"] = PlannedNode(
                id="Q1.1",
                question=question,
                raw_question=question,
                expected_type=expected_type,
                depends_on=[],
            )
        runner = AdaptiveDAGRunner(
            retriever=self.retriever,
            sub_lm=self.sub_lm,
            state=self.tool_state,
            max_nodes=self.config.max_nodes,
            max_recursion_depth=self.config.max_recursion_depth,
            tau_recurse=self.config.tau_recurse,
            max_searches=self.config.max_searches,
        )
        plan_one = (lambda q, et, n: self._plan_one(q, et, n, budget_hint)) if self.config.max_recursion_depth > 0 else None
        await execute_plan(plan, runner, plan_one, recursion_depth=0)
        return plan

    def _citation_check(self, answer: str, support_ids_raw: str) -> tuple[bool, list[str]]:
        ids = [x.strip() for x in re.split(r"[,\n]", support_ids_raw) if x.strip()][:6]
        known = {f.evidence_chunk_id for f in self.tool_state.findings if f.evidence_chunk_id}
        ids = [i for i in ids if i in known]
        if not ids:
            return False, []
        ans_norm = normalize_answer(answer)
        if not ans_norm:
            return False, ids
        for f in self.tool_state.findings:
            if f.evidence_chunk_id not in ids:
                continue
            f_norm = normalize_answer(f.answer)
            if f_norm and (ans_norm in f_norm or f_norm in ans_norm):
                return True, ids
            chunk = self.tool_state.chunks_by_id.get(f.evidence_chunk_id)
            if chunk and ans_norm in normalize_answer(chunk.text):
                return True, ids
        return False, ids

    def _expected_type_hint(self, question: str) -> str:
        ql = (question or "").lower()
        if any(x in ql for x in ["when", "what date", "what year", "year"]):
            return "date"
        if any(x in ql for x in ["how many", "how much", "rank", "number of"]):
            return "number"
        if ql.startswith("who"):
            return "person"
        if ql.startswith("where"):
            return "place"
        if ql.startswith(("is ", "was ", "are ", "were ")):
            return "yes_no"
        return "entity"

    async def run(self, question: str, budget_hint: str | None = None) -> dict[str, Any]:
        self.tool_state.reset()
        budget_hint = self._norm_budget_hint(budget_hint)
        t0 = time.time()
        root_start = len(getattr(self.root_lm, "history", []))
        sub_start = len(getattr(self.sub_lm, "history", []))

        expected_type = self._expected_type_hint(question)
        prof = classify(question)
        plan = await self._execute(question, expected_type, budget_hint)
        synth_ans, synth_ids = await self._synth(question, expected_type, plan, budget_hint)
        accepted, used_ids = self._citation_check(synth_ans, synth_ids)
        if not accepted:
            best = max(self.tool_state.findings, key=lambda f: f.confidence, default=None)
            if best and best.answer.strip() and best.evidence_chunk_id:
                synth_ans = best.answer
                used_ids = [best.evidence_chunk_id]
                accepted = True

        final_answer = _clean_final_answer(question, synth_ans)
        root_tokens = _tokens_since(self.root_lm, root_start)
        sub_tokens = _tokens_since(self.sub_lm, sub_start)
        elapsed = time.time() - t0

        node_dicts = [plan.nodes[nid].as_dict() for nid in sorted(plan.nodes)]
        topology = (
            "single_hop" if len(plan.nodes) == 1
            else "parallel_compare" if all(not plan.nodes[nid].depends_on for nid in plan.nodes) and len(plan.nodes) >= 2
            else f"dag_n{len(plan.nodes)}"
        )
        metadata = {
            "root_tokens": root_tokens,
            "sub_tokens": sub_tokens,
            "total_tokens": root_tokens + sub_tokens,
            "elapsed_s": round(elapsed, 3),
            "hops": self.tool_state.total_hops,
            "retries": self.tool_state.total_retries,
            "tool_errors": list(self.tool_state.tool_errors),
            "findings": [f.as_dict() for f in self.tool_state.findings],
            "profile": prof,
            "budget_hint": budget_hint,
            "route": "hard",
            "initial_route": "hard",
            "method": "force_hard_no_critic",
            "expected_type": expected_type,
            "topology": topology,
            "n_nodes": len(plan.nodes),
            "final_node": plan.final_node,
            "support_ids": used_ids,
            "citation_accepted": accepted,
            "expected_hops_for_profile": expected_hops(prof),
            "used_final_node_answer": False,
        }
        return {
            "question": question,
            "predicted_answer": final_answer,
            "answer": final_answer,
            "trajectory": {"plan_nodes": node_dicts, "synth_answer_raw": synth_ans},
            "metadata": metadata,
            "readable_trace": _readable_dag_trace(question, plan, final_answer, used_ids),
        }


def _readable_dag_trace(question: str, plan: PlanRun, answer: str, support_ids: list[str]) -> str:
    lines = [f"Q: {question}"]
    for d, layer in enumerate(plan.topo_layers()):
        for nid in layer:
            n = plan.nodes[nid]
            lines.append(f"  [d{d}] {n.id} ({n.expected_type}) {n.raw_question}")
            lines.append(f"        -> answer={n.answer!r} conf={n.confidence:.2f} chunk={n.chunk_id}")
    lines.append(f"FINAL: {answer!r} support={support_ids}")
    return "\n".join(lines)
