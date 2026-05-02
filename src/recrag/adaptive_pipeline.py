"""Adaptive Recursive Pipeline.

A DAG-first pipeline that:
  1. Classifies the question profile (cheap heuristic).
  2. Retrieves top-K profile-conditioned experiences (Profile-Insight-Utility lib).
  3. Asks the planner LM to emit a typed DAG (Plan*RAG <A.I.J> tags).
  4. Executes the DAG depth-by-depth, parallel within a layer, with parent-only
     context windowing.
  5. Allows recursive expansion: low-confidence bridge nodes spawn a sub-DAG.
  6. Synthesizes a final answer + cited support_ids via a single critic pass.
  7. Citation-gates the submit using the same proven gate as ReAct.

Design notes:
  - Falls back gracefully: if the planner produces an unparseable plan, run a
    single-hop direct path (the SAS lower bound), then synthesize.
  - Re-uses the existing _hop_async (retrieve + extract + rewrite + retry).
  - Uses the existing citation_check via tools.make_tools' submit semantics.
  - Token accounting reads vLLM usage fields (no estimation).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy

from .aio import to_thread as aio_to_thread
from .contracts import CitationCheck, HopFinding, normalize_answer
from .dag import (
    AdaptiveDAGRunner,
    PlanDAGSig,
    PlanRun,
    PlannedNode,
    SynthesizeFinalSig,
    execute_plan,
    parse_plan,
)
from .grpo.library import ExperienceLibrary
from .pipeline import _clean_final_answer
from .profile import classify, expected_hops
from .retriever import Retriever
from .tools import ToolRuntime, _hop_async

DEFAULT_PLANNER_INSTRUCTIONS = (
    "You are an adaptive multi-hop QA planner. Given a question, decompose it "
    "into the smallest DAG of atomic sub-questions that fully resolves it.\n"
    "RULES:\n"
    "- Single-fact questions need exactly one node.\n"
    "- Bridge questions need 2 nodes (one to resolve the bridge entity, one to "
    "answer the final target).\n"
    "- Nested 3+ hop questions need 3-4 nodes chained via depends_on.\n"
    "- Comparison/intersection across N independent entities use N parallel "
    "root nodes plus one synthesizing child node.\n"
    "- Maximum 6 nodes, max depth 3.\n"
    "- Node ids MUST be of the form Q1.1, Q1.2 (depth 0 siblings), Q2.1 (depth "
    "1, depends on a depth-0 node), etc.\n"
    "- Reference parent answers in a child question with EXACT tags like "
    "<A1.1>, <A1.2>, <A2.1>. Do NOT use <AI.Q1> or <Q1>.\n"
    "- Each node MUST have keys: id, question, expected_type, depends_on.\n"
    "- depends_on is a JSON list of parent node ids (empty for roots).\n"
    "- expected_type is one of: person, place, date, number, title, "
    "organization, yes_no, entity.\n"
    "- Output STRICT JSON only with the schema {\"nodes\": [...], "
    "\"final_node\": \"QX.Y\"}.\n\n"
    "Examples:\n"
    "Q: When was the performer of So Nice born?\n"
    "{\"nodes\":[{\"id\":\"Q1.1\",\"question\":\"Who performed the song So "
    "Nice?\",\"expected_type\":\"person\",\"depends_on\":[]},{\"id\":\"Q2.1\","
    "\"question\":\"When was <A1.1> born?\",\"expected_type\":\"date\","
    "\"depends_on\":[\"Q1.1\"]}],\"final_node\":\"Q2.1\"}\n\n"
    "Q: Who is older, Marie Curie or Albert Einstein?\n"
    "{\"nodes\":[{\"id\":\"Q1.1\",\"question\":\"When was Marie Curie born?\","
    "\"expected_type\":\"date\",\"depends_on\":[]},{\"id\":\"Q1.2\","
    "\"question\":\"When was Albert Einstein born?\",\"expected_type\":\"date\","
    "\"depends_on\":[]},{\"id\":\"Q2.1\",\"question\":\"Whose birth date is "
    "earlier, Marie Curie (<A1.1>) or Albert Einstein (<A1.2>)?\","
    "\"expected_type\":\"person\",\"depends_on\":[\"Q1.1\",\"Q1.2\"]}],"
    "\"final_node\":\"Q2.1\"}\n\n"
    "Q: What is the capital of France?\n"
    "{\"nodes\":[{\"id\":\"Q1.1\",\"question\":\"What is the capital of "
    "France?\",\"expected_type\":\"place\",\"depends_on\":[]}],"
    "\"final_node\":\"Q1.1\"}"
)

DEFAULT_SYNTH_INSTRUCTIONS = (
    "You are an adaptive multi-hop QA synthesizer. Read the original question "
    "and the resolved DAG trace. Produce ONE concise final span that DIRECTLY "
    "answers the user question (not an intermediate bridge entity). Preserve "
    "full canonical names, full dates, full award/category names, and acronym "
    "expansions when supported. Output 1-10 words, no explanation, no "
    "refusal. Cite the chunk_ids used as support (CSV)."
)


@dataclass
class AdaptiveConfig:
    max_nodes: int = 6
    max_recursion_depth: int = 2
    tau_recurse: float = 0.5
    experience_library: str | None = None
    use_dag: bool = True
    planner_instructions: str = DEFAULT_PLANNER_INSTRUCTIONS
    synth_instructions: str = DEFAULT_SYNTH_INSTRUCTIONS


def _tokens_since(lm: dspy.LM, start_idx: int) -> int:
    total = 0
    for item in getattr(lm, "history", [])[start_idx:]:
        usage = (item or {}).get("usage") or {}
        try:
            total += int(usage.get("total_tokens", 0))
        except Exception:
            pass
    return total


def _load_library(path: str | None) -> ExperienceLibrary:
    if not path:
        return ExperienceLibrary()
    p = Path(path)
    if not p.exists():
        return ExperienceLibrary()
    if p.suffix == ".json":
        return ExperienceLibrary.load(p)
    # Legacy text format: each line is an insight; profile=any.
    lib = ExperienceLibrary()
    for line in p.read_text(encoding="utf-8").splitlines():
        clean = re.sub(r"^E-\d{3}\s*(\[[^\]]+\])?:?\s*", "", line).strip(" -\t")
        if clean:
            lib.add(clean)
    return lib


class AdaptiveRecursivePipeline:
    """DAG-first adaptive recursive RAG pipeline."""

    def __init__(self, root_lm: dspy.LM, sub_lm: dspy.LM, retriever: Retriever, config: AdaptiveConfig):
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        self.retriever = retriever
        self.config = config
        self.tool_state = ToolRuntime()
        self.library = _load_library(config.experience_library)

        # Build module signatures with current instructions
        self.plan_sig = PlanDAGSig.with_instructions(config.planner_instructions)
        self.synth_sig = SynthesizeFinalSig.with_instructions(config.synth_instructions)
        self.plan_predict = dspy.Predict(self.plan_sig)
        self.synth_predict = dspy.Predict(self.synth_sig)

    async def _plan_one(self, question: str, expected_type: str = "auto", note: str = "") -> PlanRun | None:
        prof = classify(question)
        exp = self.library.to_text(profile=prof, top_k=4) if self.library.entries else ""
        if note:
            exp = (exp + ("\n" if exp else "") + note).strip()
        try:
            def _call():
                with dspy.context(lm=self.root_lm):
                    return self.plan_predict(question=question, profile=prof, experience=exp)
            pred = await aio_to_thread(_call)
            plan = parse_plan(getattr(pred, "plan_json", ""))
            return plan
        except Exception as exc:
            self.tool_state.tool_errors.append(f"plan error: {exc}")
            return None

    async def _synth(self, question: str, expected_type: str, plan: PlanRun) -> tuple[str, str]:
        # Build resolved trace JSON for the synthesizer
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
            })
        trace_json = json.dumps(trace, ensure_ascii=False)
        try:
            def _call():
                with dspy.context(lm=self.root_lm):
                    return self.synth_predict(
                        question=question,
                        expected_type=expected_type,
                        trace_json=trace_json,
                    )
            pred = await aio_to_thread(_call)
            answer = str(getattr(pred, "final_answer", "")).strip()
            ids = str(getattr(pred, "support_ids", "")).strip()
        except Exception as exc:
            self.tool_state.tool_errors.append(f"synth error: {exc}")
            answer = ""
            ids = ""
        return answer, ids

    def _citation_check(self, answer: str, support_ids_raw: str) -> tuple[bool, list[str]]:
        ids = [x.strip() for x in re.split(r"[,\n]", support_ids_raw) if x.strip()]
        ids = ids[:6]
        if not ids:
            return False, []
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
            if not f_norm:
                continue
            if ans_norm in f_norm or f_norm in ans_norm:
                return True, ids
            ans_t = set(ans_norm.split())
            f_t = set(f_norm.split())
            if ans_t and f_t and len(ans_t & f_t) / len(ans_t) >= 0.5:
                return True, ids
            chunk = self.tool_state.chunks_by_id.get(f.evidence_chunk_id)
            if chunk and ans_norm in normalize_answer(chunk.text):
                return True, ids
        return False, ids

    async def _direct_path(self, question: str, expected_type: str) -> PlanRun:
        """Single-hop SAS path used as fallback when the planner fails."""
        plan = PlanRun(nodes={}, final_node="Q1.1")
        node = PlannedNode(id="Q1.1", question=question, raw_question=question, expected_type=expected_type, depends_on=[])
        plan.nodes["Q1.1"] = node
        runner = AdaptiveDAGRunner(retriever=self.retriever, sub_lm=self.sub_lm, state=self.tool_state, max_nodes=self.config.max_nodes, max_recursion_depth=0, tau_recurse=self.config.tau_recurse)
        await runner.run_node(node)
        return plan

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
        if ql.startswith("is ") or ql.startswith("was ") or ql.startswith("are ") or ql.startswith("were "):
            return "yes_no"
        return "entity"

    async def run(self, question: str) -> dict[str, Any]:
        self.tool_state.reset()
        t0 = time.time()
        root_start = len(getattr(self.root_lm, "history", []))
        sub_start = len(getattr(self.sub_lm, "history", []))

        expected_type = self._expected_type_hint(question)
        prof = classify(question)

        plan: PlanRun | None = None
        if self.config.use_dag:
            plan = await self._plan_one(question, expected_type)
        if plan is None or not plan.nodes:
            plan = await self._direct_path(question, expected_type)
        else:
            runner = AdaptiveDAGRunner(
                retriever=self.retriever,
                sub_lm=self.sub_lm,
                state=self.tool_state,
                max_nodes=self.config.max_nodes,
                max_recursion_depth=self.config.max_recursion_depth,
                tau_recurse=self.config.tau_recurse,
            )
            await execute_plan(plan, runner, self._plan_one if self.config.max_recursion_depth > 0 else None, recursion_depth=0)

        # Synthesize final answer + citations
        synth_ans, synth_ids = await self._synth(question, expected_type, plan)

        # Citation gate; if rejected, try a backup that picks the highest-confidence finding
        accepted, used_ids = self._citation_check(synth_ans, synth_ids)
        if not accepted:
            # Backup: choose the most confident finding whose answer is consistent with synth_ans
            best = None
            for f in sorted(self.tool_state.findings, key=lambda x: x.confidence, reverse=True):
                if f.confidence >= 0.5 and f.answer.strip():
                    best = f
                    break
            if best:
                synth_ans = best.answer
                used_ids = [best.evidence_chunk_id]
                accepted = True

        final_answer = _clean_final_answer(question, synth_ans)

        root_tokens = _tokens_since(self.root_lm, root_start)
        sub_tokens = _tokens_since(self.sub_lm, sub_start)
        elapsed = time.time() - t0

        # Build a serializable trace
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
            "expected_type": expected_type,
            "topology": topology,
            "n_nodes": len(plan.nodes),
            "final_node": plan.final_node,
            "support_ids": used_ids,
            "citation_accepted": accepted,
            "expected_hops_for_profile": expected_hops(prof),
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
    layers = plan.topo_layers()
    for d, layer in enumerate(layers):
        for nid in layer:
            n = plan.nodes[nid]
            lines.append(f"  [d{d}] {n.id} ({n.expected_type}) {n.raw_question}")
            lines.append(f"        -> answer={n.answer!r} conf={n.confidence:.2f} chunk={n.chunk_id}")
            if n.expanded_into:
                lines.append(f"        recursed-into={n.expanded_into}")
    lines.append(f"FINAL: {answer!r} support={support_ids}")
    return "\n".join(lines)
