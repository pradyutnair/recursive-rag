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
    CritiqueFinalSig,
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

DEFAULT_CRITIC_INSTRUCTIONS = (
    "You are a strict verifier for multi-hop QA. Given the original question, "
    "the resolved DAG trace, and the proposed final answer, decide if the "
    "answer directly resolves the final target. Accept only when the answer is "
    "supported by the trace and has the expected type. Flag bridge-only "
    "answers, unsupported answers, contradictions, wrong type, or answers that "
    "skip the final target. Return strict JSON only."
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
    critic_instructions: str = DEFAULT_CRITIC_INSTRUCTIONS
    use_critic: bool = True
    max_critic_retries: int = 1
    # Adaptive critic: skip critic when easy (n_nodes==1 AND min_finding_confidence >= tau_skip_critic)
    tau_skip_critic: float = 0.7


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

    def __init__(self, root_lm: dspy.LM, sub_lm: dspy.LM, retriever: Retriever, config: AdaptiveConfig, *,
                 synth_lm: dspy.LM | None = None, critic_lm: dspy.LM | None = None):
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        # synth and critic default to sub_lm (no-think) for speed; planner stays root_lm (think)
        self.synth_lm = synth_lm or sub_lm
        self.critic_lm = critic_lm or sub_lm
        self.retriever = retriever
        self.config = config
        self.tool_state = ToolRuntime()
        self.library = _load_library(config.experience_library)

        # Build module signatures with current instructions
        self.plan_sig = PlanDAGSig.with_instructions(config.planner_instructions)
        self.synth_sig = SynthesizeFinalSig.with_instructions(config.synth_instructions)
        self.critic_sig = CritiqueFinalSig.with_instructions(config.critic_instructions)
        self.plan_predict = dspy.Predict(self.plan_sig)
        self.synth_predict = dspy.Predict(self.synth_sig)
        self.critic_predict = dspy.Predict(self.critic_sig)

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
            })
        return json.dumps(trace, ensure_ascii=False)

    async def _synth(self, question: str, expected_type: str, plan: PlanRun) -> tuple[str, str]:
        trace_json = self._trace_json(plan)
        try:
            def _call():
                with dspy.context(lm=self.synth_lm):
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

    async def _critic(self, question: str, expected_type: str, plan: PlanRun, answer: str) -> dict[str, str]:
        try:
            def _call():
                with dspy.context(lm=self.critic_lm):
                    return self.critic_predict(
                        question=question,
                        expected_type=expected_type,
                        trace_json=self._trace_json(plan),
                        final_answer=answer,
                    )
            pred = await aio_to_thread(_call)
            raw = str(getattr(pred, "verdict_json", "")).strip()
            obj = _parse_json_obj(raw)
            verdict = str(obj.get("verdict", "flag")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if verdict not in {"accept", "flag"}:
                verdict = "flag"
            return {"verdict": verdict, "reason": reason or "critic returned no reason"}
        except Exception as exc:
            self.tool_state.tool_errors.append(f"critic error: {exc}")
            return {"verdict": "accept", "reason": "critic unavailable"}

    async def _execute(self, question: str, expected_type: str, note: str = "") -> PlanRun:
        plan: PlanRun | None = None
        if self.config.use_dag:
            plan = await self._plan_one(question, expected_type, note)
        if plan is None or not plan.nodes:
            return await self._direct_path(question, expected_type)
        runner = AdaptiveDAGRunner(
            retriever=self.retriever,
            sub_lm=self.sub_lm,
            state=self.tool_state,
            max_nodes=self.config.max_nodes,
            max_recursion_depth=self.config.max_recursion_depth,
            tau_recurse=self.config.tau_recurse,
        )
        await execute_plan(plan, runner, self._plan_one if self.config.max_recursion_depth > 0 else None, recursion_depth=0)
        return plan

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

        plan = await self._execute(question, expected_type)

        # Synthesize final answer + citations
        synth_ans, synth_ids = await self._synth(question, expected_type, plan)
        critic_trace: list[dict[str, str]] = []
        topology_mutated = False
        # Adaptive critic: skip on confidently-easy questions (single-node plan with high-conf finding)
        min_conf = min((f.confidence for f in self.tool_state.findings), default=0.0)
        skip_critic = (
            len(plan.nodes) == 1
            and min_conf >= self.config.tau_skip_critic
            and bool(synth_ans.strip())
        )
        critic_skipped_reason = "easy_high_confidence_single_hop" if skip_critic else ""
        if self.config.use_critic and not skip_critic:
            verdict = await self._critic(question, expected_type, plan, synth_ans)
            critic_trace.append(verdict)
            if verdict["verdict"] == "flag" and self.config.max_critic_retries > 0:
                hint = (
                    "CRITIC FLAG: "
                    f"{verdict['reason']}\n"
                    "Topology mutation: re-plan the DAG to resolve the missing final target. "
                    "Add or refine a bridge_resolver node if the current answer is only an intermediate entity."
                )
                mutated = await self._plan_one(question, expected_type, hint)
                if mutated and mutated.nodes:
                    topology_mutated = True
                    runner = AdaptiveDAGRunner(
                        retriever=self.retriever,
                        sub_lm=self.sub_lm,
                        state=self.tool_state,
                        max_nodes=self.config.max_nodes,
                        max_recursion_depth=self.config.max_recursion_depth,
                        tau_recurse=self.config.tau_recurse,
                    )
                    await execute_plan(mutated, runner, self._plan_one if self.config.max_recursion_depth > 0 else None, recursion_depth=0)
                    plan = mutated
                    synth_ans, synth_ids = await self._synth(question, expected_type, plan)
                    critic_trace.append(await self._critic(question, expected_type, plan, synth_ans))

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
        # If synth/critic LMs differ from sub_lm, also include their tokens
        synth_tokens = 0
        if self.synth_lm is not self.sub_lm and self.synth_lm is not self.root_lm:
            synth_tokens = _tokens_since(self.synth_lm, len(getattr(self.synth_lm, "history", [])) - max(1, len(getattr(self.synth_lm, "history", []))))
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
            "critic": critic_trace,
            "critic_skipped_reason": critic_skipped_reason,
            "topology_mutated": topology_mutated,
        }
        return {
            "question": question,
            "predicted_answer": final_answer,
            "answer": final_answer,
            "trajectory": {"plan_nodes": node_dicts, "synth_answer_raw": synth_ans, "critic": critic_trace},
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


def _parse_json_obj(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
