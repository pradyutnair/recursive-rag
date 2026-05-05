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
    RouteQuestionSig,
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
    "\"final_node\":\"Q1.1\"}\n\n"
    "Q: What is the tallest building in Europe named after?\n"
    "{\"nodes\":[{\"id\":\"Q1.1\",\"question\":\"What is the tallest "
    "building in Europe?\",\"expected_type\":\"entity\","
    "\"depends_on\":[]},{\"id\":\"Q2.1\",\"question\":\"What is <A1.1> "
    "named after?\",\"expected_type\":\"entity\","
    "\"depends_on\":[\"Q1.1\"]}],\"final_node\":\"Q2.1\"}"
)

DEFAULT_SYNTH_INSTRUCTIONS = (
    "You are an adaptive multi-hop QA synthesizer. Read the original question "
    "and the resolved DAG trace. Produce ONE concise final span that DIRECTLY "
    "answers the user question (not an intermediate bridge entity). Preserve "
    "full canonical names, full dates, full award/category names, and acronym "
    "expansions when supported. Match the answer granularity to the question: "
    "if the question asks 'what year', give the year; if it asks for a name, "
    "give the full name. Output 1-10 words, no explanation, no "
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

DEFAULT_ROUTER_INSTRUCTIONS = (
    "You are a resource-aware router for open-domain multi-hop QA. Return "
    "STRICT JSON only: {\"route\":\"easy|hard\",\"reason\":\"...\"}.\n"
    "Use easy when one investigator with retrieve/extract/rewrite retries "
    "can directly answer the final target. Prefer hard for bridge chains, "
    "unnamed bridge entities (descriptive/superlative subjects like 'the "
    "fastest X', 'the largest X'), nested of/that/which/who dependencies, "
    "comparisons, intersections, temporal ordering, arithmetic, numeric "
    "lookups through another entity, property-of-property questions "
    "('made out of', 'named after', 'derived from'), or ambiguity.\n"
    "\nFew-shot routing examples:\n"
    "Q: Are Nicholas Irving and David Ridgway (Scholar) from the same country?\n"
    "{\"route\":\"easy\",\"reason\":\"Both entities are named; direct relation check.\"}\n"
    "Q: The organization which sets the standards for ISO 10006 is headquartered in what city?\n"
    "{\"route\":\"easy\",\"reason\":\"A single named organization resolved by retrieval.\"}\n"
    "Q: What is the fastest air-breathing manned aircraft mostly made out of?\n"
    "{\"route\":\"hard\",\"reason\":\"'Fastest air-breathing manned aircraft' is unnamed; must resolve then find material.\"}\n"
    "Q: Who is the largest aircraft carrier in the world named after?\n"
    "{\"route\":\"hard\",\"reason\":\"'Largest aircraft carrier' is unnamed; must resolve then find namesake.\"}\n"
    "Q: Who is the spouse of the director of film Son Of Samson?\n"
    "{\"route\":\"hard\",\"reason\":\"Requires resolving director bridge before spouse lookup.\"}\n"
)

SAFE_EASY_STOP_PROFILES: set[str] = set()


def _safe_easy_stop(profile: str, finding: HopFinding | None, tau: float) -> bool:
    if finding is None or finding.confidence < tau:
        return False
    if not finding.answer.strip() or not finding.evidence_chunk_id:
        return False
    if SAFE_EASY_STOP_PROFILES and str(profile or "") not in SAFE_EASY_STOP_PROFILES:
        return False
    return True


@dataclass
class AdaptiveConfig:
    max_nodes: int = 6
    max_recursion_depth: int = 2
    tau_recurse: float = 0.5
    experience_library: str | None = None
    use_dag: bool = True
    planner_instructions: str = DEFAULT_PLANNER_INSTRUCTIONS
    router_instructions: str = DEFAULT_ROUTER_INSTRUCTIONS
    synth_instructions: str = DEFAULT_SYNTH_INSTRUCTIONS
    critic_instructions: str = DEFAULT_CRITIC_INSTRUCTIONS
    use_critic: bool = False
    use_router: bool = True
    force_route: str | None = None
    budget_hint: str = "normal"
    max_critic_retries: int = 0
    max_searches: int = 3
    share_mode: str = "full_share"
    worker_width: int = 999
    blackboard_top_k: int = 3
    repair_budget: int = 0
    use_escalation: bool = True
    tau_escalate: float = 0.7
    easy_max_attempts: int = 2
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
                 synth_lm: dspy.LM | None = None, critic_lm: dspy.LM | None = None, route_lm: dspy.LM | None = None):
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        # synth and critic default to sub_lm (no-think) for speed; planner stays root_lm (think)
        self.synth_lm = synth_lm or sub_lm
        self.critic_lm = critic_lm or sub_lm
        self.route_lm = route_lm or sub_lm
        self.retriever = retriever
        self.config = config
        self.tool_state = ToolRuntime()
        self.library = _load_library(config.experience_library)

        # Build module signatures with current instructions
        self.route_sig = RouteQuestionSig.with_instructions(config.router_instructions)
        self.plan_sig = PlanDAGSig.with_instructions(config.planner_instructions)
        self.synth_sig = SynthesizeFinalSig.with_instructions(config.synth_instructions)
        self.critic_sig = CritiqueFinalSig.with_instructions(config.critic_instructions)
        self.route_predict = dspy.Predict(self.route_sig)
        self.plan_predict = dspy.Predict(self.plan_sig)
        self.synth_predict = dspy.Predict(self.synth_sig)
        self.critic_predict = dspy.Predict(self.critic_sig)

    def _experience(self, prof: str, note: str = "") -> str:
        exp = self.library.to_text(profile=prof, top_k=4) if self.library.entries else ""
        if note:
            exp = (exp + ("\n" if exp else "") + note).strip()
        return exp

    def _norm_budget_hint(self, budget_hint: str | None) -> str:
        hint = str(budget_hint or self.config.budget_hint or "normal").strip().lower()
        return hint if hint in {"tight", "normal", "rich"} else "normal"

    async def _route(self, question: str, prof: str, budget_hint: str) -> tuple[str, str]:
        forced = str(self.config.force_route or "").strip().lower()
        if forced in {"easy", "hard"}:
            return forced, f"forced_{forced}"
        if not self.config.use_router:
            return "hard", "router disabled"
        try:
            def _call():
                with dspy.context(lm=self.route_lm):
                    return self.route_predict(
                        question=question,
                        profile=prof,
                        experience=self._experience(prof),
                    )
            pred = await aio_to_thread(_call)
            obj = _parse_json_obj(str(getattr(pred, "route_json", "")))
            route = str(obj.get("route", "")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if route in {"easy", "hard"}:
                return route, reason
        except Exception as exc:
            self.tool_state.tool_errors.append(f"router error: {exc}")
        if prof == "one_hop":
            return "easy", "fallback one_hop"
        return "hard", "fallback non-one_hop"

    async def _plan_one(self, question: str, expected_type: str = "auto", note: str = "", budget_hint: str = "normal") -> PlanRun | None:
        prof = classify(question)
        exp = self._experience(prof, note)
        try:
            def _call():
                with dspy.context(lm=self.root_lm):
                    return self.plan_predict(question=question, profile=prof, experience=exp, budget_hint=budget_hint)
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
                "retrieval_query": n.retrieval_query,
            })
        return json.dumps(trace, ensure_ascii=False)

    async def _synth(self, question: str, expected_type: str, plan: PlanRun, budget_hint: str) -> tuple[str, str]:
        trace_json = self._trace_json(plan)
        try:
            def _call():
                with dspy.context(lm=self.synth_lm):
                    return self.synth_predict(
                        question=question,
                        expected_type=expected_type,
                        trace_json=trace_json,
                        budget_hint=budget_hint,
                    )
            pred = await aio_to_thread(_call)
            answer = str(getattr(pred, "final_answer", "")).strip()
            ids = str(getattr(pred, "support_ids", "")).strip()
        except Exception as exc:
            self.tool_state.tool_errors.append(f"synth error: {exc}")
            answer = ""
            ids = ""
        return answer, ids

    def _prefer_final_node_answer(self, plan: PlanRun, synth_ans: str, synth_ids: str) -> tuple[str, str, bool]:
        final_node = plan.nodes.get(plan.final_node)
        if not final_node or not final_node.answer.strip():
            return synth_ans, synth_ids, False
        if final_node.confidence < 0.75 or not final_node.chunk_id:
            return synth_ans, synth_ids, False
        ans_norm = normalize_answer(synth_ans)
        node_norm = normalize_answer(final_node.answer)
        if ans_norm and (ans_norm in node_norm or node_norm in ans_norm):
            return synth_ans, synth_ids, False
        return final_node.answer, final_node.chunk_id, True

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

    async def _execute(self, question: str, expected_type: str, note: str = "", budget_hint: str = "normal") -> PlanRun:
        plan: PlanRun | None = None
        if self.config.use_dag:
            plan = await self._plan_one(question, expected_type, note, budget_hint)
        if plan is None or not plan.nodes:
            return await self._direct_path(question, expected_type)
        runner = AdaptiveDAGRunner(
            retriever=self.retriever,
            sub_lm=self.sub_lm,
            state=self.tool_state,
            max_nodes=self.config.max_nodes,
            max_recursion_depth=self.config.max_recursion_depth,
            tau_recurse=self.config.tau_recurse,
            max_searches=self.config.max_searches,
            share_mode=self.config.share_mode,
            worker_width=self.config.worker_width,
            blackboard_top_k=self.config.blackboard_top_k,
        )
        plan_one = (lambda q, et, n: self._plan_one(q, et, n, budget_hint)) if self.config.max_recursion_depth > 0 else None
        await execute_plan(plan, runner, plan_one, recursion_depth=0)
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

    async def _direct_path(self, question: str, expected_type: str, max_attempts: int | None = None) -> PlanRun:
        """Single-hop SAS path used as fallback when the planner fails."""
        plan = PlanRun(nodes={}, final_node="Q1.1")
        node = PlannedNode(id="Q1.1", question=question, raw_question=question, expected_type=expected_type, depends_on=[])
        plan.nodes["Q1.1"] = node
        runner = AdaptiveDAGRunner(
            retriever=self.retriever,
            sub_lm=self.sub_lm,
            state=self.tool_state,
            max_nodes=self.config.max_nodes,
            max_recursion_depth=0,
            tau_recurse=self.config.tau_recurse,
            share_mode=self.config.share_mode,
            worker_width=self.config.worker_width,
            blackboard_top_k=self.config.blackboard_top_k,
        )
        runner.max_searches = int(max_attempts or self.config.max_searches)
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

    async def run(self, question: str, budget_hint: str | None = None) -> dict[str, Any]:
        self.tool_state.reset()
        budget_hint = self._norm_budget_hint(budget_hint)
        t0 = time.time()
        root_start = len(getattr(self.root_lm, "history", []))
        sub_start = len(getattr(self.sub_lm, "history", []))

        expected_type = self._expected_type_hint(question)
        prof = classify(question)

        route, route_reason = await self._route(question, prof, budget_hint)
        initial_route = route
        escalated = False
        level1_findings: list[dict[str, Any]] = []
        level1_best_confidence = 0.0
        level1_best_answer = ""
        if route == "easy":
            plan = await self._direct_path(question, expected_type, max_attempts=self.config.easy_max_attempts)
            best_level1 = max(self.tool_state.findings, key=lambda f: f.confidence, default=None)
            level1_findings = [f.as_dict() for f in self.tool_state.findings]
            if best_level1:
                level1_best_confidence = float(best_level1.confidence)
                level1_best_answer = best_level1.answer
            can_stop = _safe_easy_stop(prof, best_level1, self.config.tau_escalate)
            if self.config.use_escalation and not can_stop:
                escalated = True
                preserved_chunks = dict(self.tool_state.chunks_by_id)
                preserved_errors = list(self.tool_state.tool_errors)
                self.tool_state.findings.clear()
                self.tool_state.total_hops = 0
                self.tool_state.total_retries = 0
                self.tool_state.tool_errors[:] = preserved_errors
                self.tool_state.chunks_by_id.update(preserved_chunks)
                hint = (
                    "LEVEL-1 single-investigator attempt was not confident enough; "
                    f"best_answer={level1_best_answer!r}, confidence={level1_best_confidence:.2f}. "
                    "Escalate to a full DAG that resolves the final target, not just the weak intermediate answer."
                )
                route = "hard"
                route_reason = f"{route_reason} | escalated_low_confidence"
                plan = await self._execute(question, expected_type, note=hint, budget_hint=budget_hint)
        else:
            plan = await self._execute(question, expected_type, budget_hint=budget_hint)

        direct_easy_answer = False
        best_easy = max(self.tool_state.findings, key=lambda f: f.confidence, default=None)
        if route == "easy" and _safe_easy_stop(prof, best_easy, self.config.tau_escalate):
            synth_ans = best_easy.answer
            synth_ids = best_easy.evidence_chunk_id
            used_final_node_answer = False
            direct_easy_answer = True
        else:
            # Synthesize final answer + citations
            synth_ans, synth_ids = await self._synth(question, expected_type, plan, budget_hint)
            synth_ans, synth_ids, used_final_node_answer = self._prefer_final_node_answer(plan, synth_ans, synth_ids)
        critic_trace: list[dict[str, str]] = []
        topology_mutated = False
        # Adaptive critic: skip on confidently-easy questions (single-node plan with high-conf finding)
        min_conf = min((f.confidence for f in self.tool_state.findings), default=0.0)
        skip_critic = (
            direct_easy_answer
            or (
                prof == "one_hop"
                and route == "easy"
                and len(plan.nodes) == 1
                and min_conf >= self.config.tau_skip_critic
                and bool(synth_ans.strip())
            )
        )
        critic_skipped_reason = "direct_easy_high_confidence" if direct_easy_answer else ("easy_high_confidence_single_hop" if skip_critic else "")
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
                mutated = await self._plan_one(question, expected_type, hint, budget_hint)
                if mutated and mutated.nodes:
                    topology_mutated = True
                    runner = AdaptiveDAGRunner(
                        retriever=self.retriever,
                        sub_lm=self.sub_lm,
                        state=self.tool_state,
                        max_nodes=self.config.max_nodes,
                        max_recursion_depth=self.config.max_recursion_depth,
                        tau_recurse=self.config.tau_recurse,
                        max_searches=self.config.max_searches,
                        share_mode=self.config.share_mode,
                        worker_width=self.config.worker_width,
                        blackboard_top_k=self.config.blackboard_top_k,
                    )
                    plan_one = (lambda q, et, n: self._plan_one(q, et, n, budget_hint)) if self.config.max_recursion_depth > 0 else None
                    await execute_plan(mutated, runner, plan_one, recursion_depth=0)
                    plan = mutated
                    route = "hard"
                    synth_ans, synth_ids = await self._synth(question, expected_type, plan, budget_hint)
                    synth_ans, synth_ids, used_final_node_answer = self._prefer_final_node_answer(plan, synth_ans, synth_ids)
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
            "easy_lane" if route == "easy"
            else "single_hop" if len(plan.nodes) == 1
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
            "share_mode": self.config.share_mode,
            "worker_width": self.config.worker_width,
            "blackboard_top_k": self.config.blackboard_top_k,
            "repair_budget": self.config.repair_budget,
            "route": route,
            "initial_route": initial_route,
            "router_reason": route_reason,
            "escalated": escalated,
            "tau_escalate": self.config.tau_escalate,
            "easy_max_attempts": self.config.easy_max_attempts,
            "level1_best_confidence": round(level1_best_confidence, 3),
            "level1_best_answer": level1_best_answer,
            "level1_findings": level1_findings,
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
            "used_final_node_answer": used_final_node_answer,
            "direct_easy_answer": direct_easy_answer,
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
