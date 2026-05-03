"""Fully synchronous Adaptive Recursive RAG pipeline.

Avoids asyncio entirely so it composes cleanly with dspy.GEPA's thread-based
concurrency. Parallelism within a DAG layer is provided by a small owned
ThreadPoolExecutor (per-layer concurrent.futures.gather equivalent).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy

from .contracts import HopFinding, RetrievedChunk, normalize_answer
from .dag import (
    CritiqueFinalSig,
    PlanDAGSig,
    PlanRun,
    PlannedNode,
    RouteQuestionSig,
    SynthesizeFinalSig,
    parse_plan,
    substitute_tags,
    _looks_like_bridge,
)
from .grpo.library import ExperienceLibrary
from .pipeline import _clean_final_answer
from .profile import classify, expected_hops
from .retriever import Retriever
from .tools import (
    _SENTENCE_LIKE,
    CONF_THRESHOLD,
    MAX_ATTEMPTS,
    MAX_SPAN_WORDS,
    RETRIEVE_TOPK,
    ExtractAnswerSpan,
    ProposeQueryRewrite,
    ToolRuntime,
    _clean_answer_type,
    _format_chunks,
    _grounded,
    _parse_jsonish,
    _shape_ok,
    _trim_answer,
    _model_tokens,
)

# Per-pipeline executor for layer-parallel node execution
_LAYER_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="layer")

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

DEFAULT_ROUTER_INSTRUCTIONS = (
    "You are a resource-aware router for open-domain multi-hop QA. Return "
    "STRICT JSON only: {\"route\":\"easy|hard\",\"reason\":\"...\"}.\n"
    "Use easy only when one investigator with retrieve/extract/rewrite retries "
    "can directly answer the final target. Prefer hard for bridge chains, "
    "unnamed bridge entities, nested of/that/which/who dependencies, "
    "comparisons, intersections, temporal ordering, arithmetic, numeric "
    "lookups through another entity, or ambiguity. Route hard when the question "
    "contains phrases like 'where X is located', 'city where', 'country where', "
    "'alma mater of', 'composer of', 'director of', or asks for a quantity after "
    "resolving another entity. Route yes/no questions easy only when both "
    "entities are explicitly named and the relation can be checked directly. "
    "\nFew-shot routing examples from fresh SAS-oracle training data:\n"
    "Q: Are Nicholas Irving and David Ridgway (Scholar) from the same country?\n"
    "{\"route\":\"easy\",\"reason\":\"Both entities are named and the final answer is a direct yes/no relation.\"}\n"
    "Q: Musicality features covers of songs from a jukebox musical written by who?\n"
    "{\"route\":\"easy\",\"reason\":\"One named work leads to one recoverable author attribute.\"}\n"
    "Q: The organization which sets the standards for ISO 10006 is headquartered in what city?\n"
    "{\"route\":\"easy\",\"reason\":\"A single named organization can be resolved by retrieval rewrites.\"}\n"
    "Q: Who is the spouse of the director of film Son Of Samson?\n"
    "{\"route\":\"hard\",\"reason\":\"Requires resolving a director bridge before spouse lookup.\"}\n"
    "Q: Which portion of the Nile runs from the nation descendants of African Americans migrated from, to the country where Hay Al-Arab is found?\n"
    "{\"route\":\"hard\",\"reason\":\"Nested bridges over nations and locations require planned decomposition.\"}\n"
    "Q: The 17th Premier of Nova Scotia and leader of the federal Progressive Conservative Party of Canada Robert Stanfield fought a few times and lost in general elections against which politician who is currently one of the longest-serving Prime Minister in Canadian history?\n"
    "{\"route\":\"hard\",\"reason\":\"Long temporal/entity chain with distractors; use the DAG lane.\"}\n"
    "Budget hints adjust caution: tight may route simple named-subject "
    "attribute questions to easy; rich should route borderline bridge questions "
    "to hard."
)


@dataclass
class AdaptiveConfig:
    max_nodes: int = 6
    max_recursion_depth: int = 1
    tau_recurse: float = 0.5
    experience_library: str | None = None
    use_dag: bool = True
    planner_instructions: str = DEFAULT_PLANNER_INSTRUCTIONS
    router_instructions: str = DEFAULT_ROUTER_INSTRUCTIONS
    synth_instructions: str = DEFAULT_SYNTH_INSTRUCTIONS
    critic_instructions: str = DEFAULT_CRITIC_INSTRUCTIONS
    use_critic: bool = True
    use_router: bool = True
    force_route: str | None = None
    budget_hint: str = "normal"
    max_critic_retries: int = 0
    tau_skip_critic: float = 0.7
    max_searches: int = 3


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
    lib = ExperienceLibrary()
    for line in p.read_text(encoding="utf-8").splitlines():
        clean = re.sub(r"^E-\d{3}\s*(\[[^\]]+\])?:?\s*", "", line).strip(" -\t")
        if clean:
            lib.add(clean)
    return lib


def _hop_sync(
    question: str,
    expected_answer_type: str,
    retriever: Retriever,
    sub_lm: dspy.LM,
    state: ToolRuntime,
    max_attempts: int | None = None,
    initial_query: str | None = None,
) -> dict:
    """Synchronous mirror of tools._hop_async (retrieve + extract + rewrite + retry)."""
    question = str(question or "").strip()
    expected_answer_type = _clean_answer_type(expected_answer_type)
    if not question:
        finding = HopFinding(confidence=0.0)
        state.findings.append(finding)
        return finding.model_dump()

    queries: list[str] = []
    seen_chunk_sets: list[set[str]] = []
    best = HopFinding(confidence=0.0)
    attempts = max(1, int(max_attempts or MAX_ATTEMPTS))
    for attempt in range(attempts):
        if attempt == 0:
            query = str(initial_query or question).strip()
        else:
            try:
                with dspy.context(lm=sub_lm):
                    pred = dspy.Predict(ProposeQueryRewrite)(
                        question=question,
                        expected_answer_type=expected_answer_type,
                        previous_queries="\n".join(queries),
                        best_answer_so_far=best.answer,
                    )
                query = str(getattr(pred, "rewritten_query", "")).strip() or f"{question} answer"
            except Exception:
                query = f"{question} answer"
        if query in queries:
            query = f"{query} evidence"
        queries.append(query)
        chunks = retriever._retrieve_batch_sync([query], k=RETRIEVE_TOPK)
        chunks = chunks[0] if chunks else []
        chunk_ids = {c.chunk_id for c in chunks}
        for chunk in chunks:
            state.chunks_by_id[chunk.chunk_id] = chunk
        if seen_chunk_sets and chunk_ids == seen_chunk_sets[-1]:
            if attempt < attempts - 1:
                continue
        seen_chunk_sets.append(chunk_ids)
        finding = _extract_sync(sub_lm, question, expected_answer_type, chunks)
        finding.queries_used = list(queries)
        if finding.confidence > best.confidence:
            best = finding
        if finding.confidence >= CONF_THRESHOLD:
            break

    best.queries_used = list(queries)
    state.total_hops += 1
    state.total_retries += max(0, len(queries) - 1)
    state.findings.append(best)
    return best.model_dump()


def _extract_sync(sub_lm: dspy.LM, question: str, expected_answer_type: str, chunks: list[RetrievedChunk]) -> HopFinding:
    try:
        with dspy.context(lm=sub_lm):
            pred = dspy.Predict(ExtractAnswerSpan)(
                question=question,
                expected_answer_type=_clean_answer_type(expected_answer_type),
                chunks_json=_format_chunks(chunks),
            )
    except Exception:
        return HopFinding(confidence=0.0)

    obj = _parse_jsonish(getattr(pred, "extraction_json", ""))
    raw_answer = str(obj.get("answer_span", "")).strip()
    ev = str(obj.get("evidence_chunk_id", "")).strip()
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    answer = _trim_answer(raw_answer, _clean_answer_type(expected_answer_type))
    word_count = len(answer.split())
    if answer and word_count > MAX_SPAN_WORDS:
        conf = min(conf, 0.45)
    elif answer and word_count > 8:
        conf = min(conf, 0.6)
    if answer and _SENTENCE_LIKE.search(answer):
        conf = min(conf, 0.4)
    if answer and not _shape_ok(answer, question, expected_answer_type):
        conf = min(conf, 0.35)
    if answer and conf > 0.4 and _clean_answer_type(expected_answer_type) != "yes_no" and not _grounded(answer, ev, chunks):
        conf = min(conf, 0.4)
    return HopFinding(answer=answer, evidence_chunk_id=ev, confidence=conf, expected_answer_type=_clean_answer_type(expected_answer_type))


def _run_node_sync(node: PlannedNode, retriever: Retriever, sub_lm: dspy.LM, state: ToolRuntime, max_searches: int = 3) -> None:
    data = _hop_sync(node.question, node.expected_type, retriever, sub_lm, state, max_attempts=max_searches, initial_query=node.retrieval_query or None)
    node.answer = str(data.get("answer", ""))
    node.confidence = float(data.get("confidence", 0.0) or 0.0)
    node.chunk_id = str(data.get("evidence_chunk_id", ""))
    node.queries_used = list(data.get("queries_used", []) or [])


def _execute_plan_sync(
    plan: PlanRun,
    retriever: Retriever,
    sub_lm: dspy.LM,
    state: ToolRuntime,
    plan_one_fn=None,
    config: AdaptiveConfig | None = None,
    recursion_depth: int = 0,
) -> None:
    layers = plan.topo_layers()
    for layer in layers:
        resolved = {nid: plan.nodes[nid].answer for nid in plan.nodes if plan.nodes[nid].answer}
        for nid in layer:
            n = plan.nodes[nid]
            n.question = substitute_tags(n.raw_question, resolved)
            if n.retrieval_query:
                n.retrieval_query = substitute_tags(n.retrieval_query, resolved)
        if len(layer) == 1:
            _run_node_sync(plan.nodes[layer[0]], retriever, sub_lm, state, config.max_searches if config else 3)
        else:
            futures = [_LAYER_EXECUTOR.submit(_run_node_sync, plan.nodes[nid], retriever, sub_lm, state, config.max_searches if config else 3) for nid in layer]
            for fut in futures:
                fut.result()
        # Recursion expansion (sequential - rare path)
        if plan_one_fn is not None and config and config.max_recursion_depth > recursion_depth:
            for nid in layer:
                n = plan.nodes[nid]
                if n.confidence >= config.tau_recurse:
                    continue
                if not _looks_like_bridge(n.question):
                    continue
                if len(plan.nodes) >= config.max_nodes:
                    continue
                sub = plan_one_fn(n.question, n.expected_type, "(recurse on low-confidence bridge)")
                if not sub or not sub.nodes:
                    continue
                prefix = n.id.replace("Q", "R")
                rename = {sid: f"{prefix}_{sid}" for sid in sub.nodes}
                for old, new in rename.items():
                    nn = sub.nodes.pop(old)
                    nn.id = new
                    nn.depends_on = [rename.get(d, d) for d in nn.depends_on]
                    sub.nodes[new] = nn
                new_final = rename.get(sub.final_node, sub.final_node)
                _execute_plan_sync(sub, retriever, sub_lm, state, None, None, recursion_depth + 1)
                for new_id, nn in sub.nodes.items():
                    plan.nodes[new_id] = nn
                    n.expanded_into.append(new_id)
                sub_final = sub.nodes.get(new_final)
                if sub_final and sub_final.confidence >= n.confidence:
                    n.answer = sub_final.answer
                    n.confidence = sub_final.confidence
                    n.chunk_id = sub_final.chunk_id


class SyncAdaptivePipeline:
    """Synchronous Adaptive Recursive RAG pipeline (no asyncio)."""

    def __init__(self, root_lm: dspy.LM, sub_lm: dspy.LM, retriever: Retriever, config: AdaptiveConfig):
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        self.retriever = retriever
        self.config = config
        self.tool_state = ToolRuntime()
        self.library = _load_library(config.experience_library)
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

    def _route(self, question: str, prof: str, budget_hint: str) -> tuple[str, str]:
        forced = str(self.config.force_route or "").strip().lower()
        if forced in {"easy", "hard"}:
            return forced, f"forced_{forced}"
        if not self.config.use_router:
            return "hard", "router disabled"
        try:
            with dspy.context(lm=self.sub_lm):
                pred = self.route_predict(
                    question=question,
                    profile=prof,
                    experience=self._experience(prof),
                    budget_hint=budget_hint,
                )
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

    def _plan_one(self, question: str, expected_type: str = "auto", note: str = "", budget_hint: str = "normal") -> PlanRun | None:
        prof = classify(question)
        exp = self._experience(prof, note)
        try:
            with dspy.context(lm=self.root_lm):
                pred = self.plan_predict(question=question, profile=prof, experience=exp, budget_hint=budget_hint)
            return parse_plan(getattr(pred, "plan_json", ""))
        except Exception as exc:
            self.tool_state.tool_errors.append(f"plan error: {exc}")
            return None

    def _trace_json(self, plan: PlanRun) -> str:
        trace = []
        for nid in sorted(plan.nodes):
            n = plan.nodes[nid]
            trace.append({
                "id": n.id, "question": n.question, "answer": n.answer,
                "confidence": round(n.confidence, 3), "chunk_id": n.chunk_id,
                "expected_type": n.expected_type, "depth": n.depth,
                "retrieval_query": n.retrieval_query,
            })
        return json.dumps(trace, ensure_ascii=False)

    def _synth(self, question: str, expected_type: str, plan: PlanRun, budget_hint: str) -> tuple[str, str]:
        try:
            with dspy.context(lm=self.sub_lm):
                pred = self.synth_predict(
                    question=question,
                    expected_type=expected_type,
                    trace_json=self._trace_json(plan),
                    budget_hint=budget_hint,
                )
            return str(getattr(pred, "final_answer", "")).strip(), str(getattr(pred, "support_ids", "")).strip()
        except Exception as exc:
            self.tool_state.tool_errors.append(f"synth error: {exc}")
            return "", ""

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

    def _critic(self, question: str, expected_type: str, plan: PlanRun, answer: str) -> dict[str, str]:
        try:
            with dspy.context(lm=self.sub_lm):
                pred = self.critic_predict(
                    question=question,
                    expected_type=expected_type,
                    trace_json=self._trace_json(plan),
                    final_answer=answer,
                )
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

    def _citation_check(self, answer: str, support_ids_raw: str) -> tuple[bool, list[str]]:
        ids = [x.strip() for x in re.split(r"[,\n]", support_ids_raw) if x.strip()][:6]
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

    def _direct_path(self, question: str, expected_type: str) -> PlanRun:
        plan = PlanRun(nodes={}, final_node="Q1.1")
        node = PlannedNode(id="Q1.1", question=question, raw_question=question, expected_type=expected_type, depends_on=[])
        plan.nodes["Q1.1"] = node
        _run_node_sync(node, self.retriever, self.sub_lm, self.tool_state, self.config.max_searches)
        return plan

    def run(self, question: str, budget_hint: str | None = None) -> dict[str, Any]:
        self.tool_state.reset()
        budget_hint = self._norm_budget_hint(budget_hint)
        t0 = time.time()
        root_start = len(getattr(self.root_lm, "history", []))
        sub_start = len(getattr(self.sub_lm, "history", []))

        expected_type = self._expected_type_hint(question)
        prof = classify(question)

        route, route_reason = self._route(question, prof, budget_hint)
        plan: PlanRun | None = None
        if route == "hard" and self.config.use_dag:
            plan = self._plan_one(question, expected_type, budget_hint=budget_hint)
        if route == "easy" or plan is None or not plan.nodes:
            plan = self._direct_path(question, expected_type)
        else:
            plan_one = lambda q, et, n: self._plan_one(q, et, n, budget_hint)
            _execute_plan_sync(plan, self.retriever, self.sub_lm, self.tool_state, plan_one, self.config, 0)

        direct_easy_answer = False
        best_easy = max(self.tool_state.findings, key=lambda f: f.confidence, default=None)
        if route == "easy" and best_easy and best_easy.confidence >= 0.7 and best_easy.answer.strip() and best_easy.evidence_chunk_id:
            synth_ans = best_easy.answer
            synth_ids = best_easy.evidence_chunk_id
            used_final_node_answer = False
            direct_easy_answer = True
        else:
            synth_ans, synth_ids = self._synth(question, expected_type, plan, budget_hint)
            synth_ans, synth_ids, used_final_node_answer = self._prefer_final_node_answer(plan, synth_ans, synth_ids)
        critic_trace: list[dict[str, str]] = []
        topology_mutated = False
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
            verdict = self._critic(question, expected_type, plan, synth_ans)
            critic_trace.append(verdict)
            if verdict["verdict"] == "flag" and self.config.max_critic_retries > 0:
                hint = (
                    "CRITIC FLAG: " + verdict["reason"] + "\n"
                    "Topology mutation: re-plan the DAG to resolve the missing final target. "
                    "Add or refine a bridge_resolver node if the current answer is only an intermediate entity."
                )
                mutated = self._plan_one(question, expected_type, hint, budget_hint)
                if mutated and mutated.nodes:
                    topology_mutated = True
                    plan_one = lambda q, et, n: self._plan_one(q, et, n, budget_hint)
                    _execute_plan_sync(mutated, self.retriever, self.sub_lm, self.tool_state, plan_one, self.config, 0)
                    plan = mutated
                    route = "hard"
                    synth_ans, synth_ids = self._synth(question, expected_type, plan, budget_hint)
                    synth_ans, synth_ids, used_final_node_answer = self._prefer_final_node_answer(plan, synth_ans, synth_ids)
                    critic_trace.append(self._critic(question, expected_type, plan, synth_ans))

        accepted, used_ids = self._citation_check(synth_ans, synth_ids)
        if not accepted:
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
            "route": route,
            "router_reason": route_reason,
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
        readable = _readable_dag_trace_sync(question, plan, final_answer, used_ids)
        return {
            "question": question,
            "predicted_answer": final_answer,
            "answer": final_answer,
            "trajectory": {"plan_nodes": node_dicts, "synth_answer_raw": synth_ans, "critic": critic_trace},
            "metadata": metadata,
            "readable_trace": readable,
        }


def _parse_json_obj(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _readable_dag_trace_sync(question: str, plan: PlanRun, answer: str, support_ids: list[str]) -> str:
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
