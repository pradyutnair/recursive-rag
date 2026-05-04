from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import dspy

from .aio import to_thread as aio_to_thread
from .contracts import RetrievedChunk, normalize_answer
from .pipeline import _clean_final_answer
from .profile import classify
from .retriever import Retriever
RETRIEVE_TOPK = 5


@dataclass
class SASConfig:
    max_searches: int = 5
    retrieve_topk: int = RETRIEVE_TOPK
    excerpt_chars: int = 520


class SASResearchPlanSig(dspy.Signature):
    """Plan searches for a single-agent open-domain QA run.

    You are one search agent, not a DAG planner. Produce a focused sequential
    search plan that will let you answer the original question exactly.

    Rules:
    - If max_searches >= 3, use exactly 3 queries for bridge, descriptive, or
      nested questions. Use 2 queries only for truly direct single-fact
      questions.
    - For comparison or exclusion questions ("which X", "not related",
      "not associated", "unlike"), use 4 to max_searches queries: candidate
      list, relation target, candidate-relation checks, final verification.
    - For bridge questions, query the bridge entity first, then query the final
      requested property of the resolved entity.
    - The last query must verify the final answer relation.
    - Preserve the requested answer type and granularity: county, year, date,
      person, organization, title, number, yes/no.
    - Do not answer intermediate entities. Rewrite the question into the exact
      final property that must be answered after bridge resolution.
    - "derived from" asks for the source/precursor, not the thing derived.
    - "who built/developed/created" asks for the builder, not the artifact.
    - "which X is not related to Y" requires comparing candidate Xs; do not
      answer Y, the organization, or an executive unless that is the requested X.
    - Prefer queries that name likely candidates when the question implies a
      small candidate set.
    - Queries must be different from each other.

    Return strict JSON:
    {
      "queries":["query 1","query 2","query 3"],
      "final_question":"the exact final-property question to answer",
      "answer_focus":"what kind of span is acceptable and what is not acceptable",
      "answer_type":"person|place|date|number|title|organization|yes_no|entity"
    }
    """

    question: str = dspy.InputField()
    profile: str = dspy.InputField()
    max_searches: int = dspy.InputField()
    plan_json: str = dspy.OutputField(desc="Strict JSON search plan")


class SASNextActionSig(dspy.Signature):
    """Choose the next step for one single-agent open-domain QA run.

    You are one search agent, not a DAG planner. Inspect the evidence gathered
    so far, then either issue one more focused Wikipedia search query or decide
    that the evidence is sufficient to answer.

    Rules:
    - Search at least twice for direct questions and at least three times for
      bridge, comparison, descriptive, temporal, or numeric questions unless
      max_searches is smaller.
    - For bridge questions, first resolve the bridge entity, then use that
      concrete entity name in the next query. Never search with placeholders
      such as [location], [Series Name], this city, that person, or the entity.
    - For comparison or exclusion questions ("which X", "not related",
      "not associated", "unlike"), search for the candidate list, the relation
      target, and candidate-relation checks before answering.
    - Preserve the requested answer type and granularity: county, year, date,
      person, organization, title, number, yes/no.
    - Track the exact final property that must be answered after bridge
      resolution. Do not answer intermediate entities.
    - "derived from" asks for the source/precursor, not the thing derived.
    - "who built/developed/created" asks for the builder, not the artifact.
    - "which X is not related to Y" requires comparing candidate Xs; do not
      answer Y, the organization, or an executive unless that is the requested X.
    - Prefer queries that name likely candidates when the question implies a
      small candidate set.
    - The next query must be different from previous queries.
    - Use background knowledge only to form better search queries; final answers
      must come from retrieved evidence.

    Return strict JSON:
    {
      "action":"search|answer",
      "query":"next concrete search query, empty only if action is answer",
      "final_question":"the exact final-property question to answer",
      "answer_focus":"what kind of span is acceptable and what is not acceptable",
      "answer_type":"person|place|date|number|title|organization|yes_no|entity"
    }
    """

    question: str = dspy.InputField()
    profile: str = dspy.InputField()
    search_count: int = dspy.InputField()
    max_searches: int = dspy.InputField()
    previous_queries: str = dspy.InputField()
    evidence_json: str = dspy.InputField()
    action_json: str = dspy.OutputField(desc="Strict JSON next action")


class SASFocusedAnswerSig(dspy.Signature):
    """Extract the final answer span from retrieved Wikipedia evidence.

    Answer the final_question, using original_question only to preserve context.
    Return the smallest exact span that satisfies answer_focus.

    Rules:
    - Do not return the bridge entity, search target, article title, or subject
      if the question asks for a property of that entity.
    - Return complete official named entities from evidence. Do not shorten an
      organization, company, person, or title if the evidence gives the full
      name.
    - For "derived from", return the source/precursor material.
    - For "built/developed/created", return the builder/creator organization or
      person, not the artifact.
    - For "which X is not related to Y", return the X candidate that lacks the
      relation. Verify each candidate against Y before answering. Do not return
      Y, the central company, an executive, or a candidate that evidence says is
      related to Y.
    - If evidence is insufficient, return the best directly supported span.

    Return strict JSON:
    {"answer_span":"minimal answer","evidence_chunk_id":"chunk id"}
    """

    original_question: str = dspy.InputField()
    final_question: str = dspy.InputField()
    answer_focus: str = dspy.InputField()
    expected_answer_type: str = dspy.InputField()
    chunks_json: str = dspy.InputField()
    extraction_json: str = dspy.OutputField(desc="Strict JSON with answer_span and evidence_chunk_id")


def _tokens_since(lm: dspy.LM, start_idx: int) -> int:
    total = 0
    for item in getattr(lm, "history", [])[start_idx:]:
        usage = (item or {}).get("usage") or {}
        try:
            total += int(usage.get("total_tokens", 0))
        except Exception:
            pass
    return total


def _safe_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_support_ids(support_ids: str) -> list[str]:
    text = str(support_ids or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"[,\n]", text) if x.strip()]


def _expected_type_hint(question: str) -> str:
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


class SingleAgentSearchPipeline:
    def __init__(self, lm: dspy.LM, retriever: Retriever, config: SASConfig):
        self.lm = lm
        self.retriever = retriever
        self.config = config
        self.plan_predict = dspy.Predict(SASResearchPlanSig)
        self.next_action_predict = dspy.Predict(SASNextActionSig)
        self.answer_predict = dspy.Predict(SASFocusedAnswerSig)
        self.chunks_by_id: dict[str, RetrievedChunk] = {}
        self.search_steps: list[dict[str, Any]] = []
        self.tool_errors: list[str] = []
        self.plan: dict[str, Any] = {}

    def _reset(self) -> None:
        self.chunks_by_id.clear()
        self.search_steps.clear()
        self.tool_errors.clear()
        self.plan.clear()

    def _evidence_json(self, *, chunks_per_query: int = 5, chars: int | None = None) -> str:
        limit = chars or self.config.excerpt_chars
        items = []
        for step in self.search_steps:
            chunks = []
            for c in step.get("chunks", [])[:chunks_per_query]:
                chunks.append({"chunk_id": c.chunk_id, "text": c.text[:limit]})
            items.append({"query": step.get("query", ""), "chunks": chunks})
        return json.dumps(items, ensure_ascii=False)

    def _min_searches(self, question: str, profile: str) -> int:
        ql = question.lower()
        complex_markers = [
            "birthplace",
            "where",
            "when",
            "which",
            "not related",
            "not associated",
            "compared",
            "succeeded",
            "performer of",
            "creator",
            "designer",
            "record label",
            "headquarters",
            "battle",
            "derived from",
            "built",
            "developed",
        ]
        target = 3 if profile != "one_hop" or any(x in ql for x in complex_markers) else 2
        return max(1, min(target, self.config.max_searches))

    def _fallback_query(self, question: str, final_question: str) -> str:
        if self.search_steps and final_question and final_question not in {s.get("query", "") for s in self.search_steps}:
            return final_question
        return question

    async def _plan_queries(self, question: str, profile: str) -> tuple[list[str], str]:
        try:
            def _call() -> Any:
                with dspy.context(lm=self.lm):
                    return self.plan_predict(question=question, profile=profile, max_searches=self.config.max_searches)

            pred = await aio_to_thread(_call)
            plan = _safe_json(getattr(pred, "plan_json", ""))
        except Exception as exc:
            self.tool_errors.append(f"plan error: {exc}")
            plan = {}
        raw_queries = plan.get("queries", [])
        queries: list[str] = []
        if isinstance(raw_queries, list):
            for q in raw_queries:
                q = str(q or "").strip()
                if q and q not in queries:
                    queries.append(q)
        if not queries:
            queries = [question]
        final_question = str(plan.get("final_question") or question).strip()
        if final_question and final_question != question and len(queries) < self.config.max_searches:
            if final_question not in queries:
                queries.append(final_question)
        queries = queries[: max(1, self.config.max_searches)]
        answer_type = str(plan.get("answer_type") or _expected_type_hint(question)).strip().lower()
        answer_focus = str(plan.get("answer_focus") or f"Return a minimal {answer_type} span that answers the final question.").strip()
        self.plan = {
            "queries": queries,
            "final_question": final_question,
            "answer_focus": answer_focus,
            "answer_type": answer_type,
            "raw": plan,
        }
        return queries, answer_type

    async def _next_action(self, question: str, profile: str) -> dict[str, Any]:
        try:
            def _call() -> Any:
                with dspy.context(lm=self.lm):
                    return self.next_action_predict(
                        question=question,
                        profile=profile,
                        search_count=len(self.search_steps),
                        max_searches=self.config.max_searches,
                        previous_queries=json.dumps([s.get("query", "") for s in self.search_steps], ensure_ascii=False),
                        evidence_json=self._evidence_json(chunks_per_query=3, chars=360),
                    )

            pred = await aio_to_thread(_call)
            action = _safe_json(getattr(pred, "action_json", ""))
        except Exception as exc:
            self.tool_errors.append(f"next-action error: {exc}")
            action = {}
        final_question = str(action.get("final_question") or self.plan.get("final_question") or question).strip()
        answer_type = str(action.get("answer_type") or self.plan.get("answer_type") or _expected_type_hint(question)).strip().lower()
        answer_focus = str(
            action.get("answer_focus")
            or self.plan.get("answer_focus")
            or f"Return a minimal {answer_type} span that answers the final question."
        ).strip()
        query = str(action.get("query") or "").strip()
        self.plan.update({
            "final_question": final_question,
            "answer_focus": answer_focus,
            "answer_type": answer_type,
        })
        return {
            "action": str(action.get("action") or "search").strip().lower(),
            "query": query,
            "final_question": final_question,
            "answer_type": answer_type,
            "raw": action,
        }

    async def _search(self, query: str) -> None:
        chunks = await self.retriever.retrieve(query, k=self.config.retrieve_topk)
        for chunk in chunks:
            self.chunks_by_id[chunk.chunk_id] = chunk
        self.search_steps.append({"query": query, "chunks": chunks})

    async def _search_loop(self, question: str, profile: str) -> str:
        min_searches = self._min_searches(question, profile)
        expected_type = _expected_type_hint(question)
        self.plan = {
            "final_question": question,
            "answer_focus": f"Return a minimal {expected_type} span that answers the final question.",
            "answer_type": expected_type,
            "min_searches": min_searches,
            "actions": [],
        }
        seen_queries: set[str] = set()
        for _ in range(self.config.max_searches):
            action = await self._next_action(question, profile)
            expected_type = str(action.get("answer_type") or expected_type)
            self.plan.setdefault("actions", []).append(action.get("raw", action))
            if action.get("action") == "answer" and len(self.search_steps) >= min_searches:
                break
            query = str(action.get("query") or "").strip()
            if not query or "[" in query or "]" in query:
                query = self._fallback_query(question, str(self.plan.get("final_question") or question))
            if query in seen_queries:
                query = self._fallback_query(question, str(self.plan.get("final_question") or question))
            if query in seen_queries:
                break
            seen_queries.add(query)
            try:
                await self._search(query)
            except Exception as exc:
                self.tool_errors.append(f"search error: {exc}")
        return expected_type

    def _citation_ok(self, answer: str, support_ids: str) -> tuple[bool, list[str], str]:
        ids = _parse_support_ids(support_ids)
        ids = [x for x in ids if x in self.chunks_by_id]
        if not ids:
            return False, [], "missing_support"
        ans_norm = normalize_answer(answer)
        if not ans_norm:
            return False, ids, "empty_answer"
        ans_tokens = set(ans_norm.split())
        for cid in ids:
            chunk_norm = normalize_answer(self.chunks_by_id[cid].text)
            if ans_norm in chunk_norm:
                return True, ids, "exact"
            if ans_tokens and len(ans_tokens & set(chunk_norm.split())) / len(ans_tokens) >= 0.8:
                return True, ids, "overlap"
        return False, ids, "not_grounded"

    def _answer_chunks(self) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for step in self.search_steps:
            for chunk in step.get("chunks", []):
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                chunks.append(chunk)
        return chunks[:12]

    async def _answer(self, question: str, expected_type: str) -> tuple[str, str]:
        final_question = str(self.plan.get("final_question") or question)
        answer_focus = str(self.plan.get("answer_focus") or "")
        try:
            def _call() -> Any:
                with dspy.context(lm=self.lm):
                    return self.answer_predict(
                        original_question=question,
                        final_question=final_question,
                        answer_focus=answer_focus,
                        expected_answer_type=expected_type,
                        chunks_json=json.dumps(
                            [
                                {"chunk_id": c.chunk_id, "text": c.text[: self.config.excerpt_chars]}
                                for c in self._answer_chunks()
                            ],
                            ensure_ascii=False,
                        ),
                    )

            pred = await aio_to_thread(_call)
            obj = _safe_json(getattr(pred, "extraction_json", ""))
        except Exception as exc:
            self.tool_errors.append(f"answer error: {exc}")
            obj = {}
        return str(obj.get("answer_span", "")).strip(), str(obj.get("evidence_chunk_id", "")).strip()

    async def run(self, question: str) -> dict[str, Any]:
        self._reset()
        question = str(question or "").strip()
        t0 = time.time()
        lm_start = len(getattr(self.lm, "history", []))
        profile = classify(question)
        expected_type = _expected_type_hint(question)
        queries, planned_type = await self._plan_queries(question, profile)
        expected_type = planned_type or expected_type

        for query in queries:
            try:
                await self._search(query)
            except Exception as exc:
                self.tool_errors.append(f"search error: {exc}")

        raw_answer, raw_support = await self._answer(question, expected_type)
        accepted, support_ids, citation_reason = self._citation_ok(raw_answer, raw_support)
        answer = _clean_final_answer(question, raw_answer)
        total_tokens = _tokens_since(self.lm, lm_start)
        metadata = {
            "method": "true_sas_research_plan",
            "profile": profile,
            "total_tokens": total_tokens,
            "root_tokens": total_tokens,
            "sub_tokens": 0,
            "elapsed_s": round(time.time() - t0, 3),
            "search_calls": len(self.search_steps),
            "search_queries": [s.get("query", "") for s in self.search_steps],
            "retrieved_chunks": list(self.chunks_by_id),
            "tool_errors": list(self.tool_errors),
            "route": "sas",
            "topology": "single_agent",
            "n_nodes": 0,
            "expected_type": expected_type,
            "support_ids": support_ids,
            "citation_accepted": accepted,
            "citation_reason": citation_reason,
            "max_searches": self.config.max_searches,
            "retrieve_topk": self.config.retrieve_topk,
            "search_plan": self.plan,
        }
        return {
            "question": question,
            "predicted_answer": answer,
            "answer": answer,
            "trajectory": {
                "plan": self.plan,
                "search_steps": [
                    {
                        "query": s.get("query", ""),
                        "chunks": [
                            {"chunk_id": c.chunk_id, "score": c.score, "text": c.text[: self.config.excerpt_chars]}
                            for c in s.get("chunks", [])
                        ],
                    }
                    for s in self.search_steps
                ],
                "raw_answer": raw_answer,
                "raw_support_ids": raw_support,
            },
            "metadata": metadata,
            "readable_trace": json.dumps(
                {
                    "question": question,
                    "final_question": self.plan.get("final_question", question),
                    "answer": answer,
                    "queries": metadata["search_queries"],
                },
                ensure_ascii=False,
            ),
        }
