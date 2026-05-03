"""Plan*RAG-style DAG planner with adaptive recursive expansion.

The planner emits a DAG plan as JSON. The executor traverses depth-by-depth,
runs same-depth nodes in parallel via asyncio.gather, and substitutes <A.I.J>
tags from parent answers. Each node is a sub-agent (the existing _hop_async
extract+rewrite loop). If a node ends with low extraction confidence AND its
question contains a bridge phrase AND the recursion budget is non-zero, the
node spawns a sub-DAG.

Schema (JSON the planner is asked to emit):
  {
    "nodes": [
      {"id": "Q1.1", "question": "...", "expected_type": "person", "depends_on": []},
      {"id": "Q1.2", "question": "...", "expected_type": "place", "depends_on": []},
      {"id": "Q2.1", "question": "Where did <A1.1> meet <A1.2>?", "expected_type": "place",
       "depends_on": ["Q1.1", "Q1.2"]}
    ],
    "final_node": "Q2.1"
  }

Tag substitution: <A1.1> in a question is replaced by the answer of node Q1.1
at execution time, immediately before the node runs.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import dspy

from .contracts import HopFinding
from .retriever import Retriever
from .tools import ToolRuntime, _hop_async  # reuse the proven retrieve/extract/rewrite loop

# Accept: <A1.1>, <AI.Q1>, <A1>, <AQ1>, <AI.1>, <ANS:Q1>, etc.
_TAG_RE = re.compile(r"<\s*A(?:I|NS|ns)?[\s:.]*(?:Q\s*)?(\d+)(?:\s*\.\s*(\d+))?\s*>", re.IGNORECASE)
# Accept either "Q1" or "Q1.1" (compact and Plan*RAG-style)
_NODE_RE = re.compile(r"^Q\s*(\d+)(?:\s*\.\s*(\d+))?\s*$", re.IGNORECASE)
_BRIDGE_TOKENS = (
    " of ", " who ", " that ", " which ", " where ", " when ", " by ", " from ",
    " whose ", " in which ", " for the ", " for which ",
)


def _looks_like_bridge(q: str) -> bool:
    ql = " " + (q or "").lower() + " "
    return any(t in ql for t in _BRIDGE_TOKENS)


@dataclass
class PlannedNode:
    id: str
    question: str
    expected_type: str
    depends_on: list[str]
    depth: int = 0
    answer: str = ""
    confidence: float = 0.0
    chunk_id: str = ""
    queries_used: list[str] = field(default_factory=list)
    expanded_into: list[str] = field(default_factory=list)  # ids of children if recursed
    raw_question: str = ""
    retrieval_query: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_type": self.expected_type,
            "depends_on": list(self.depends_on),
            "depth": self.depth,
            "answer": self.answer,
            "confidence": self.confidence,
            "chunk_id": self.chunk_id,
            "queries_used": list(self.queries_used),
            "expanded_into": list(self.expanded_into),
            "raw_question": self.raw_question,
            "retrieval_query": self.retrieval_query,
        }


@dataclass
class PlanRun:
    nodes: dict[str, PlannedNode] = field(default_factory=dict)
    final_node: str = ""
    plan_attempts: int = 0
    plan_errors: list[str] = field(default_factory=list)

    def topo_layers(self) -> list[list[str]]:
        """Return node ids grouped by topological depth (parents resolved first)."""
        ready: list[list[str]] = []
        done: set[str] = set()
        remaining = set(self.nodes)
        while remaining:
            layer = sorted([nid for nid in remaining if all(d in done for d in self.nodes[nid].depends_on)])
            if not layer:
                break  # cycle or missing dep; bail
            for nid in layer:
                self.nodes[nid].depth = len(ready)
            ready.append(layer)
            done.update(layer)
            remaining -= set(layer)
        return ready


class PlanDAGSig(dspy.Signature):
    """Decompose a multi-hop question into a minimal DAG of atomic sub-questions.

    A sub-question is "atomic" iff one retrieval round is enough to answer it.
    Use as few nodes as possible. Single-fact questions need exactly one node.
    Independent facts (e.g. comparison of two entities) go at the same depth as
    parallel siblings. Sequential bridges chain via depends_on. Reference parent
    answers in child questions using <AI.J> tags, e.g. <A1.1>.

    Return strict JSON only:
    {"nodes": [{"id":"Q1.1","question":"...","expected_type":"person|place|date|number|title|organization|yes_no|entity","depends_on":[]}, ...], "final_node":"QX.Y"}

    Rules:
    - Use Q1.1, Q1.2 for root (depth-0) nodes.
    - Use Q2.1, Q2.2 for nodes at depth 1, etc.
    - Maximum 6 nodes. Maximum 3 levels of depth.
    - One node only when the question is genuinely single-hop.
    - final_node must be the node whose answer is the final answer to the user question.
    - Never use the answer of a node before it executes; reference via <AI.J> tags.
    """

    question: str = dspy.InputField()
    profile: str = dspy.InputField(desc="Question profile: one_hop, parallel_compare, bridge_2hop, bridge_3hop_plus, temporal, numeric, yes_no")
    experience: str = dspy.InputField(desc="Profile-conditioned strategy hints (may be empty)")
    budget_hint: str = dspy.InputField(desc="Runtime budget: tight, normal, or rich")
    plan_json: str = dspy.OutputField(desc="Strict JSON DAG schema")


class RouteQuestionSig(dspy.Signature):
    """Route a question to the cheap SAS lane or the full DAG lane.

    Return strict JSON only:
    {"route":"easy|hard","reason":"short reason"}

    Choose easy only when a single investigator with retrieval rewrites should
    answer the final target directly. Choose hard for bridge chains, unnamed
    bridge entities, nested "of/that/which/who" dependencies, comparisons,
    intersections, temporal reasoning, arithmetic, or any ambiguity.
    """

    question: str = dspy.InputField()
    profile: str = dspy.InputField(desc="Heuristic profile")
    experience: str = dspy.InputField(desc="Profile-conditioned routing hints")
    budget_hint: str = dspy.InputField(desc="Runtime budget: tight, normal, or rich")
    route_json: str = dspy.OutputField(desc='Strict JSON: {"route":"easy|hard","reason":"..."}')


class SynthesizeFinalSig(dspy.Signature):
    """Read the full DAG trace and produce the final concise answer to the user question.

    You see: question, expected answer type, and a list of resolved nodes with
    their question, answer, confidence, and source chunk_id. Choose the final
    answer that DIRECTLY answers the user question (not an intermediate bridge).
    Preserve full canonical names, full dates, full award/category names. Output
    one short span (usually 1-10 words). If the trace is contradictory or weak,
    return the best-supported concise span; never refuse, never say the question
    is invalid.

    Also return the chunk_ids you used as evidence (CSV) for the citation gate.
    """

    question: str = dspy.InputField()
    expected_type: str = dspy.InputField()
    trace_json: str = dspy.InputField(desc="JSON list of resolved nodes")
    budget_hint: str = dspy.InputField(desc="Runtime budget: tight, normal, or rich")
    final_answer: str = dspy.OutputField(desc="The concise final span")
    support_ids: str = dspy.OutputField(desc="CSV of chunk_ids cited")


class CritiqueFinalSig(dspy.Signature):
    """Verify whether the final answer directly resolves the original question.

    Return strict JSON:
    {"verdict":"accept|flag","reason":"short actionable reason"}

    Flag answers that are only bridge entities, have the wrong answer type, are
    unsupported by the trace, contradict the trace, or skip the final target.
    """

    question: str = dspy.InputField()
    expected_type: str = dspy.InputField()
    trace_json: str = dspy.InputField(desc="JSON list of resolved DAG nodes")
    final_answer: str = dspy.InputField(desc="Synthesized final answer")
    verdict_json: str = dspy.OutputField(desc="Strict JSON verdict")


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    # Try direct JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Try to extract a JSON object substring
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _canon_node_id(raw: str, fallback_index: int) -> str | None:
    """Normalize various node-id shapes to canonical Qd.d form. Returns None if invalid."""
    m = _NODE_RE.match(str(raw or "").strip())
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else 1  # "Q1" -> "Q1.1"
    return f"Q{a}.{b}"


def parse_plan(plan_text: str) -> PlanRun | None:
    obj = _safe_parse_json(plan_text)
    if not obj:
        return None
    nodes_raw = obj.get("nodes", [])
    if not isinstance(nodes_raw, list) or not nodes_raw:
        return None
    # First pass: assign canonical ids and a rename map for deps
    rename: dict[str, str] = {}
    canon: list[tuple[str, dict]] = []
    seen_canon: set[str] = set()
    for i, n in enumerate(nodes_raw[:8]):  # hard cap
        if not isinstance(n, dict):
            continue
        raw_id = str(n.get("id", "")).strip()
        cid = _canon_node_id(raw_id, i)
        if not cid:
            continue
        # Disambiguate collisions by bumping the .j suffix
        base_a = cid.split(".")[0]
        j = int(cid.split(".")[1])
        while cid in seen_canon:
            j += 1
            cid = f"{base_a}.{j}"
        seen_canon.add(cid)
        rename[raw_id] = cid
        canon.append((cid, n))
    if not canon:
        return None
    nodes: dict[str, PlannedNode] = {}
    for cid, n in canon:
        q = str(n.get("question", "")).strip()
        if not q:
            continue
        deps_raw = n.get("depends_on") or n.get("depends") or n.get("parents") or []
        deps_in = [str(x) for x in deps_raw if isinstance(x, (str, int))] if isinstance(deps_raw, list) else []
        deps: list[str] = []
        for d in deps_in:
            mapped = rename.get(d.strip())
            if mapped is None:
                cd = _canon_node_id(d, 0)
                if cd in seen_canon:
                    mapped = cd
            if mapped:
                deps.append(mapped)
        et = str(n.get("expected_type") or n.get("type") or "auto").strip().lower() or "auto"
        rq = str(n.get("retrieval_query") or n.get("search_query") or n.get("query") or "").strip()
        nodes[cid] = PlannedNode(id=cid, question=q, raw_question=q, expected_type=et, depends_on=deps, retrieval_query=rq)
    if not nodes:
        return None
    raw_final = str(obj.get("final_node") or obj.get("final") or "").strip()
    final = rename.get(raw_final) or _canon_node_id(raw_final, 0) or ""
    if final not in nodes:
        final = sorted(nodes.keys())[-1]
    return PlanRun(nodes=nodes, final_node=final)


def substitute_tags(question: str, resolved: dict[str, str]) -> str:
    """Replace <A1.1> / <AI.Q1> / <A1> tags by parent answers (raw text).

    Resolution order:
      1. Try Qa.b lookup if both groups present.
      2. Try Qa.1 fallback (compact "Q1" form).
      3. Try matching any resolved key whose numeric prefix == a (best effort).
    """

    keys_by_prefix: dict[int, list[str]] = {}
    for k in resolved:
        m = _NODE_RE.match(k)
        if m:
            keys_by_prefix.setdefault(int(m.group(1)), []).append(k)

    def _sub(m: re.Match) -> str:
        a_raw, b_raw = m.group(1), m.group(2)
        try:
            a = int(a_raw)
        except Exception:
            return m.group(0)
        if b_raw is not None:
            try:
                b = int(b_raw)
                key = f"Q{a}.{b}"
                if key in resolved:
                    return resolved[key]
            except Exception:
                pass
        # Fallback to Qa.1
        key = f"Q{a}.1"
        if key in resolved:
            return resolved[key]
        # Or any key with numeric prefix a
        if a in keys_by_prefix:
            return resolved[keys_by_prefix[a][0]]
        return m.group(0)

    return _TAG_RE.sub(_sub, question)


@dataclass
class AdaptiveDAGRunner:
    retriever: Retriever
    sub_lm: dspy.LM
    state: ToolRuntime
    max_nodes: int = 8
    max_recursion_depth: int = 2
    tau_recurse: float = 0.5
    max_searches: int = 3

    async def run_node(self, node: PlannedNode) -> None:
        """Execute a single node (retrieve + extract + maybe rewrite)."""
        finding_json = await _hop_async(
            node.question, node.expected_type, self.retriever, self.sub_lm, self.state,
            max_attempts=self.max_searches, initial_query=node.retrieval_query or None,
        )
        try:
            data = json.loads(finding_json)
        except Exception:
            data = {}
        node.answer = str(data.get("answer", ""))
        node.confidence = float(data.get("confidence", 0.0) or 0.0)
        node.chunk_id = str(data.get("evidence_chunk_id", ""))
        node.queries_used = list(data.get("queries_used", []) or [])


    async def maybe_recurse(
        self,
        node: PlannedNode,
        plan_one: Callable[[str, str, str], Awaitable[PlanRun | None]],
        recursion_depth: int,
        plan: PlanRun,
    ) -> None:
        """If extraction is weak AND the question is bridge-shaped AND budget remains,
        spawn a sub-DAG for this node and replace its answer with the sub-DAG's final.
        """
        if recursion_depth >= self.max_recursion_depth:
            return
        if node.confidence >= self.tau_recurse:
            return
        if not _looks_like_bridge(node.question):
            return
        if len(plan.nodes) >= self.max_nodes:
            return
        sub = await plan_one(node.question, node.expected_type, "(recurse on low-confidence bridge)")
        if not sub or not sub.nodes:
            return
        # Inline the sub-DAG: rename ids to avoid collision (prefix with parent.id)
        prefix = node.id.replace("Q", "R")
        rename: dict[str, str] = {}
        for sid in list(sub.nodes.keys()):
            rename[sid] = f"{prefix}_{sid}"
        for old, new in rename.items():
            n = sub.nodes.pop(old)
            n.id = new
            n.depends_on = [rename.get(d, d) for d in n.depends_on]
            sub.nodes[new] = n
        new_final = rename.get(sub.final_node, sub.final_node)
        # Execute the sub-DAG bottom-up
        sub_layers = sub.topo_layers()
        for layer in sub_layers:
            await asyncio.gather(*[self.run_node(sub.nodes[nid]) for nid in layer])
        # Adopt nodes into parent plan for trace; mark expansion linkage
        for new_id, n in sub.nodes.items():
            plan.nodes[new_id] = n
            node.expanded_into.append(new_id)
        # Replace this node's answer with the sub-DAG final answer if better
        sub_final = sub.nodes.get(new_final)
        if sub_final and sub_final.confidence >= node.confidence:
            node.answer = sub_final.answer
            node.confidence = sub_final.confidence
            node.chunk_id = sub_final.chunk_id

async def execute_plan(
    plan: PlanRun,
    runner: AdaptiveDAGRunner,
    plan_one: Callable[[str, str, str], Awaitable[PlanRun | None]] | None,
    recursion_depth: int = 0,
) -> None:
    layers = plan.topo_layers()
    for layer in layers:
        # Resolve tag substitutions for this layer using ANY ancestors that have answers
        resolved: dict[str, str] = {nid: plan.nodes[nid].answer for nid in plan.nodes if plan.nodes[nid].answer}
        for nid in layer:
            n = plan.nodes[nid]
            n.question = substitute_tags(n.raw_question, resolved)
            if n.retrieval_query:
                n.retrieval_query = substitute_tags(n.retrieval_query, resolved)
        await asyncio.gather(*[runner.run_node(plan.nodes[nid]) for nid in layer])
        if plan_one is not None:
            # After running the layer, attempt recursion for low-conf bridge nodes
            for nid in layer:
                await runner.maybe_recurse(plan.nodes[nid], plan_one, recursion_depth, plan)
