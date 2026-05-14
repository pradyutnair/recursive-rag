from __future__ import annotations

import dspy


class SummarizeAdaptiveRollout(dspy.Signature):
    """Analyze one Adaptive Recursive RAG rollout and extract transferable strategy lessons.

    The trace shows the DAG plan: each node has id (Qd.d), question, expected_type, depth,
    answer, confidence, source chunk_id. The synthesizer then produced a final span and
    a citation. The reward includes EM/F1/contain plus grounding/shape and a token-cost
    efficiency factor.

    Focus on:
    - Did the planner pick the right topology (1-hop, 2-hop bridge, parallel sibling, 3+hop)?
    - Did each node return a high-confidence (>=0.7) and grounded answer?
    - Did the synthesizer pick the correct final-target span (NOT an intermediate bridge)?
    - Did the run waste tokens (over-decomposition, redundant retries)?
    - What is transferable to OTHER questions of the SAME profile?

    Output a 2-4 sentence summary of reusable strategy insights tagged with the profile
    of this question. Do NOT name specific entities; generalize to question patterns.
    """

    question: str = dspy.InputField()
    profile: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    trajectory: str = dspy.InputField(desc="Readable DAG trace + synthesizer output + citation")
    reward_breakdown: str = dspy.InputField(desc="JSON of em/f1/contain/grounded/shape/quality/efficiency/composite/tokens")
    summary: str = dspy.OutputField(desc="2-4 sentence transferable strategy insight, prefixed with [profile=...]")


class ExtractGroupOps(dspy.Signature):
    """Given G rollout summaries for ONE question (different temperatures), extract experience
    library operations.

    Compare winning trajectories (high composite reward) against losing ones. What did
    winners do differently? What patterns are TRANSFERABLE to other questions of the same
    profile? Do NOT add question-specific facts.

    Output a JSON list of ops:
      [
        {"op": "ADD", "text": "...", "rationale": "...", "profile": "bridge_2hop|parallel_compare|temporal|numeric|bridge_3hop_plus|one_hop|yes_no|any"},
        {"op": "MODIFY", "id": "E-001", "text": "...", "profile": "..."},
        {"op": "DELETE", "id": "E-003"},
        {"op": "KEEP", "id": "E-002"}
      ]
    Only ADD entries that are genuinely transferable. Maximum 2 new entries per ops list.
    """

    summaries: str = dspy.InputField(desc="Newline-separated rollout summaries for one question, with composite reward")
    profile: str = dspy.InputField(desc="The shared profile of all rollouts in this group")
    current_library: str = dspy.InputField(desc="Current experience library entries (E-NNN [profile|u=N]: text)")
    ops_json: str = dspy.OutputField(desc="JSON list of ADD/MODIFY/DELETE/KEEP operations")


class OptimizeBatch(dspy.Signature):
    """Consolidate experience libraries from a batch of questions into one concise library.

    Merge duplicate or overlapping entries. Remove entries that contradict each other.
    Keep the library at most ~24 entries total, ~6 per profile. Each entry should be a
    transferable strategy principle, not a question-specific fact. Preserve the profile
    tag of each entry.
    """

    batch_proposals: str = dspy.InputField(desc="Newline-separated library snapshots from each question in the batch")
    current_library: str = dspy.InputField()
    merged_library: str = dspy.OutputField(desc="One entry per line: E-NNN [profile|u=N]: strategy text")
