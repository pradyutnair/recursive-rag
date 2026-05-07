"""Prompts for SPARC-RAG agents.

These mirror the paper's appendix prompts (Yang et al. 2026, arXiv:2602.00083).
The Multi-Path Dispatch Rewrite prompt's `bm25|dense` strategy field is
restricted to `dense` because we only reproduce with dense retrieval.

Each prompt instructs the model to terminate with the literal tag `[END]`,
which we use as the vLLM stop sequence.
"""
from __future__ import annotations


CONTEXT_MANAGER_UPDATE_PROMPT = """Act as the context manager for a Retrieval-Augmented Generation (RAG) system. Your job is to maintain a single, up-to-date note that contains all the information relevant to answering the original query. Please ensure that the note includes all original text information useful for answering the question.
Steps:
- Based on the retrieved documents, supplement the notes with content not yet included but useful for answering the question.
- Resolve conflicts: if statements disagree, keep the most reliable or recent version.
End your response with the literal tag [END].
Original query: {query}
Old note: {note}
New information: {new_context}
Updated note:"""


ANSWER_GENERATOR_PROMPT = """Answer the question based on the given notes.
Output ONLY the exact answer in as few words as possible.
Do not include the question, reasoning, or any extra text.
End your response with the literal tag [END].
The following are given notes:
{note}
Question: {query}
Answer:"""


MULTI_PATH_DISPATCH_PROMPT = """You are an intelligent assistant in a Retrieval-Augmented Generation (RAG) system. Your goal is to (a) diagnose retrieval needs for the current question and (b) produce exactly {N} rewritten queries.
Information:
- Original Query: {query}
- Current Query: {current_query}
- Context: {context}
Instructions:
1. Use the context to reflect on what is missing to answer the query. Think about both the big picture and the small atomic facts that might need verification.
2. Generate exactly {N} rewritten queries.
- Do not just paraphrase — each query should explore a different angle, granularity, or fact.
- Avoid near-duplicates.
- Each query must serve a distinct retrieval purpose.
Output Format (strict):
1. First provide your analysis and rationale in a <think> block, including per-item justification.
2. Then output exactly {N} query rewrites using the following structure:
<queries>
<item rank="1"><query>...</query></item>
<item rank="2"><query>...</query></item>
...
<item rank="{N}"><query>...</query></item>
</queries>
3. End your response with the literal tag [END].
Output:"""


ANSWER_SELECTION_PROMPT = """Question: {question}
All Generated Answers:
{answer_blocks}
Based on all the answers and reasoning provided, select the best answer. Consider accuracy and relevance to the question. Give the final answer directly. Then, provide a direct, concise, and accurate answer inside <answer> </answer> tags. End your response with the literal tag [END].
Final answer:"""


CONTEXT_MERGING_PROMPT = """You are an expert at combining contexts in a Retrieval Augmented Generation system for answering question {question}. Here are several notes: {reasoning_list}.
Combine the above notes into a single note that includes all information and is useful for final answer generation. Please ensure that the note includes all original text information useful for answering the question. End your response with the literal tag [END].
Your response:"""


# The paper supplies an ANSWER EVALUATOR with STOP/CONTINUE outputs but the
# concrete prompt is not in the appendix excerpt provided. We use a minimal
# faithful version that only emits the decision token.
ANSWER_EVALUATOR_PROMPT = """You are an answer evaluator inside a Retrieval-Augmented Generation system. Per Section 3.2 of SPARC-RAG, you take as input the original question, the current sub-query, the candidate answer, and the current note (memory). Decide whether the reasoning process should STOP (the candidate answer is correct, complete, and well-grounded in the note for the original question) or CONTINUE (the candidate answer is wrong, under-specified, or not supported by the note). Be conservative: when uncertain, prefer CONTINUE.
Original question: {question}
Current sub-query: {current_query}
Note:
{note}
Candidate answer: {answer}
Output exactly one token (STOP or CONTINUE) inside <decision> </decision> tags. End your response with the literal tag [END].
Decision:"""
