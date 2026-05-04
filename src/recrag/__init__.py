"""Force-hard no-critic recursive RAG base."""

import os
from pathlib import Path

os.environ.setdefault("DSPY_CACHEDIR", str(Path.cwd() / ".dspy_cache"))

from .adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from .contracts import CitationCheck, HopFinding, RetrievedChunk, normalize_answer
from .metric import RewardBreakdown, composite_reward
from .profile import classify, expected_hops

__all__ = [
    "AdaptiveConfig",
    "AdaptiveRecursivePipeline",
    "CitationCheck",
    "HopFinding",
    "RetrievedChunk",
    "normalize_answer",
    "RewardBreakdown",
    "composite_reward",
    "classify",
    "expected_hops",
]
