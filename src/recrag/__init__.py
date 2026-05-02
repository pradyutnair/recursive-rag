"""Recursive adaptive RAG."""

from .adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from .contracts import CitationCheck, HopFinding, RetrievedChunk, normalize_answer
from .metric import RewardBreakdown, composite_reward, feedback_text
from .pipeline import PipelineConfig, ReactRagPipeline
from .profile import classify, expected_hops
from .trace import build_readable_trace, build_structured_stats

__all__ = [
    "AdaptiveConfig", "AdaptiveRecursivePipeline",
    "CitationCheck", "HopFinding", "RetrievedChunk", "normalize_answer",
    "PipelineConfig", "ReactRagPipeline",
    "RewardBreakdown", "composite_reward", "feedback_text",
    "classify", "expected_hops",
    "build_readable_trace", "build_structured_stats",
]
