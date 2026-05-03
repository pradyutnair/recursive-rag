from __future__ import annotations

import asyncio
import threading

import dspy

from .adaptive_pipeline import AdaptiveRecursivePipeline
from .pipeline import ReactRagPipeline
from .sync_pipeline import SyncAdaptivePipeline

_thread_loop = threading.local()


def _run_async(coro):
    loop = getattr(_thread_loop, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_loop.loop = loop
    return loop.run_until_complete(coro)


class ReactProgram(dspy.Module):
    """Thin wrapper exposing the ReAct predictor for GEPA (legacy)."""

    def __init__(self, pipeline: ReactRagPipeline):
        super().__init__()
        self.pipeline = pipeline
        self.react = pipeline.react.react
        self.extract = pipeline.react.extract

    def forward(self, question: str):
        return self.pipeline.react(question=question)


class AdaptiveProgram(dspy.Module):
    """GEPA-compatible wrapper around the *sync* AdaptiveRecursivePipeline.

    Uses SyncAdaptivePipeline to avoid asyncio interactions with dspy.GEPA's
    thread-based optimization. Exposes named `planner` and `synthesizer`
    Predict modules; before every forward we sync them back into the pipeline
    so GEPA-created candidates (which may rebind those attributes) are used.
    """

    def __init__(self, pipeline: SyncAdaptivePipeline):
        super().__init__()
        self.pipeline = pipeline
        self.router = pipeline.route_predict
        self.planner = pipeline.plan_predict
        self.synthesizer = pipeline.synth_predict
        self.critic = pipeline.critic_predict

    def _sync(self) -> None:
        self.pipeline.route_predict = self.router
        self.pipeline.plan_predict = self.planner
        self.pipeline.synth_predict = self.synthesizer
        self.pipeline.critic_predict = self.critic

    def forward(self, question: str, budget_hint: str = "normal") -> dspy.Prediction:
        self._sync()
        result = self.pipeline.run(question, budget_hint=budget_hint)
        return dspy.Prediction(
            answer=result.get("answer", ""),
            trajectory=result.get("trajectory", {}),
            metadata=result.get("metadata", {}),
        )


class AsyncAdaptiveProgram(dspy.Module):
    """Legacy async-pipeline wrapper for non-GEPA usage."""

    def __init__(self, pipeline: AdaptiveRecursivePipeline):
        super().__init__()
        self.pipeline = pipeline
        self.router = pipeline.route_predict
        self.planner = pipeline.plan_predict
        self.synthesizer = pipeline.synth_predict

    def _sync(self) -> None:
        self.pipeline.route_predict = self.router
        self.pipeline.plan_predict = self.planner
        self.pipeline.synth_predict = self.synthesizer

    def forward(self, question: str, budget_hint: str = "normal") -> dspy.Prediction:
        self._sync()
        result = _run_async(self.pipeline.run(question, budget_hint=budget_hint))
        return dspy.Prediction(
            answer=result.get("answer", ""),
            trajectory=result.get("trajectory", {}),
            metadata=result.get("metadata", {}),
        )
