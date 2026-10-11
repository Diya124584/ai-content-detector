"""Shared test doubles. Fakes replace the heavy models so tests check
plumbing / schema / logic; they say nothing about real detection accuracy."""
from __future__ import annotations

from app.config import Settings
from app.detector import ClassifierOutput
from app.perplexity import PerplexityResult
from app.pipeline import TextDetectionPipeline

SAMPLE = (
    "The weather was lovely this morning, so we walked down to the harbour. "
    "Gulls were fighting over scraps near the fish market. My brother insisted on buying coffee, "
    "although the queue was absurdly long. Eventually we sat on a wall and watched the ferries. "
    "It was one of those unremarkable days that somehow stays with you for years."
)


class FakeRoberta:
    is_loaded = True

    def __init__(self, p_ai: float = 0.8) -> None:
        self.p_ai = p_ai

    def predict(self, text: str) -> ClassifierOutput:
        return ClassifierOutput(self.p_ai, 1 - self.p_ai, 1, False, "fake", "pretrained-baseline", "fake mapping")


class FakePerplexity:
    is_loaded = True

    def __init__(self, ppl: float = 20.0) -> None:
        self.ppl = ppl

    def compute(self, text: str) -> PerplexityResult:
        return PerplexityResult(self.ppl, 50, False, None, "fake-lm")


def make_pipeline(p_ai: float = 0.8, ppl: float = 20.0, **kw) -> TextDetectionPipeline:
    return TextDetectionPipeline(Settings(), roberta=FakeRoberta(p_ai), perplexity=FakePerplexity(ppl), **kw)
