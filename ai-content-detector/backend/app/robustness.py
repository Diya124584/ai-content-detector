"""Robustness testing helpers.

Compare a detector's output on an original text and a modified version
(paraphrased AI text, human-edited AI text, text from an unseen generator,
a different domain, ...). The modified texts must be supplied by you; this
module does not fabricate paraphrases. A few trivial synthetic perturbations
are provided for smoke-testing only; they are NOT a substitute for real
paraphrase / human-edit data.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .scoring import LABEL_UNCERTAIN
from .trust_score import RobustnessReport

PERTURBATION_TYPES = ("original", "paraphrased_ai", "human_edited_ai", "unseen_generator", "different_domain", "synthetic_typos", "synthetic_shuffle")


@dataclass
class ComparisonResult:
    original_prediction: str
    modified_prediction: str
    prediction_changed: bool
    original_ai_probability: float
    modified_ai_probability: float
    probability_change: float  # modified - original
    original_detection_score: float
    modified_detection_score: float
    score_change: float
    perturbation_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_texts(detect_fn: Callable[[str], Any], original_text: str, modified_text: str,
                  perturbation_type: Optional[str] = None) -> ComparisonResult:
    """``detect_fn`` returns an object with prediction / ai_probability / detection_score
    (e.g. ``TextDetectionPipeline.analyze``)."""
    a, b = detect_fn(original_text), detect_fn(modified_text)
    return ComparisonResult(
        a.prediction, b.prediction, a.prediction != b.prediction,
        a.ai_probability, b.ai_probability, b.ai_probability - a.ai_probability,
        a.detection_score, b.detection_score, b.detection_score - a.detection_score, perturbation_type,
    )


def summarize(results: Sequence[ComparisonResult], dataset: str = "unspecified") -> RobustnessReport:
    """Aggregate pair comparisons into a report usable by the Trust Score."""
    if not results:
        raise ValueError("no comparison results")
    n = len(results)
    stable = sum(1 for r in results if not r.prediction_changed) / n
    mean_abs = sum(abs(r.probability_change) for r in results) / n
    types = sorted({r.perturbation_type for r in results if r.perturbation_type})
    return RobustnessReport(n, stable, mean_abs, dataset, types)


# -------- trivial synthetic perturbations (smoke tests only; seeded) --------
def synthetic_typos(text: str, rate: float = 0.03, seed: int = 0) -> str:
    rng = random.Random(seed)
    chars = list(text)
    letters = [i for i, c in enumerate(chars) if c.isalpha()]
    for i in rng.sample(letters, k=int(len(letters) * rate)):
        chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def synthetic_sentence_shuffle(text: str, seed: int = 0) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    random.Random(seed).shuffle(parts)
    return " ".join(parts)
