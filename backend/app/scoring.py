"""Score fusion: combine independent signals into one detection score.

Each signal is first mapped to an "AI-ness" score in [0, 1]:
  roberta        - P(AI) from the pretrained baseline (already a probability)
  perplexity     - logistic map on log-perplexity (lower perplexity -> more AI)
  linguistic     - weighted mean of burstiness / TTR / repetition mappings
  feature_model  - P(AI) from the trained TF-IDF + LR model (if available)

The final score is a weighted mean over the signals that are *available*
(weights renormalised). The default weights and mapping constants are
PROTOTYPE PRIORS chosen for a runnable system. They are not scientifically
optimal and are not fitted to data. Use ``fit_fusion_weights`` on a labelled
validation set to learn weights, and a calibrator (trust_score.LogitCalibrator)
to turn the fused score into a calibrated probability.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .features import TextFeatures

LABEL_AI = "AI-generated"
LABEL_HUMAN = "Human-written"
LABEL_UNCERTAIN = "Uncertain"
COMPONENTS = ("roberta", "perplexity", "linguistic", "feature_model")


@dataclass
class FusionConfig:
    # --- top-level fusion weights (prototype, configurable) ---
    roberta_weight: float = 0.40
    perplexity_weight: float = 0.20
    linguistic_weight: float = 0.15
    feature_model_weight: float = 0.25
    # --- decision thresholds (three-state output) ---
    ai_threshold: float = 0.65  # score >= this -> AI-generated
    human_threshold: float = 0.35  # score <= this -> Human-written; in between -> Uncertain
    # --- perplexity -> AI-score mapping: sigmoid((ln(center) - ln(ppl)) / scale) ---
    perplexity_center: float = 40.0  # prior pivot for GPT-2 perplexity; model/domain dependent
    perplexity_scale: float = 0.7
    # --- linguistic sub-signals: sigmoid(direction * (value - center) / scale) ---
    burstiness_center: float = 0.5  # low burstiness (uniform sentences) -> more AI-like
    burstiness_scale: float = 0.2
    ttr_center: float = 0.6  # low TTR -> more AI-like; very length dependent
    ttr_scale: float = 0.1
    repetition_center: float = 0.05  # high n-gram repetition -> more AI-like
    repetition_scale: float = 0.05
    burstiness_weight: float = 0.4
    ttr_weight: float = 0.2
    repetition_weight: float = 0.4
    # --- reliability gates: ignore a feature when the text is too short for it ---
    min_sentences_for_burstiness: int = 3
    min_words_for_ttr: int = 30
    min_words_for_repetition: int = 30

    def weights(self) -> Dict[str, float]:
        return {
            "roberta": self.roberta_weight,
            "perplexity": self.perplexity_weight,
            "linguistic": self.linguistic_weight,
            "feature_model": self.feature_model_weight,
        }

    def validate(self) -> None:
        if not (0.0 <= self.human_threshold <= self.ai_threshold <= 1.0):
            raise ValueError("require 0 <= human_threshold <= ai_threshold <= 1")
        if any(w < 0 for w in self.weights().values()):
            raise ValueError("weights must be non-negative")

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "FusionConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        cfg.validate()
        return cfg


@dataclass
class RawSignals:
    roberta_ai: Optional[float] = None
    perplexity: Optional[float] = None
    features: Optional[TextFeatures] = None
    feature_model_ai: Optional[float] = None


@dataclass
class FusionResult:
    prediction: str
    ai_probability: float
    human_probability: float
    detection_score: float
    signals: Dict[str, Optional[float]]
    signal_scores: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    contributions: Dict[str, float]
    notes: List[str] = field(default_factory=list)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def perplexity_ai_score(ppl: Optional[float], cfg: FusionConfig) -> Optional[float]:
    if ppl is None or not math.isfinite(ppl) or ppl <= 0:
        return None
    return _sigmoid((math.log(cfg.perplexity_center) - math.log(ppl)) / cfg.perplexity_scale)


def linguistic_subscores(f: Optional[TextFeatures], cfg: FusionConfig) -> Dict[str, Optional[float]]:
    """AI-ness of each linguistic feature, or None when the text is too short."""
    out: Dict[str, Optional[float]] = {"burstiness": None, "ttr": None, "repetition": None}
    if f is None:
        return out
    if f.sentence_count >= cfg.min_sentences_for_burstiness:
        out["burstiness"] = _sigmoid((cfg.burstiness_center - f.burstiness) / cfg.burstiness_scale)
    if f.word_count >= cfg.min_words_for_ttr:
        out["ttr"] = _sigmoid((cfg.ttr_center - f.ttr) / cfg.ttr_scale)
    if f.word_count >= cfg.min_words_for_repetition:
        out["repetition"] = _sigmoid((f.repetition_ratio - cfg.repetition_center) / cfg.repetition_scale)
    return out


def linguistic_score(subs: Dict[str, Optional[float]], cfg: FusionConfig) -> Optional[float]:
    w = {"burstiness": cfg.burstiness_weight, "ttr": cfg.ttr_weight, "repetition": cfg.repetition_weight}
    used = {k: w[k] for k, v in subs.items() if v is not None}
    total = sum(used.values())
    if total <= 0:
        return None
    return sum(used[k] * subs[k] for k in used) / total  # type: ignore[operator]


def component_scores(raw: RawSignals, cfg: FusionConfig) -> Dict[str, Optional[float]]:
    return {
        "roberta": raw.roberta_ai,
        "perplexity": perplexity_ai_score(raw.perplexity, cfg),
        "linguistic": linguistic_score(linguistic_subscores(raw.features, cfg), cfg),
        "feature_model": raw.feature_model_ai,
    }


def classify(score: float, cfg: FusionConfig) -> str:
    if score >= cfg.ai_threshold:
        return LABEL_AI
    if score <= cfg.human_threshold:
        return LABEL_HUMAN
    return LABEL_UNCERTAIN


def fuse(
    raw: RawSignals,
    cfg: Optional[FusionConfig] = None,
    calibrator: Optional[Callable[[float], float]] = None,
) -> FusionResult:
    cfg = cfg or FusionConfig()
    scores = component_scores(raw, cfg)
    base_w = cfg.weights()
    used = {k: base_w[k] for k, v in scores.items() if v is not None and base_w[k] > 0}
    total = sum(used.values())
    f = raw.features
    signals: Dict[str, Optional[float]] = {
        "roberta": raw.roberta_ai,
        "perplexity": raw.perplexity,
        "burstiness": f.burstiness if f and f.sentence_count >= cfg.min_sentences_for_burstiness else None,
        "ttr": f.ttr if f and f.word_count >= cfg.min_words_for_ttr else None,
        "repetition": f.repetition_ratio if f and f.word_count >= cfg.min_words_for_repetition else None,
        "feature_model": raw.feature_model_ai,
    }
    if total <= 0:
        return FusionResult(LABEL_UNCERTAIN, 0.5, 0.5, 0.5, signals, scores, {}, {}, ["no usable signals; defaulting to Uncertain"])

    weights = {k: w / total for k, w in used.items()}
    score = sum(weights[k] * scores[k] for k in weights)  # type: ignore[operator]
    score = min(1.0, max(0.0, score))
    contributions = {k: weights[k] * (scores[k] - 0.5) for k in weights}  # type: ignore[operator]
    p_ai = min(1.0, max(0.0, calibrator(score))) if calibrator else score
    notes = []
    if calibrator is None:
        notes.append("ai_probability equals the fused detection score; no calibrator is loaded, so it is not a calibrated probability")
    return FusionResult(classify(p_ai, cfg), p_ai, 1.0 - p_ai, score, signals, scores, weights, contributions, notes)


def fit_fusion_weights(
    component_matrix: np.ndarray, labels: Sequence[int], base: Optional[FusionConfig] = None
) -> FusionConfig:
    """Learn top-level weights from VALIDATION data (never the test set).

    component_matrix: shape (n, 4), columns = COMPONENTS, NaN where a signal was
    unavailable (filled with the neutral value 0.5). Weights are the
    non-negative parts of a logistic-regression fit, renormalised to sum to 1.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.nan_to_num(np.asarray(component_matrix, float), nan=0.5)
    y = np.asarray(labels, int)
    if X.shape[1] != len(COMPONENTS):
        raise ValueError(f"expected {len(COMPONENTS)} columns: {COMPONENTS}")
    if len(set(y.tolist())) < 2:
        raise ValueError("need both classes to fit fusion weights")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    w = np.clip(clf.coef_[0], 0.0, None)
    cfg = FusionConfig(**asdict(base)) if base else FusionConfig()
    if w.sum() <= 0:
        return cfg
    w = w / w.sum()
    cfg.roberta_weight, cfg.perplexity_weight, cfg.linguistic_weight, cfg.feature_model_weight = (float(v) for v in w)
    return cfg
