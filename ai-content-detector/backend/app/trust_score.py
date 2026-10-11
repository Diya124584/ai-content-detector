"""Trust Score = how much to rely on a prediction. It is NOT the AI probability.

A text can have AI probability 0.95 and a low Trust Score (e.g. 12 words long,
signals disagree) or probability 0.50 and a high Trust Score (long text, all
signals agree it is ambiguous). The two numbers answer different questions.

Components (each in [0, 1], missing ones are skipped and listed):
  model_confidence  - distance of the final probability from 0.5
  signal_agreement  - do the independent signals point the same way?
  input_reliability - out-of-distribution indicator: length, language/prose
                      heuristics, truncation
  calibration       - from a CalibrationReport (ECE on held-out data); absent
                      until you run a calibration experiment
  robustness        - from a RobustnessReport (prediction stability under
                      perturbation); absent until you run robustness tests

Until BOTH calibration and robustness artifacts exist the score is labelled
"prototype / not yet calibrated". It is a heuristic, not a validated quantity.

This module also provides the calibration tooling: temperature scaling, Brier
score, Expected Calibration Error and reliability curves.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import TextFeatures

PROTOTYPE_STATUS = "prototype / not yet calibrated"
PARTIAL_STATUS = "partially informed prototype (one of calibration/robustness artifacts missing)"
INFORMED_STATUS = "informed by calibration + robustness artifacts (re-validate on your own data)"
TRUST_DISCLAIMER = (
    "Trust Score estimates how reliable this particular prediction is. It is not the probability "
    "that the text is AI-generated and it has not been validated as a statistical quantity."
)

_EPS = 1e-6


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ------------------------------------------------------------ calibration
@dataclass
class LogitCalibrator:
    """p' = sigmoid(logit(p) / temperature + bias). Identity by default."""

    temperature: float = 1.0
    bias: float = 0.0

    def __call__(self, p: float) -> float:
        return float(_sigmoid(_logit(p) / self.temperature + self.bias))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "LogitCalibrator":
        d = json.loads(Path(path).read_text())
        return cls(float(d.get("temperature", 1.0)), float(d.get("bias", 0.0)))


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


def reliability_curve(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> List[Dict[str, float]]:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if mask.any():
            out.append({"bin_low": float(lo), "bin_high": float(hi), "count": int(mask.sum()),
                        "mean_confidence": float(p[mask].mean()), "empirical_accuracy": float(y[mask].mean())})
    return out


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """ECE over P(AI) bins: sum_b (n_b/N) * |mean(p_b) - frac_positive_b|."""
    n = len(probs)
    if n == 0:
        return float("nan")
    return float(sum(b["count"] / n * abs(b["mean_confidence"] - b["empirical_accuracy"]) for b in reliability_curve(probs, labels, n_bins)))


def fit_temperature(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Grid-search temperature minimising log-loss on a *validation* set."""
    z, y = _logit(probs), np.asarray(labels, float)
    best_t, best_nll = 1.0, math.inf
    for t in np.logspace(-1, 1.5, 250):
        p = np.clip(_sigmoid(z / t), _EPS, 1 - _EPS)
        nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


@dataclass
class CalibrationReport:
    n_samples: int
    ece: float
    brier: float
    temperature: float
    dataset: str
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    @property
    def quality(self) -> float:
        return float(max(0.0, 1.0 - self.ece))  # simple, documented mapping: ECE 0 -> 1.0

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationReport":
        return cls(**json.loads(Path(path).read_text()))


def build_calibration_report(val_probs: Sequence[float], val_labels: Sequence[int], dataset: str) -> Tuple[CalibrationReport, LogitCalibrator]:
    """Fit temperature on validation data and report calibrated ECE/Brier.
    Evaluate again on a separate test split before trusting the numbers."""
    t = fit_temperature(val_probs, val_labels)
    cal = LogitCalibrator(temperature=t)
    adj = [cal(p) for p in val_probs]
    rep = CalibrationReport(len(adj), expected_calibration_error(adj, val_labels), brier_score(adj, val_labels), t, dataset)
    return rep, cal


@dataclass
class RobustnessReport:
    n_pairs: int
    prediction_stability: float  # fraction of pairs whose label did not change
    mean_abs_probability_change: float
    dataset: str
    perturbation_types: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    @property
    def quality(self) -> float:
        return float(min(1.0, max(0.0, self.prediction_stability)))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "RobustnessReport":
        return cls(**json.loads(Path(path).read_text()))


# ------------------------------------------------------------ trust score
@dataclass
class TrustConfig:
    # Prototype weights, not fitted to data.
    model_confidence_weight: float = 0.35
    signal_agreement_weight: float = 0.25
    input_reliability_weight: float = 0.20
    calibration_weight: float = 0.10
    robustness_weight: float = 0.10
    reliable_words: int = 50  # full reliability at/above this length
    min_stopword_ratio: float = 0.10  # ordinary English prose is well above this; lower hints at non-English / non-prose
    min_sentences: int = 3


@dataclass
class TrustResult:
    value: float  # 0-100
    status: str
    components: Dict[str, Optional[float]]
    missing_components: List[str]
    ood_flags: List[str]
    disclaimer: str = TRUST_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_trust(
    ai_probability: float,
    component_scores: Dict[str, Optional[float]],
    features: TextFeatures,
    *,
    truncated: bool = False,
    calibration: Optional[CalibrationReport] = None,
    robustness: Optional[RobustnessReport] = None,
    cfg: Optional[TrustConfig] = None,
) -> TrustResult:
    cfg = cfg or TrustConfig()
    flags: List[str] = []

    confidence = abs(2.0 * ai_probability - 1.0)

    avail = [v for v in component_scores.values() if v is not None]
    agreement: Optional[float] = None
    if len(avail) >= 2:
        agreement = float(max(0.0, 1.0 - 2.0 * float(np.std(avail))))  # std of values in [0,1] is <= 0.5

    length_factor = min(1.0, features.word_count / cfg.reliable_words) if cfg.reliable_words else 1.0
    if features.word_count < cfg.reliable_words:
        flags.append(f"short_text: {features.word_count} words (< {cfg.reliable_words})")
    language_factor = 1.0
    if features.word_count >= 5 and features.stopword_ratio < cfg.min_stopword_ratio:
        language_factor = features.stopword_ratio / cfg.min_stopword_ratio
        flags.append("low_stopword_ratio: text may be non-English or not ordinary prose")
    if features.sentence_count < cfg.min_sentences:
        flags.append(f"few_sentences: {features.sentence_count}")
    if truncated:
        flags.append("input_truncated: only part of the text was analysed")
    reliability = min(length_factor, language_factor) * (0.8 if truncated else 1.0)

    comps: Dict[str, Optional[float]] = {
        "model_confidence": confidence,
        "signal_agreement": agreement,
        "input_reliability": reliability,
        "calibration": calibration.quality if calibration else None,
        "robustness": robustness.quality if robustness else None,
    }
    weights = {
        "model_confidence": cfg.model_confidence_weight,
        "signal_agreement": cfg.signal_agreement_weight,
        "input_reliability": cfg.input_reliability_weight,
        "calibration": cfg.calibration_weight,
        "robustness": cfg.robustness_weight,
    }
    used = {k: w for k, w in weights.items() if comps[k] is not None}
    total = sum(used.values())
    value = 100.0 * sum(w * comps[k] for k, w in used.items()) / total if total > 0 else 0.0  # type: ignore[operator]
    missing = [k for k, v in comps.items() if v is None]

    if calibration and robustness:
        status = INFORMED_STATUS
    elif calibration or robustness:
        status = PARTIAL_STATUS
    else:
        status = PROTOTYPE_STATUS
    return TrustResult(round(value, 1), status, comps, missing, flags)
