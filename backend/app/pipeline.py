"""Orchestrates preprocessing -> signals -> fusion -> trust -> explanation.

Optional components (RoBERTa, perplexity model, TF-IDF model) may be disabled
or fail to load (e.g. no internet for the first download). The pipeline then
continues with the signals that are available, records a warning, and the
fusion renormalises its weights. It never invents a value for a missing signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import Settings
from .detector import MODEL_ROLE_BASELINE, RobertaBaseline
from .explainability import (
    Explainer, SentenceOcclusionExplainer, TfidfTermExplainer, explain_signals, integrated_gradients_attributions,
)
from .feature_model import FeatureTextClassifier
from .features import TextFeatures, extract_features, preprocess_text
from .perplexity import PerplexityScorer
from .scoring import FusionConfig, FusionResult, RawSignals, fuse
from .trust_score import (
    CalibrationReport, LogitCalibrator, RobustnessReport, TrustConfig, compute_trust,
)

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Automated AI-text detection is probabilistic and error-prone. Results are signals, not proof of "
    "authorship; do not use them as sole evidence for academic or disciplinary decisions."
)


@dataclass
class DetectionResult:
    prediction: str
    ai_probability: float
    human_probability: float
    detection_score: float
    signals: Dict[str, Optional[float]]
    signal_scores: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    trust: Dict[str, Any]
    explanation: List[Dict[str, Any]]
    warnings: List[str]
    features: Dict[str, float]
    model_version: str
    components_active: List[str]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class TextDetectionPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        roberta: Optional[RobertaBaseline] = None,
        perplexity: Optional[PerplexityScorer] = None,
        feature_model: Optional[FeatureTextClassifier] = None,
        fusion_cfg: Optional[FusionConfig] = None,
        calibrator: Optional[LogitCalibrator] = None,
        calibration_report: Optional[CalibrationReport] = None,
        robustness_report: Optional[RobustnessReport] = None,
        trust_cfg: Optional[TrustConfig] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.roberta, self.perplexity, self.feature_model = roberta, perplexity, feature_model
        self.fusion_cfg = fusion_cfg or FusionConfig()
        self.calibrator = calibrator
        self.calibration_report, self.robustness_report = calibration_report, robustness_report
        self.trust_cfg = trust_cfg or TrustConfig(reliable_words=self.settings.reliable_words)
        self._failed: Dict[str, str] = {}  # component -> error, so we do not retry every request

    # ------------------------------------------------------------ build
    @classmethod
    def from_settings(cls, s: Settings) -> "TextDetectionPipeline":
        roberta = RobertaBaseline(s.roberta_model_name, device=s.device, max_chunks=s.max_roberta_chunks,
                                  ai_index_override=s.roberta_ai_label_index) if s.enable_roberta else None
        ppl = PerplexityScorer(s.perplexity_model_name, device=s.device, max_tokens=s.max_perplexity_tokens) if s.enable_perplexity else None
        fm = None
        if s.feature_model_path.exists():
            try:
                fm = FeatureTextClassifier.load(s.feature_model_path)
            except Exception as e:  # corrupt / incompatible artifact
                log.warning("could not load feature model: %s", e)
        fcfg = FusionConfig.load(s.fusion_config_path) if s.fusion_config_path.exists() else FusionConfig()
        cal = LogitCalibrator.load(s.calibration_path) if s.calibration_path.exists() else None
        cal_rep = CalibrationReport.load(s.calibration_path) if _is_report(s.calibration_path, "ece") else None
        rob = RobustnessReport.load(s.robustness_report_path) if s.robustness_report_path.exists() else None
        return cls(s, roberta=roberta, perplexity=ppl, feature_model=fm, fusion_cfg=fcfg,
                   calibrator=cal if cal_rep is None else LogitCalibrator(cal_rep.temperature),
                   calibration_report=cal_rep, robustness_report=rob)

    def preload(self) -> None:
        for name, comp in (("roberta", self.roberta), ("perplexity", self.perplexity)):
            if comp is not None:
                try:
                    comp._ensure_loaded()
                except Exception as e:
                    self._failed[name] = str(e)
                    log.warning("preload of %s failed: %s", name, e)

    def status(self) -> Dict[str, Any]:
        def st(name: str, comp: Any) -> str:
            if comp is None:
                return "disabled"
            if name in self._failed:
                return "failed"
            return "loaded" if getattr(comp, "is_loaded", True) else "not_loaded_yet"
        return {
            "roberta_baseline": st("roberta", self.roberta),
            "perplexity": st("perplexity", self.perplexity),
            "feature_model": "loaded" if self.feature_model else "absent (train it to enable)",
            "calibrator": "loaded" if self.calibrator else "absent",
            "robustness_report": "loaded" if self.robustness_report else "absent",
            "errors": dict(self._failed),
        }

    # ---------------------------------------------------------- signals
    def _safe(self, name: str, fn: Callable[[], Any], warnings: List[str]) -> Any:
        if name in self._failed:
            warnings.append(f"{name} unavailable: {self._failed[name]}")
            return None
        try:
            return fn()
        except Exception as e:
            self._failed[name] = f"{type(e).__name__}: {e}"
            log.exception("%s failed", name)
            warnings.append(f"{name} unavailable: {self._failed[name]}")
            return None

    def compute_signals(self, text: str, warnings: Optional[List[str]] = None) -> tuple[RawSignals, Dict[str, Any]]:
        warnings = warnings if warnings is not None else []
        clean = preprocess_text(text)
        feats = extract_features(clean)
        meta: Dict[str, Any] = {"truncated": False}
        raw = RawSignals(features=feats)
        if self.roberta is not None:
            out = self._safe("roberta", lambda: self.roberta.predict(clean), warnings)  # type: ignore[union-attr]
            if out is not None:
                raw.roberta_ai = out.ai_probability
                meta["truncated"] = meta["truncated"] or out.truncated
        if self.perplexity is not None:
            res = self._safe("perplexity", lambda: self.perplexity.compute(clean), warnings)  # type: ignore[union-attr]
            if res is not None:
                raw.perplexity = res.perplexity
                meta["truncated"] = meta["truncated"] or res.truncated
                if res.note:
                    warnings.append(f"perplexity not computed: {res.note}")
        if self.feature_model is not None:
            raw.feature_model_ai = self._safe("feature_model", lambda: self.feature_model.predict_one(clean), warnings)  # type: ignore[union-attr]
        return raw, meta

    # ---------------------------------------------------------- analyse
    def analyze(self, text: str) -> DetectionResult:
        warnings: List[str] = []
        raw, meta = self.compute_signals(text, warnings)
        feats = raw.features or TextFeatures()
        fused: FusionResult = fuse(raw, self.fusion_cfg, self.calibrator)
        if feats.word_count < self.settings.reliable_words:
            warnings.append(f"Text has {feats.word_count} words; fewer than {self.settings.reliable_words} makes every signal unreliable.")
        if meta["truncated"]:
            warnings.append("Input was longer than the configured analysis window; only the first part was scored by some models.")
        warnings.extend(fused.notes)
        trust = compute_trust(fused.ai_probability, fused.signal_scores, feats, truncated=meta["truncated"],
                              calibration=self.calibration_report, robustness=self.robustness_report, cfg=self.trust_cfg)
        explanation = explain_signals(fused, feats, self.fusion_cfg,
                                      perplexity_model=self.settings.perplexity_model_name,
                                      warnings=[w for w in warnings if "unavailable" in w or "words" in w])
        active = [k for k, v in fused.signal_scores.items() if v is not None]
        return DetectionResult(
            prediction=fused.prediction, ai_probability=fused.ai_probability, human_probability=fused.human_probability,
            detection_score=fused.detection_score, signals=fused.signals, signal_scores=fused.signal_scores,
            weights_used=fused.weights_used, trust=trust.to_dict(), explanation=explanation, warnings=warnings,
            features=feats.to_dict(), model_version=self.settings.model_version, components_active=active,
        )

    def score(self, text: str) -> float:
        """Fused detection score only (used by occlusion / robustness)."""
        raw, _ = self.compute_signals(text)
        return fuse(raw, self.fusion_cfg, self.calibrator).ai_probability

    # ---------------------------------------------------------- explain
    def explain(self, text: str, method: str = "signals", **kw: Any) -> Dict[str, Any]:
        if method == "signals":
            return {"method": method, "items": self.analyze(text).explanation}
        if method == "tfidf_terms":
            if self.feature_model is None:
                raise RuntimeError("TF-IDF model is not available; train it first (see README)")
            return {"method": method, "items": self.feature_model.term_contributions(text, top_k=kw.get("top_k", 8))}
        if method == "occlusion":
            return {"method": method, "items": SentenceOcclusionExplainer(self.score, kw.get("max_sentences", 15)).explain(text)}
        if method == "integrated_gradients":
            if self.roberta is None:
                raise RuntimeError("RoBERTa baseline is disabled")
            return {"method": method, "items": integrated_gradients_attributions(self.roberta, preprocess_text(text))}
        raise ValueError(f"unknown explanation method: {method}")


def _is_report(path: Any, key: str) -> bool:
    try:
        import json
        return path.exists() and key in json.loads(path.read_text())
    except Exception:
        return False
