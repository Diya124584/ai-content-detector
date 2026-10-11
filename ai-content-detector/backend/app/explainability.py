"""Explainability layer.

Design rule: every explanation is derived from the same numbers the fusion
model used, so it cannot contradict the prediction.

Methods
  signals     (default, cheap) - per-signal contribution to the fused score.
  tfidf_terms  - exact additive term contributions of the TF-IDF + LR model.
  occlusion    - leave-one-sentence-out change in the detection score
                 (model-agnostic; costs one full detection per sentence).
  integrated_gradients - token attributions for a transformer classifier via
                 Captum. OPTIONAL: needs `captum` and a loaded model, and is
                 slow on CPU, so it is never run inside /detect.
SHAP can be plugged in by implementing the ``Explainer`` interface; for the
linear TF-IDF model the exact contributions above already coincide with linear
SHAP values under feature independence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .features import TextFeatures, split_sentences
from .scoring import FusionConfig, FusionResult, linguistic_subscores

TOWARD_AI, TOWARD_HUMAN, NEUTRAL, CAVEAT = "toward_ai", "toward_human", "neutral", "caveat"


@dataclass
class ExplainConfig:
    neutral_band: float = 0.10  # |score - 0.5| below this is reported as inconclusive
    strong: float = 0.80  # score >= this (or <= 1 - this) is described as "strongly"


class Explainer(ABC):
    """Interface so SHAP / IG / other methods can be added without API changes."""

    name: str = "base"

    @abstractmethod
    def explain(self, text: str, **kwargs: Any) -> List[Dict[str, Any]]: ...


def _direction(score: float, cfg: ExplainConfig) -> str:
    if score >= 0.5 + cfg.neutral_band:
        return TOWARD_AI
    if score <= 0.5 - cfg.neutral_band:
        return TOWARD_HUMAN
    return NEUTRAL


def _item(factor: str, score: float, contribution: float, detail: str, cfg: ExplainConfig) -> Dict[str, Any]:
    return {
        "factor": factor,
        "direction": _direction(score, cfg),
        "strength": round(abs(score - 0.5) * 2, 3),
        "contribution": round(contribution, 4),
        "detail": detail,
    }


def explain_signals(
    result: FusionResult,
    features: Optional[TextFeatures],
    fusion_cfg: FusionConfig,
    *,
    perplexity_model: str = "gpt2",
    warnings: Optional[List[str]] = None,
    cfg: Optional[ExplainConfig] = None,
) -> List[Dict[str, Any]]:
    """Feature-level explanation of a fusion result (sorted by |contribution|)."""
    cfg = cfg or ExplainConfig()
    items: List[Dict[str, Any]] = []
    sc, contrib = result.signal_scores, result.contributions

    s = sc.get("roberta")
    if s is not None:
        if s >= cfg.strong:
            d = "strongly favors AI"
        elif s <= 1 - cfg.strong:
            d = "strongly favors human"
        else:
            d = {TOWARD_AI: "leans AI", TOWARD_HUMAN: "leans human", NEUTRAL: "is inconclusive"}[_direction(s, cfg)]
        items.append(_item("roberta_baseline", s, contrib.get("roberta", 0.0),
                           f"RoBERTa baseline {d} (P(AI)={s:.2f}). It is a GPT-2-era detector and may be unreliable on modern LLM text.", cfg))

    s = sc.get("perplexity")
    ppl = result.signals.get("perplexity")
    if s is not None and ppl is not None:
        rel = {TOWARD_AI: "relatively low (more predictable, nudges toward AI)",
               TOWARD_HUMAN: "relatively high (less predictable, nudges toward human)",
               NEUTRAL: "near the prototype pivot (inconclusive)"}[_direction(s, cfg)]
        items.append(_item("perplexity", s, contrib.get("perplexity", 0.0),
                           f"Perplexity under {perplexity_model} is {ppl:.1f}, {rel}. Perplexity is only a weak signal.", cfg))

    if "feature_model" in contrib or sc.get("feature_model") is not None:
        s = sc["feature_model"]
        if s is not None:
            items.append(_item("tfidf_model", s, contrib.get("feature_model", 0.0),
                               f"TF-IDF + logistic-regression model gives P(AI)={s:.2f}. Use /explain with method=tfidf_terms for the words behind it.", cfg))

    subs = linguistic_subscores(features, fusion_cfg)
    lw = result.weights_used.get("linguistic", 0.0)
    sub_w = {"burstiness": fusion_cfg.burstiness_weight, "ttr": fusion_cfg.ttr_weight, "repetition": fusion_cfg.repetition_weight}
    tot = sum(sub_w[k] for k, v in subs.items() if v is not None) or 1.0
    if features is not None:
        texts = {
            "burstiness": {TOWARD_AI: f"Sentence lengths are unusually uniform (burstiness {features.burstiness:.2f})",
                           TOWARD_HUMAN: f"Sentence lengths vary a lot (burstiness {features.burstiness:.2f})",
                           NEUTRAL: f"Sentence-length variation is unremarkable (burstiness {features.burstiness:.2f})"},
            "ttr": {TOWARD_AI: f"Low lexical diversity (TTR {features.ttr:.2f}; length-dependent)",
                    TOWARD_HUMAN: f"High lexical diversity (TTR {features.ttr:.2f}; length-dependent)",
                    NEUTRAL: f"Lexical diversity is unremarkable (TTR {features.ttr:.2f})"},
            "repetition": {TOWARD_AI: f"High repetition: {features.repetition_ratio:.0%} of 3-word sequences repeat",
                           TOWARD_HUMAN: f"Little repetition ({features.repetition_ratio:.0%} repeated 3-word sequences)",
                           NEUTRAL: f"Repetition is unremarkable ({features.repetition_ratio:.0%})"},
        }
        for k, v in subs.items():
            if v is None:
                continue
            c = lw * (sub_w[k] / tot) * (v - 0.5)
            items.append(_item(k, v, c, texts[k][_direction(v, cfg)] + ". Signal only, not proof.", cfg))

    items.sort(key=lambda i: abs(i["contribution"]), reverse=True)
    for w in warnings or []:
        items.append({"factor": "warning", "direction": CAVEAT, "strength": 0.0, "contribution": 0.0, "detail": w})
    return items


class SignalExplainer(Explainer):
    name = "signals"

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def explain(self, text: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.pipeline.analyze(text).explanation


class TfidfTermExplainer(Explainer):
    name = "tfidf_terms"

    def __init__(self, model: Any) -> None:
        self.model = model

    def explain(self, text: str, top_k: int = 8, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.model.term_contributions(text, top_k=top_k)


class SentenceOcclusionExplainer(Explainer):
    """delta > 0 means removing the sentence LOWERS the AI score, i.e. the
    sentence pushed the score toward AI."""

    name = "occlusion"

    def __init__(self, score_fn: Callable[[str], float], max_sentences: int = 15) -> None:
        self.score_fn = score_fn
        self.max_sentences = max_sentences

    def explain(self, text: str, **kwargs: Any) -> List[Dict[str, Any]]:
        sents = split_sentences(text)
        if len(sents) < 2:
            return []
        base = self.score_fn(text)
        out = []
        for i, s in enumerate(sents[: self.max_sentences]):
            rest = " ".join(sents[:i] + sents[i + 1:])
            delta = base - self.score_fn(rest)
            out.append({"sentence": s, "delta": round(delta, 4),
                        "direction": TOWARD_AI if delta > 0 else TOWARD_HUMAN})
        out.sort(key=lambda d: abs(d["delta"]), reverse=True)
        return out


def integrated_gradients_attributions(classifier: Any, text: str, n_steps: int = 20, max_tokens: int = 256) -> List[Dict[str, Any]]:
    """Token attributions toward the AI class using Captum LayerIntegratedGradients.

    Optional and slow. Requires `pip install captum` and a transformer
    classifier (RobertaBaseline / FineTunedClassifier) with a RoBERTa-style
    ``model.roberta.embeddings`` layer. Not exercised by the automated tests.
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("captum is not installed (pip install captum)") from e
    import torch

    classifier._ensure_loaded()
    tok, model = classifier._tok, classifier._model
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(classifier._device)
    input_ids, mask = enc["input_ids"], enc["attention_mask"]
    ref = torch.full_like(input_ids, tok.pad_token_id)
    ref[:, 0], ref[:, -1] = input_ids[:, 0], input_ids[:, -1]  # keep special tokens

    def forward(ids: Any, attn: Any) -> Any:
        return torch.softmax(model(input_ids=ids, attention_mask=attn).logits, dim=-1)[:, classifier.ai_index]

    with torch.enable_grad():
        lig = LayerIntegratedGradients(forward, model.roberta.embeddings)
        attrs = lig.attribute(inputs=input_ids, baselines=ref, additional_forward_args=(mask,), n_steps=n_steps)
    per_token = attrs.sum(dim=-1).squeeze(0)
    per_token = per_token / (per_token.norm() + 1e-9)
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    return [{"token": t, "attribution": float(a)} for t, a in zip(tokens, per_token.tolist())]
