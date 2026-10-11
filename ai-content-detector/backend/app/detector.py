"""Transformer sequence classifiers.

RobertaBaseline wraps ``openai-community/roberta-base-openai-detector``.
THIS IS A BASELINE, NOT A RESEARCH CONTRIBUTION. It was trained mainly on
GPT-2 output and is known to be unreliable on modern LLM text. Treat its score
as one input to the fusion model and as a point of comparison in experiments.

FineTunedClassifier loads a local checkpoint produced by training/train.py
(e.g. DeBERTa-v3) and shares the same inference code.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .features import preprocess_text

MODEL_ROLE_BASELINE = "pretrained-baseline"
MODEL_ROLE_FINETUNED = "fine-tuned-research-model"

_AI_HINTS = ("fake", "ai", "generated", "machine", "synthetic", "chatgpt", "gpt", "llm")
_HUMAN_HINTS = ("real", "human", "original", "authentic")


class LabelMappingError(RuntimeError):
    """Raised when the model's labels cannot be mapped to AI/human safely."""


def resolve_label_mapping(
    id2label: Mapping[Any, str], ai_index_override: Optional[int] = None
) -> Tuple[int, int, bool, str]:
    """Work out which output index means "AI" and which means "human".

    Returns (ai_index, human_index, verified_from_names, note). Labels are
    inspected rather than assumed. If they are generic (LABEL_0/LABEL_1) and no
    override is supplied we refuse to guess.
    """
    labels = {int(k): str(v) for k, v in id2label.items()}
    if len(labels) != 2:
        raise LabelMappingError(f"expected a binary classifier, got labels {labels}")
    idx = sorted(labels)
    if ai_index_override is not None:
        if ai_index_override not in labels:
            raise LabelMappingError(f"ai_index_override={ai_index_override} not in {labels}")
        other = [i for i in idx if i != ai_index_override][0]
        return ai_index_override, other, False, f"AI index set manually; labels were {labels}"

    def has(label: str, hints: Tuple[str, ...]) -> bool:
        lab = label.lower()
        return any(lab == h or lab.startswith(h) or h in lab.replace("_", " ").split() for h in hints)

    ai = [i for i in idx if has(labels[i], _AI_HINTS) and not has(labels[i], _HUMAN_HINTS)]
    hu = [i for i in idx if has(labels[i], _HUMAN_HINTS) and not has(labels[i], _AI_HINTS)]
    if len(ai) == 1 and len(hu) == 1 and ai[0] != hu[0]:
        return ai[0], hu[0], True, f"mapped from label names {labels}"
    raise LabelMappingError(
        f"cannot map labels {labels} to AI/human from their names. "
        "Set ROBERTA_AI_LABEL_INDEX after checking the model card."
    )


@dataclass
class ClassifierOutput:
    ai_probability: float
    human_probability: float
    n_chunks: int
    truncated: bool
    model_name: str
    role: str
    label_note: str


class _TransformerClassifier:
    role = "classifier"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_chunks: int = 8,
        ai_index_override: Optional[int] = None,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device_pref = device
        self.max_chunks = max_chunks
        self.ai_index_override = ai_index_override
        self.batch_size = batch_size
        self._model: Any = None
        self._tok: Any = None
        self._device: Any = None
        self.ai_index = 1
        self.human_index = 0
        self.label_note = ""
        self.id2label: Dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device_pref == "auto" else self.device_pref
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.eval()
            model.to(device)
            ai, hu, _verified, note = resolve_label_mapping(model.config.id2label, self.ai_index_override)
            self.ai_index, self.human_index, self.label_note = ai, hu, note
            self.id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            self._tok, self._model, self._device = tok, model, device

    def _chunk_ids(self, ids: List[int]) -> List[List[int]]:
        body = int(self._tok.model_max_length)
        body = min(body, 512) - 2  # leave room for <s> ... </s>
        return [ids[i : i + body] for i in range(0, len(ids), body)] or [[]]

    def predict(self, text: object) -> ClassifierOutput:
        clean = preprocess_text(text)
        if not clean:
            raise ValueError("empty text")
        self._ensure_loaded()
        import torch

        ids = self._tok(clean, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        chunks = self._chunk_ids(ids)
        truncated = len(chunks) > self.max_chunks
        chunks = chunks[: self.max_chunks]
        weights = [len(c) for c in chunks]
        probs_ai: List[float] = []
        with torch.no_grad():
            for b in range(0, len(chunks), self.batch_size):
                batch = [self._tok.build_inputs_with_special_tokens(c) for c in chunks[b : b + self.batch_size]]
                width = max(len(x) for x in batch)
                pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
                input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in batch], device=self._device)
                mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in batch], device=self._device)
                logits = self._model(input_ids=input_ids, attention_mask=mask).logits
                p = torch.softmax(logits, dim=-1)[:, self.ai_index]
                probs_ai.extend(p.tolist())
        total = float(sum(weights)) or 1.0
        ai = sum(p * w for p, w in zip(probs_ai, weights)) / total  # token-weighted mean
        ai = min(1.0, max(0.0, ai))
        return ClassifierOutput(ai, 1.0 - ai, len(chunks), truncated, self.model_name, self.role, self.label_note)

    def predict_proba_many(self, texts: Sequence[str]) -> List[float]:
        """AI probability for each text (used by experiments)."""
        return [self.predict(t).ai_probability for t in texts]


class RobertaBaseline(_TransformerClassifier):
    """PRETRAINED BASELINE (GPT-2-era detector). Not our research model."""

    role = MODEL_ROLE_BASELINE

    def __init__(self, model_name: str = "openai-community/roberta-base-openai-detector", **kw: Any) -> None:
        super().__init__(model_name, **kw)


class FineTunedClassifier(_TransformerClassifier):
    """Local checkpoint from training/train.py (id2label = {0: human, 1: ai})."""

    role = MODEL_ROLE_FINETUNED
