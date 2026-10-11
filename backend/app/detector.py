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


_AI_HINTS = (
    "fake",
    "ai",
    "generated",
    "machine",
    "synthetic",
    "chatgpt",
    "gpt",
    "llm",
)

_HUMAN_HINTS = (
    "real",
    "human",
    "original",
    "authentic",
)


class LabelMappingError(RuntimeError):
    """Raised when the model's labels cannot be mapped to AI/human safely."""


def resolve_label_mapping(
    id2label: Mapping[Any, str],
    ai_index_override: Optional[int] = None,
) -> Tuple[int, int, bool, str]:
    """Work out which output index means AI and which means human.

    Returns:
        (ai_index, human_index, verified_from_names, note)

    Labels are inspected rather than blindly assumed. If they are generic
    (for example LABEL_0/LABEL_1) and no override is supplied, we refuse
    to guess.
    """

    labels = {
        int(k): str(v)
        for k, v in id2label.items()
    }

    if len(labels) != 2:
        raise LabelMappingError(
            f"expected a binary classifier, got labels {labels}"
        )

    indices = sorted(labels)

    if ai_index_override is not None:
        if ai_index_override not in labels:
            raise LabelMappingError(
                f"ai_index_override={ai_index_override} not in {labels}"
            )

        other_indices = [
            index
            for index in indices
            if index != ai_index_override
        ]

        human_index = other_indices[0]

        return (
            ai_index_override,
            human_index,
            False,
            f"AI index set manually; labels were {labels}",
        )

    def has_hint(
        label: str,
        hints: Tuple[str, ...],
    ) -> bool:
        label_lower = label.lower()

        return any(
            label_lower == hint
            or label_lower.startswith(hint)
            or hint in label_lower.replace("_", " ").split()
            for hint in hints
        )

    ai_candidates = [
        index
        for index in indices
        if has_hint(labels[index], _AI_HINTS)
        and not has_hint(labels[index], _HUMAN_HINTS)
    ]

    human_candidates = [
        index
        for index in indices
        if has_hint(labels[index], _HUMAN_HINTS)
        and not has_hint(labels[index], _AI_HINTS)
    ]

    if (
        len(ai_candidates) == 1
        and len(human_candidates) == 1
        and ai_candidates[0] != human_candidates[0]
    ):
        return (
            ai_candidates[0],
            human_candidates[0],
            True,
            f"mapped from label names {labels}",
        )

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
        """Load tokenizer and model once."""

        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            if self.device_pref == "auto":
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )
            else:
                device = self.device_pref

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            model.eval()
            model.to(device)

            (
                ai_index,
                human_index,
                _verified,
                note,
            ) = resolve_label_mapping(
                model.config.id2label,
                self.ai_index_override,
            )

            self.ai_index = ai_index
            self.human_index = human_index
            self.label_note = note

            self.id2label = {
                int(k): str(v)
                for k, v in model.config.id2label.items()
            }

            self._tok = tokenizer
            self._model = model
            self._device = device

    def _get_model_max_length(self) -> int:
        """Return a safe model context length."""

        model_max_length = self._tok.model_max_length

        # Some Hugging Face tokenizers use a very large sentinel value
        # instead of an actual context length.
        if (
            model_max_length is None
            or model_max_length > 512
        ):
            model_max_length = 512

        return min(int(model_max_length), 512)

    def _chunk_ids(
        self,
        ids: List[int],
    ) -> List[List[int]]:
        """Split token IDs into model-sized chunks.

        Two positions are reserved for RoBERTa special tokens:
        <s> and </s>.
        """

        max_length = self._get_model_max_length()

        body_length = max(
            1,
            max_length - 2,
        )

        chunks = [
            ids[i : i + body_length]
            for i in range(
                0,
                len(ids),
                body_length,
            )
        ]

        return chunks or [[]]

    def predict(
        self,
        text: object,
    ) -> ClassifierOutput:
        """Predict AI probability for a single text."""

        clean = preprocess_text(text)

        if not clean:
            raise ValueError("empty text")

        self._ensure_loaded()

        import torch

        # First tokenize without special tokens.
        # This allows us to manually divide long documents into chunks.
        encoded = self._tok(
            clean,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        ids = encoded["input_ids"]

        chunks = self._chunk_ids(ids)

        truncated = len(chunks) > self.max_chunks

        chunks = chunks[: self.max_chunks]

        # Longer chunks should contribute proportionally more to the
        # final document-level probability.
        weights = [
            max(1, len(chunk))
            for chunk in chunks
        ]

        probabilities_ai: List[float] = []

        with torch.no_grad():

            for start in range(
                0,
                len(chunks),
                self.batch_size,
            ):
                chunk_batch = chunks[
                    start : start + self.batch_size
                ]

                # IMPORTANT:
                #
                # The old implementation directly called:
                #
                # self._tok.build_inputs_with_special_tokens(...)
                #
                # That is not available through the current tokenizer
                # interface used by this environment.
                #
                # Instead, decode each token chunk back to text and use
                # the public tokenizer API. The tokenizer then correctly
                # handles:
                #   - special tokens
                #   - padding
                #   - truncation
                #   - attention masks
                #   - PyTorch tensors
                chunk_texts = [
                    self._tok.decode(
                        chunk,
                        skip_special_tokens=True,
                    )
                    for chunk in chunk_batch
                ]

                batch = self._tok(
                    chunk_texts,
                    padding=True,
                    truncation=True,
                    max_length=self._get_model_max_length(),
                    return_tensors="pt",
                )

                # Move tensors to the same device as the model.
                batch = {
                    key: value.to(self._device)
                    for key, value in batch.items()
                    if key in (
                        "input_ids",
                        "attention_mask",
                    )
                }

                outputs = self._model(
                    **batch
                )

                logits = outputs.logits

                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                )[:, self.ai_index]

                probabilities_ai.extend(
                    probabilities.detach()
                    .cpu()
                    .tolist()
                )

        total_weight = float(
            sum(weights)
        ) or 1.0

        ai_probability = sum(
            probability * weight
            for probability, weight in zip(
                probabilities_ai,
                weights,
            )
        ) / total_weight

        ai_probability = min(
            1.0,
            max(
                0.0,
                float(ai_probability),
            ),
        )

        human_probability = 1.0 - ai_probability

        return ClassifierOutput(
            ai_probability=ai_probability,
            human_probability=human_probability,
            n_chunks=len(chunks),
            truncated=truncated,
            model_name=self.model_name,
            role=self.role,
            label_note=self.label_note,
        )

    def predict_proba_many(
        self,
        texts: Sequence[str],
    ) -> List[float]:
        """Return AI probability for each text."""

        return [
            self.predict(text).ai_probability
            for text in texts
        ]


class RobertaBaseline(_TransformerClassifier):
    """PRETRAINED BASELINE.

    This is the GPT-2-era pretrained detector and is NOT the research
    contribution of the project.
    """

    role = MODEL_ROLE_BASELINE

    def __init__(
        self,
        model_name: str = (
            "openai-community/"
            "roberta-base-openai-detector"
        ),
        **kw: Any,
    ) -> None:
        super().__init__(
            model_name,
            **kw,
        )


class FineTunedClassifier(_TransformerClassifier):
    """Fine-tuned research classifier.

    This loads a local checkpoint produced by training/train.py,
    such as a fine-tuned DeBERTa-v3 model.

    Expected binary mapping:
        0 = human
        1 = AI

    The actual mapping is still checked from model.config.id2label
    rather than blindly assumed.
    """

    role = MODEL_ROLE_FINETUNED

    def __init__(
        self,
        model_name: str,
        **kw: Any,
    ) -> None:
        super().__init__(
            model_name,
            **kw,
        )