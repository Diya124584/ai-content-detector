"""Perplexity under a causal language model (GPT-2 by default).

Perplexity is ONE weak signal. Low perplexity means the text is predictable to
*this* language model; it does not prove AI authorship (famous/boilerplate human
text is also low-perplexity, and strong LLM output can be high-perplexity).
It also depends on the scoring model, tokenizer and text length.

The model is loaded once (lazily, thread-safe) and reused for every request.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Optional

from .features import preprocess_text


@dataclass
class PerplexityResult:
    perplexity: Optional[float]  # None when it could not be computed
    n_tokens: int = 0
    truncated: bool = False
    note: Optional[str] = None
    model_name: str = ""


class PerplexityScorer:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_tokens: int = 2048,
        min_tokens: int = 10,
        stride_fraction: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.device_pref = device
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.stride_fraction = stride_fraction  # window overlap for long text
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device_pref == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_pref
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.eval()
            model.to(device)
            self._tokenizer, self._model, self._device = tok, model, device

    def _context_length(self) -> int:
        cfg = self._model.config
        return int(getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None) or 1024)

    def compute(self, text: object) -> PerplexityResult:
        clean = preprocess_text(text)
        if not clean:
            return PerplexityResult(None, 0, False, "empty text", self.model_name)
        self._ensure_loaded()
        import torch

        ids = self._tokenizer(clean, return_tensors="pt", truncation=False)["input_ids"][0]
        truncated = ids.numel() > self.max_tokens
        ids = ids[: self.max_tokens]
        seq_len = int(ids.numel())
        if seq_len < self.min_tokens:
            return PerplexityResult(None, seq_len, False, f"fewer than {self.min_tokens} tokens", self.model_name)

        ids = ids.unsqueeze(0).to(self._device)
        max_len = min(self._context_length(), seq_len)
        stride = max(1, int(max_len * self.stride_fraction))

        # Sliding window with overlap (the standard HF recipe) so texts longer
        # than the model context are scored without silently dropping the tail.
        nll_sum, n_scored, prev_end = 0.0, 0, 0
        with torch.no_grad():
            for begin in range(0, seq_len, stride):
                end = min(begin + max_len, seq_len)
                trg_len = end - prev_end
                window = ids[:, begin:end]
                target = window.clone()
                target[:, :-trg_len] = -100  # context only, not scored
                loss = self._model(window, labels=target).loss
                # HF shifts labels by one, so count scored positions after shift.
                scored = int((target[:, 1:] != -100).sum().item())
                if scored > 0:
                    nll_sum += float(loss.item()) * scored
                    n_scored += scored
                prev_end = end
                if end == seq_len:
                    break
        if n_scored == 0:
            return PerplexityResult(None, seq_len, truncated, "no scorable tokens", self.model_name)
        ppl = math.exp(min(nll_sum / n_scored, 50.0))  # clamp avoids overflow
        return PerplexityResult(ppl, seq_len, truncated, None, self.model_name)
