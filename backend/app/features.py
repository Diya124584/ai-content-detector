"""Linguistic / statistical features.

IMPORTANT: every feature here is only a *signal*. A low type-token ratio does
not prove a text is AI-written, and neither does low burstiness. Human writers
(especially in formulaic genres) can produce all of these patterns, and modern
LLMs can be prompted to avoid them. Features are consumed by the fusion scorer,
which treats them as weak evidence.

Pure Python + stdlib on purpose: no model downloads, easy to test.
"""
from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Sequence

# A small built-in English stop-word list avoids a runtime NLTK/spaCy download.
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not now
    of off on once only or other our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves""".split()
)

_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def preprocess_text(text: object) -> str:
    """Normalise unicode, strip control characters and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _CONTROL_RE.sub(" ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def tokenize_words(text: str) -> List[str]:
    """Lower-cased word tokens (letters/digits, keeps internal apostrophes)."""
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    """Heuristic sentence splitter (punctuation + newlines).

    It will mis-split abbreviations such as "e.g." or "Dr."; this is acceptable
    for a statistical signal, and avoids a heavy NLP dependency.
    """
    parts = _SENT_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p and _WORD_RE.search(p)]


# ---------------------------------------------------------------- primitives
def char_count(text: str) -> int:
    return len(text)


def word_count(words: Sequence[str]) -> int:
    return len(words)


def average_word_length(words: Sequence[str]) -> float:
    return sum(len(w) for w in words) / len(words) if words else 0.0


def type_token_ratio(words: Sequence[str]) -> float:
    """unique words / total words. Strongly length-dependent: longer texts have
    lower TTR for both humans and machines, so compare only similar lengths."""
    return len(set(words)) / len(words) if words else 0.0


def sentence_length_stats(sentence_lengths: Sequence[int]) -> tuple[float, float]:
    """Mean and population standard deviation of sentence lengths (in words)."""
    if not sentence_lengths:
        return 0.0, 0.0
    mean = statistics.fmean(sentence_lengths)
    std = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    return mean, std


def burstiness(sentence_lengths: Sequence[int]) -> float:
    """std(sentence length) / mean(sentence length).

    Undefined for fewer than two sentences; 0.0 is returned and the scorer
    ignores the value via a reliability check.
    """
    mean, std = sentence_length_stats(sentence_lengths)
    return std / mean if mean > 0 else 0.0


def stopword_ratio(words: Sequence[str]) -> float:
    return sum(1 for w in words if w in STOPWORDS) / len(words) if words else 0.0


def punctuation_ratio(text: str) -> float:
    """Punctuation characters / non-whitespace characters."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(1 for c in non_ws if unicodedata.category(c).startswith("P")) / len(non_ws)


def repetition_ratio(words: Sequence[str], n: int = 3) -> float:
    """Share of word n-grams that are repeats: (total - unique) / total."""
    if n < 1 or len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values())
    return repeated / len(grams)


# ------------------------------------------------------------------ bundle
@dataclass
class TextFeatures:
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    ttr: float = 0.0
    sentence_length_mean: float = 0.0
    sentence_length_std: float = 0.0
    burstiness: float = 0.0
    stopword_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    repetition_ratio: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


FEATURE_NAMES: List[str] = [f.name for f in fields(TextFeatures)]


def extract_features(text: object, repetition_n: int = 3) -> TextFeatures:
    """Compute all features. Never raises on odd input: empty, punctuation-only,
    non-string, very short and very long text all return a valid object."""
    clean = preprocess_text(text)
    if not clean:
        return TextFeatures()
    words = tokenize_words(clean)
    sentences = split_sentences(clean)
    lengths = [len(tokenize_words(s)) for s in sentences]
    mean, std = sentence_length_stats(lengths)
    return TextFeatures(
        char_count=char_count(clean),
        word_count=word_count(words),
        sentence_count=len(sentences),
        avg_word_length=average_word_length(words),
        ttr=type_token_ratio(words),
        sentence_length_mean=mean,
        sentence_length_std=std,
        burstiness=burstiness(lengths),
        stopword_ratio=stopword_ratio(words),
        punctuation_ratio=punctuation_ratio(clean),
        repetition_ratio=repetition_ratio(words, repetition_n),
    )


def feature_vector(text: object) -> List[float]:
    """Fixed-order numeric vector (order = FEATURE_NAMES) for ML models."""
    d = extract_features(text).to_dict()
    return [float(d[name]) for name in FEATURE_NAMES]


def is_finite_features(f: TextFeatures) -> bool:
    return all(math.isfinite(float(v)) for v in f.to_dict().values())
