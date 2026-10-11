"""Classical, dataset-trained models (no pretrained transformers).

1. FeatureTextClassifier  - TF-IDF + Logistic Regression ("Experiment A").
2. NumericFeatureClassifier - scaler + Logistic Regression on the linguistic
   (and optionally perplexity) feature vector ("Experiment C").

Both are trained from a labelled dataset (label 0 = human, 1 = AI); nothing is
hard-coded. Models are persisted with joblib: only load files you created
yourself, because pickle can execute code.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, feature_vector, preprocess_text


def _validate_xy(texts: Sequence[str], labels: Sequence[int]) -> Tuple[List[str], np.ndarray]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels differ in length")
    y = np.asarray(labels, dtype=int)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0 (human) or 1 (AI)")
    if len(set(y.tolist())) < 2:
        raise ValueError("training data must contain both classes")
    return [preprocess_text(t) for t in texts], y


class FeatureTextClassifier:
    def __init__(self, pipeline: Optional[Pipeline] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.pipeline = pipeline
        self.metadata = metadata or {}

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int = 50_000,
        C: float = 1.0,
        seed: int = 42,
    ) -> "FeatureTextClassifier":
        x, y = _validate_xy(texts, labels)
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features, sublinear_tf=True)),
                ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
        pipe.fit(x, y)
        meta = {
            "kind": "tfidf_logreg",
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_train": int(len(y)),
            "n_ai": int(y.sum()),
            "label_definition": {"0": "human", "1": "ai"},
            "hyperparameters": {"ngram_range": list(ngram_range), "min_df": min_df, "max_features": max_features, "C": C, "seed": seed},
        }
        return cls(pipe, meta)

    def _require(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("model not trained or loaded")
        return self.pipeline

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """P(AI) for each text."""
        pipe = self._require()
        return pipe.predict_proba([preprocess_text(t) for t in texts])[:, 1]

    def predict_one(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    def term_contributions(self, text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Exact per-term contribution (coef * tf-idf value) to the logit.

        For a linear model this equals the model's own additive decomposition,
        so it can never contradict the prediction.
        """
        pipe = self._require()
        vec: TfidfVectorizer = pipe.named_steps["tfidf"]
        clf: LogisticRegression = pipe.named_steps["clf"]
        row = vec.transform([preprocess_text(text)])
        names = vec.get_feature_names_out()
        coefs = clf.coef_[0]
        items = [(names[j], float(row[0, j] * coefs[j])) for j in row.nonzero()[1]]
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return [
            {"term": t, "contribution": c, "direction": "toward_ai" if c > 0 else "toward_human"}
            for t, c in items[:top_k]
        ]

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._require(), "metadata": self.metadata}, p)

    @classmethod
    def load(cls, path: Path | str) -> "FeatureTextClassifier":
        blob = joblib.load(Path(path))
        return cls(blob["pipeline"], blob.get("metadata", {}))


class NumericFeatureClassifier:
    """Logistic regression on linguistic features (+ optional extra columns,
    e.g. log-perplexity). Input is a precomputed matrix so expensive features
    are computed once."""

    def __init__(self, pipeline: Optional[Pipeline] = None, columns: Optional[List[str]] = None) -> None:
        self.pipeline = pipeline
        self.columns = columns or list(FEATURE_NAMES)

    @classmethod
    def train(cls, X: np.ndarray, labels: Sequence[int], columns: Sequence[str], seed: int = 42) -> "NumericFeatureClassifier":
        y = np.asarray(labels, dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("training data must contain both classes")
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))])
        pipe.fit(np.nan_to_num(np.asarray(X, dtype=float)), y)
        return cls(pipe, list(columns))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model not trained")
        return self.pipeline.predict_proba(np.nan_to_num(np.asarray(X, dtype=float)))[:, 1]

    def coefficients(self) -> Dict[str, float]:
        clf = self.pipeline.named_steps["clf"]  # type: ignore[union-attr]
        return {c: float(w) for c, w in zip(self.columns, clf.coef_[0])}


def linguistic_matrix(texts: Sequence[str]) -> np.ndarray:
    return np.asarray([feature_vector(t) for t in texts], dtype=float)
