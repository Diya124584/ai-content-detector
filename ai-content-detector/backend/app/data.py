"""Dataset loading and leakage-aware splitting.

Canonical schema:  text,label,source,generator,domain
  label: 0 = human, 1 = AI.   Optional columns default to "unknown".

No dataset is downloaded automatically. Document where every dataset comes from
(see data/raw/README.md) and check its licence yourself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import preprocess_text

COLUMNS = ["text", "label", "source", "generator", "domain"]
OPTIONAL = ["source", "generator", "domain"]


def standardize(df: pd.DataFrame, text_col: str = "text", label_col: str = "label",
                label_map: Optional[Dict[Any, int]] = None, min_chars: int = 1) -> pd.DataFrame:
    """Rename/validate columns, map labels to {0,1}, clean text, drop empties,
    and drop duplicates (texts that appear with conflicting labels are removed)."""
    df = df.rename(columns={text_col: "text", label_col: "label"}).copy()
    if "text" not in df or "label" not in df:
        raise ValueError(f"need '{text_col}' and '{label_col}' columns, got {list(df.columns)}")
    if label_map:
        df["label"] = df["label"].map(label_map)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    for c in OPTIONAL:
        if c not in df:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str)
    df["text"] = df["text"].map(preprocess_text)
    df = df[df["text"].str.len() >= min_chars].copy()
    df["_h"] = df["text"].map(text_hash)
    conflicting = df.groupby("_h")["label"].nunique()
    df = df[~df["_h"].isin(conflicting[conflicting > 1].index)]
    df = df.drop_duplicates("_h").drop(columns="_h")
    return df[COLUMNS].reset_index(drop=True)


def text_hash(t: str) -> str:
    return hashlib.sha1(preprocess_text(t).lower().encode("utf-8")).hexdigest()


def load_csv(path: str | Path, **kw: Any) -> pd.DataFrame:
    return standardize(pd.read_csv(path), **kw)


def load_json(path: str | Path, **kw: Any) -> pd.DataFrame:
    """JSON array of records, or JSON Lines."""
    p = Path(path)
    try:
        df = pd.read_json(p)
    except ValueError:
        df = pd.read_json(p, lines=True)
    return standardize(df, **kw)


def load_hf(name: str, split: str = "train", config: Optional[str] = None, *, text_col: str = "text",
            label_col: str = "label", label_map: Optional[Dict[Any, int]] = None,
            generator: Optional[str] = None, domain: Optional[str] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a Hugging Face dataset (needs `datasets` and internet). Column names
    differ per dataset, so you map them explicitly; ``source`` is set to the
    dataset id for provenance."""
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    df = ds.to_pandas()
    if max_rows:
        df = df.head(max_rows)
    df["source"] = name
    if generator and "generator" not in df:
        df["generator"] = generator
    if domain and "domain" not in df:
        df["domain"] = domain
    return standardize(df, text_col=text_col, label_col=label_col, label_map=label_map)


def load_dataset_any(path_or_name: str, **kw: Any) -> pd.DataFrame:
    p = Path(path_or_name)
    if p.suffix == ".csv":
        return load_csv(p, **kw)
    if p.suffix in {".json", ".jsonl"}:
        return load_json(p, **kw)
    return load_hf(path_or_name, **kw)


# ------------------------------------------------------------------ splits
def _stratified_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    chosen: List[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        chosen.extend(idx[: max(1, int(round(len(idx) * frac)))] if len(idx) > 1 else idx[:0])
    return np.asarray(sorted(chosen), dtype=int)


def split_random(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split (in-distribution). Duplicates were removed in
    ``standardize`` so exact-text leakage is impossible; near-duplicates and
    shared prompts can still leak: prefer split_grouped / split_holdout."""
    df = df.reset_index(drop=True)
    y = df["label"].to_numpy()
    test_idx = _stratified_indices(y, test_frac, seed)
    rest = df.drop(index=test_idx)
    val_idx = _stratified_indices(rest["label"].to_numpy(), val_frac / (1 - test_frac), seed + 1)
    val = rest.iloc[val_idx]
    train = rest.drop(index=rest.index[val_idx])
    return train.reset_index(drop=True), val.reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_grouped(df: pd.DataFrame, group_col: str, val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Whole groups (e.g. `source` document/prompt id) go to exactly one split,
    which prevents leakage from paired human/AI texts about the same prompt."""
    groups = np.array(sorted(set(df[group_col].astype(str))), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_test, n_val = max(1, int(len(groups) * test_frac)), max(1, int(len(groups) * val_frac))
    test_g, val_g = set(groups[:n_test]), set(groups[n_test : n_test + n_val])
    g = df[group_col].astype(str)
    return (df[~g.isin(test_g | val_g)].reset_index(drop=True), df[g.isin(val_g)].reset_index(drop=True), df[g.isin(test_g)].reset_index(drop=True))


def split_holdout(df: pd.DataFrame, column: str, holdout_values: Iterable[str], val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Unseen-generator / unseen-domain evaluation: the test set contains ONLY
    rows whose `column` (e.g. "generator" or "domain") is in holdout_values; the
    train/val sets contain none of them."""
    hv = {str(v) for v in holdout_values}
    mask = df[column].astype(str).isin(hv)
    if not mask.any():
        raise ValueError(f"no rows with {column} in {sorted(hv)}")
    test = df[mask].reset_index(drop=True)
    rest = df[~mask].reset_index(drop=True)
    val_idx = _stratified_indices(rest["label"].to_numpy(), val_frac, seed)
    val = rest.iloc[val_idx]
    train = rest.drop(index=rest.index[val_idx])
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def check_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, int]:
    """Count exact-text overlaps between splits (should all be 0)."""
    h = {n: set(d["text"].map(text_hash)) for n, d in (("train", train), ("val", val), ("test", test))}
    return {"train_val": len(h["train"] & h["val"]), "train_test": len(h["train"] & h["test"]), "val_test": len(h["val"] & h["test"])}


def describe(df: pd.DataFrame) -> Dict[str, Any]:
    return {"n": int(len(df)), "n_ai": int(df["label"].sum()), "n_human": int((df["label"] == 0).sum()),
            "generators": df["generator"].value_counts().to_dict(), "domains": df["domain"].value_counts().to_dict()}


def save_splits(out_dir: str | Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(out / f"{name}.csv", index=False)
