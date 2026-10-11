"""Metrics, result files and plots. Numbers are only ever computed from real
predictions passed in; nothing is pre-filled."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, Any]:
    y, p = np.asarray(y_true, int), np.asarray(y_prob, float)
    pred = (p >= threshold).astype(int)
    auc: Optional[float]
    try:
        auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None
    except ValueError:
        auc = None
    return {
        "n": int(len(y)), "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),  # [[TN, FP], [FN, TP]]
    }


def metrics_with_abstention(y_true: Sequence[int], y_prob: Sequence[float], human_thr: float, ai_thr: float) -> Dict[str, Any]:
    """Three-state evaluation: 'Uncertain' predictions abstain. Reports coverage
    and accuracy on the covered subset, which is not comparable to full accuracy."""
    y, p = np.asarray(y_true, int), np.asarray(y_prob, float)
    covered = (p >= ai_thr) | (p <= human_thr)
    if not covered.any():
        return {"coverage": 0.0, "accuracy_on_covered": None, "n_uncertain": int(len(y))}
    pred = (p[covered] >= ai_thr).astype(int)
    return {"coverage": float(covered.mean()), "accuracy_on_covered": float((pred == y[covered]).mean()),
            "n_uncertain": int((~covered).sum())}


def save_results(name: str, metrics: Dict[str, Any], out_dir: str | Path, y_true: Optional[Sequence[int]] = None,
                 y_prob: Optional[Sequence[float]] = None, extra: Optional[Dict[str, Any]] = None, plots: bool = True) -> None:
    """Writes <name>_metrics.json, appends a row to results_summary.csv, and
    (optionally) <name>_confusion.png / <name>_roc.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": name, **metrics, **(extra or {})}
    (out / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2))
    import pandas as pd

    row = {k: v for k, v in payload.items() if k != "confusion_matrix" and not isinstance(v, (dict, list))}
    summary = out / "results_summary.csv"
    df = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    df = df[df.get("experiment", pd.Series(dtype=str)) != name] if "experiment" in df else df
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(summary, index=False)
    if plots and y_true is not None and y_prob is not None:
        _plots(name, np.asarray(y_true, int), np.asarray(y_prob, float), metrics, out)


def _plots(name: str, y: np.ndarray, p: np.ndarray, metrics: Dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["human", "AI"], yticklabels=["human", "AI"],
           xlabel="predicted", ylabel="true", title=f"{name}: confusion matrix")
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion.png", dpi=150)
    plt.close(fig)
    if len(set(y.tolist())) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlabel="false positive rate", ylabel="true positive rate", title=f"{name}: ROC")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / f"{name}_roc.png", dpi=150)
        plt.close(fig)
