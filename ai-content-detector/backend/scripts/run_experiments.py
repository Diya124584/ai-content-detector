"""Run experiments A-E on a labelled dataset and write real results to --out.

A  TF-IDF + Logistic Regression
B  Pretrained RoBERTa baseline (openai-community/roberta-base-openai-detector)
C  Linguistic features (+ log-perplexity) + Logistic Regression
D  Fusion (prototype weights, and weights learned on the validation split)
E  Fine-tuned transformer checkpoint from training/train.py (optional)

  python -m scripts.run_experiments --data ../data/processed/dataset.csv \
       --experiments A B C D --holdout-generator gpt4 --max-test 500

CPU note: B, C and D run transformer models text by text. Use --max-train /
--max-test to bound runtime. Nothing is reported unless it was computed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from app.config import Settings
from app.data import check_leakage, describe, load_dataset_any, split_holdout, split_random
from app.evaluation import compute_metrics, metrics_with_abstention, save_results
from app.feature_model import FeatureTextClassifier, NumericFeatureClassifier, linguistic_matrix
from app.features import FEATURE_NAMES
from app.pipeline import TextDetectionPipeline
from app.scoring import COMPONENTS, FusionConfig, RawSignals, component_scores, fit_fusion_weights, fuse
from app.trust_score import build_calibration_report


def _cap(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed).reset_index(drop=True) if n and len(df) > n else df


def _signals(pipe: TextDetectionPipeline, texts: List[str]) -> List[RawSignals]:
    out = []
    for i, t in enumerate(texts):
        raw, _ = pipe.compute_signals(t)
        out.append(raw)
        if (i + 1) % 25 == 0:
            print(f"  signals {i + 1}/{len(texts)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D"], choices=list("ABCDE"))
    ap.add_argument("--holdout-generator", nargs="*")
    ap.add_argument("--holdout-domain", nargs="*")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--checkpoint", default=None, help="fine-tuned model dir for experiment E")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load_dataset_any(a.data)
    if a.holdout_generator:
        tr, va, te = split_holdout(df, "generator", a.holdout_generator, seed=a.seed)
        split = "unseen-generator"
    elif a.holdout_domain:
        tr, va, te = split_holdout(df, "domain", a.holdout_domain, seed=a.seed)
        split = "unseen-domain"
    else:
        tr, va, te = split_random(df, seed=a.seed)
        split = "random"
    tr, va, te = _cap(tr, a.max_train, a.seed), _cap(va, a.max_val, a.seed), _cap(te, a.max_test, a.seed)
    print("data:", json.dumps(describe(df))[:300], "| split:", split, "| leakage:", check_leakage(tr, va, te))
    extra = {"split": split, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "data": str(a.data)}
    out = Path(a.out)
    ytr, yva, yte = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()

    settings = Settings.from_env()
    exp = set(a.experiments)
    pipe = TextDetectionPipeline.from_settings(settings) if exp & {"B", "C", "D"} else None

    if "A" in exp:
        m = FeatureTextClassifier.train(tr["text"], ytr, seed=a.seed)
        p = m.predict_proba(te["text"])
        save_results("A_tfidf_logreg", compute_metrics(yte, p), out, yte, p, extra)

    sig_va = sig_te = None
    if exp & {"B", "D"}:
        print("computing signals on test split ...")
        sig_te = _signals(pipe, list(te["text"]))
    if "B" in exp:
        p = np.asarray([s.roberta_ai if s.roberta_ai is not None else np.nan for s in sig_te])
        ok = ~np.isnan(p)
        if ok.any():
            save_results("B_roberta_baseline", compute_metrics(yte[ok], p[ok]), out, yte[ok], p[ok],
                         {**extra, "n_scored": int(ok.sum()), "note": "pretrained GPT-2-era baseline, no training on this data"})
        else:
            print("B skipped: RoBERTa produced no scores (check model download / warnings)")

    if "C" in exp:
        print("computing linguistic + perplexity features ...")
        def feat(texts):
            L = linguistic_matrix(texts)
            ppl = np.asarray([(pipe.perplexity.compute(t).perplexity if pipe.perplexity else None) or np.nan for t in texts], float)
            return np.column_stack([L, np.log(ppl)])
        cols = list(FEATURE_NAMES) + ["log_perplexity"]
        Xtr, Xte = feat(list(tr["text"])), feat(list(te["text"]))
        fill = np.nanmedian(Xtr, axis=0)  # fit on train only
        Xtr, Xte = np.where(np.isnan(Xtr), fill, Xtr), np.where(np.isnan(Xte), fill, Xte)
        clf = NumericFeatureClassifier.train(Xtr, ytr, cols, seed=a.seed)
        p = clf.predict_proba(Xte)
        save_results("C_linguistic_perplexity_logreg", compute_metrics(yte, p), out, yte, p, {**extra, "coefficients": clf.coefficients()})

    if "D" in exp:
        print("computing signals on validation split ...")
        sig_va = _signals(pipe, list(va["text"]))
        cfg0 = pipe.fusion_cfg if pipe.fusion_cfg else FusionConfig()
        def matrix(sigs):
            return np.asarray([[(v if v is not None else np.nan) for v in (component_scores(s, cfg0)[c] for c in COMPONENTS)] for s in sigs], float)
        p_proto = np.asarray([fuse(s, cfg0).detection_score for s in sig_te])
        save_results("D_fusion_prototype_weights", {**compute_metrics(yte, p_proto),
                     "abstention": metrics_with_abstention(yte, p_proto, cfg0.human_threshold, cfg0.ai_threshold)}, out, yte, p_proto,
                     {**extra, "weights": cfg0.weights(), "note": "hand-set prototype weights"})
        fitted = fit_fusion_weights(matrix(sig_va), yva, cfg0)
        p_fit = np.asarray([fuse(s, fitted).detection_score for s in sig_te])
        save_results("D_fusion_learned_weights", compute_metrics(yte, p_fit), out, yte, p_fit,
                     {**extra, "weights": fitted.weights(), "note": "weights fitted on validation split only"})
        fitted.save(out / "fusion_config_fitted.json")
        rep, cal = build_calibration_report([fuse(s, fitted).detection_score for s in sig_va], yva, dataset=f"{a.data} (validation split)")
        rep.save(out / "calibration_report_candidate.json")
        print("calibration candidate (fit on validation):", rep)

    if "E" in exp:
        if not a.checkpoint:
            print("E skipped: pass --checkpoint <dir produced by training/train.py>")
        else:
            from app.detector import FineTunedClassifier
            clf = FineTunedClassifier(a.checkpoint, device=settings.device, max_chunks=settings.max_roberta_chunks)
            p = np.asarray(clf.predict_proba_many(list(te["text"])))
            save_results("E_finetuned_transformer", compute_metrics(yte, p), out, yte, p, {**extra, "checkpoint": a.checkpoint})
    print("done; results in", out.resolve())


if __name__ == "__main__":
    main()
