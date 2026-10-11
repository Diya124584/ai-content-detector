"""Train the TF-IDF + Logistic Regression model from a labelled dataset.

  python -m scripts.train_feature_model --data ../data/processed/dataset.csv \
      [--holdout-generator gpt4] [--out ../data/processed/models/tfidf_lr.joblib]
"""
from __future__ import annotations

import argparse
import json

from app.data import check_leakage, describe, load_dataset_any, split_holdout, split_random
from app.evaluation import compute_metrics, save_results
from app.feature_model import FeatureTextClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="CSV/JSON/HF dataset id (columns: text,label,source,generator,domain)")
    ap.add_argument("--out", default="../data/processed/models/tfidf_lr.joblib")
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--holdout-generator", nargs="*", help="generators reserved for the test set (unseen-generator evaluation)")
    ap.add_argument("--holdout-domain", nargs="*")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load_dataset_any(a.data)
    print("dataset:", json.dumps(describe(df), indent=2))
    if a.holdout_generator:
        tr, va, te = split_holdout(df, "generator", a.holdout_generator, seed=a.seed)
    elif a.holdout_domain:
        tr, va, te = split_holdout(df, "domain", a.holdout_domain, seed=a.seed)
    else:
        tr, va, te = split_random(df, seed=a.seed)
    print("leakage check:", check_leakage(tr, va, te))
    model = FeatureTextClassifier.train(tr["text"], tr["label"], min_df=a.min_df, seed=a.seed)
    model.save(a.out)
    print(f"saved model to {a.out}")
    for name, part in (("validation", va), ("test", te)):
        m = compute_metrics(part["label"], model.predict_proba(part["text"]))
        print(name, {k: v for k, v in m.items() if k != "confusion_matrix"})
        if name == "test":
            save_results("A_tfidf_logreg", m, a.results_dir, part["label"], model.predict_proba(part["text"]),
                         extra={"split": "unseen-generator" if a.holdout_generator else "unseen-domain" if a.holdout_domain else "random"})


if __name__ == "__main__":
    main()
