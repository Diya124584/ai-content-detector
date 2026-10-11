"""Run robustness comparisons from a CSV you provide.

Required columns: original_text, modified_text
Optional columns: perturbation_type (paraphrased_ai | human_edited_ai | unseen_generator | different_domain ...)

  python -m scripts.run_robustness --pairs ../data/test/pairs.csv --out ../results

Writes robustness_pairs.csv, robustness_report.json (usable by the Trust Score
after you copy it to data/processed/robustness_report.json) and prints a summary.
Use --synthetic to build smoke-test pairs from a CSV with a `text` column
(typo / shuffle perturbations only; these are NOT real paraphrases).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.pipeline import TextDetectionPipeline
from app.robustness import compare_texts, summarize, synthetic_sentence_shuffle, synthetic_typos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.pairs)
    if a.synthetic:
        base = df["text"].tolist()
        df = pd.DataFrame({"original_text": base + base, "modified_text": [synthetic_typos(t) for t in base] + [synthetic_sentence_shuffle(t) for t in base],
                           "perturbation_type": ["synthetic_typos"] * len(base) + ["synthetic_shuffle"] * len(base)})
    if a.max_pairs:
        df = df.head(a.max_pairs)
    pipe = TextDetectionPipeline.from_settings(Settings.from_env())
    results = [compare_texts(pipe.analyze, r.original_text, r.modified_text, getattr(r, "perturbation_type", None)) for r in df.itertuples()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in results]).to_csv(out / "robustness_pairs.csv", index=False)
    rep = summarize(results, dataset=str(a.pairs))
    rep.save(out / "robustness_report.json")
    print(rep)


if __name__ == "__main__":
    main()
