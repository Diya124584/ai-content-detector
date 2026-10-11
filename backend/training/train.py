"""Fine-tune microsoft/deberta-v3-base (or any HF encoder) for AI-vs-human text.

NOT run automatically. Provide a dataset first (columns: text,label,source,generator,domain):

  python -m training.train --data ../data/processed/dataset.csv \
      --output-dir ../data/processed/models/deberta --epochs 3 --holdout-generator gpt4

Features: stratified/unseen-generator/domain splits, class-imbalance handling
(weighted cross-entropy), early stopping on validation F1, best-checkpoint saving,
fixed seeds, final test metrics written to <output-dir>/test_metrics.json.
Needs: pip install -r requirements-research.txt (accelerate, sentencepiece, datasets).
GPU strongly recommended; CPU training of DeBERTa-base is impractically slow.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from app.data import check_leakage, describe, load_dataset_any, split_holdout, split_random
from app.evaluation import compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--output-dir", default="../data/processed/models/deberta")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--patience", type=int, default=2, help="early-stopping patience (epochs)")
    ap.add_argument("--holdout-generator", nargs="*")
    ap.add_argument("--holdout-domain", nargs="*")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only; DeBERTa-v3 can be unstable in fp16)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer,
                              TrainingArguments, set_seed)

    set_seed(a.seed)
    df = load_dataset_any(a.data)
    print("dataset:", json.dumps(describe(df))[:400])
    if a.holdout_generator:
        tr, va, te = split_holdout(df, "generator", a.holdout_generator, seed=a.seed)
    elif a.holdout_domain:
        tr, va, te = split_holdout(df, "domain", a.holdout_domain, seed=a.seed)
    else:
        tr, va, te = split_random(df, seed=a.seed)
    print("leakage:", check_leakage(tr, va, te), "| sizes:", len(tr), len(va), len(te))

    tok = AutoTokenizer.from_pretrained(a.model_name)
    def enc(d):
        ds = Dataset.from_pandas(d[["text", "label"]].reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_length), batched=True, remove_columns=["text"])
    dtr, dva, dte = enc(tr), enc(va), enc(te)

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_name, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1})

    counts = np.bincount(tr["label"].to_numpy(), minlength=2).astype(float)
    class_w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float)  # inverse-frequency

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=class_w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
        m = compute_metrics(p.label_ids, prob)
        return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if m[k] is not None}

    kw = dict(output_dir=a.output_dir, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
              per_device_eval_batch_size=a.batch_size * 2, learning_rate=a.lr, weight_decay=0.01, warmup_ratio=0.1,
              save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
              save_total_limit=2, seed=a.seed, fp16=a.fp16, report_to="none", logging_steps=50)
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kw[key] = "epoch"
    trainer = WeightedTrainer(model=model, args=TrainingArguments(**kw), train_dataset=dtr, eval_dataset=dva,
                              tokenizer=tok, compute_metrics=metrics,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
    trainer.train()
    trainer.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)

    pred = trainer.predict(dte)
    prob = torch.softmax(torch.tensor(pred.predictions), -1)[:, 1].numpy()
    result = {"model": a.model_name, "seed": a.seed, "n_test": len(te), **compute_metrics(pred.label_ids, prob)}
    Path(a.output_dir, "test_metrics.json").write_text(json.dumps(result, indent=2))
    print("test metrics:", {k: v for k, v in result.items() if k != "confusion_matrix"})


if __name__ == "__main__":
    main()
