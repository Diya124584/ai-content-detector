# Multimodal Framework for AI-Generated Content Detection - Text Module

BE AIML final-year research project. **Current scope: text only.** Image and audio detection are future extensions and are intentionally not implemented.

The text module labels input as **AI-generated**, **Human-written** or **Uncertain**, and returns an AI probability, a human probability, a detection score, linguistic/statistical signals, perplexity, an explanation, and a separate **Trust Score**.

> **Honesty note.** No accuracy figure is claimed anywhere in this repository. Every metric is produced only when you run the experiment scripts on a dataset you supply. The RoBERTa model used here is a *pretrained baseline*, not the research contribution. The fusion weights and Trust Score are *prototype* quantities until you calibrate/validate them.

## 1. Architecture

```
Text
 ↓
Preprocessing (unicode normalise, strip control chars)
 ↓
 ┌────────────────────────────┬─────────────────────────────────┬──────────────────────┐
 │                            │                                 │                      │
RoBERTa baseline        Linguistic features               TF-IDF + LogReg (optional,
(GPT-2-era detector)    Perplexity (GPT-2) / TTR /        trained from your dataset)
 │                      Burstiness / Repetition                   │
 └──────────────┬─────────────┴─────────────────────────────────┘
                ↓
        Score fusion (scoring.py)  - configurable weights, missing signals dropped & renormalised
                ↓
   AI-generated / Human-written / Uncertain  (configurable thresholds)
                ↓
   Trust Score (trust_score.py)  +  Explainability (explainability.py)
                ↓
           REST API (FastAPI)  ←→  SQLite history
                ↓
          React dashboard
```

Concepts you should be able to explain in the viva:

| Concept | Where | What it is |
|---|---|---|
| Pretrained baseline | `detector.py` `RobertaBaseline` | `openai-community/roberta-base-openai-detector`, trained mainly on GPT-2 output. Reference point only. |
| Feature-based model | `feature_model.py` | TF-IDF + Logistic Regression and Logistic Regression on linguistic features, trained from a dataset. |
| Fusion model | `scoring.py` | Weighted combination of signal "AI-ness" scores. Weights are prototype priors; `fit_fusion_weights` learns them from validation data. |
| Evaluation | `evaluation.py`, `scripts/run_experiments.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, abstention coverage. |
| Explainability | `explainability.py` | Per-signal contributions (always consistent with the fusion), TF-IDF term contributions, sentence occlusion, optional Captum Integrated Gradients. |
| Trust Score | `trust_score.py` | Reliability of *this prediction* (confidence, signal agreement, input reliability / OOD flags, calibration, robustness). Not the AI probability. |
| Research model (later) | `training/train.py` | Fine-tuned `microsoft/deberta-v3-base`, to be evaluated on unseen generators/domains. |

### Fusion in one paragraph
Each signal is mapped to an AI-ness score in [0,1]: RoBERTa gives P(AI); perplexity goes through a logistic map on log-perplexity (lower → more AI-like); burstiness/TTR/repetition are mapped with logistic functions around prior centres and averaged into one "linguistic" score; the TF-IDF model gives P(AI) if a model file exists. The final **detection score** is the weighted mean over the signals that are available. If a calibrator file exists, `ai_probability = calibrator(detection_score)`; otherwise it equals the detection score and a warning says so.

## 2. Tech stack
Python 3.11+, FastAPI, Uvicorn, PyTorch, Hugging Face Transformers/Datasets, scikit-learn, pandas, numpy, SQLite (stdlib `sqlite3` behind a repository interface), React + TypeScript + Tailwind (Vite), pytest, Docker/docker-compose. Optional: Captum, SHAP.

## 3. Installation (Windows, PowerShell)

Prerequisites: Python 3.11 or 3.12 (64-bit), Node.js 18+ (20 recommended), Git. ~3 GB free disk for model downloads (RoBERTa ~0.5 GB and GPT-2 ~0.5 GB are downloaded on first use and cached in `%USERPROFILE%\.cache\huggingface`).

```powershell
cd ai-content-detector\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# only for dataset loading from HF / DeBERTa training / Captum:
pip install -r requirements-research.txt
```

## 4. Run the backend
```powershell
cd ai-content-detector\backend
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs. Models load lazily on the first `/detect` call (set `PRELOAD_MODELS=true` to load at startup). To run without downloading anything (linguistic features only): `$env:ENABLE_ROBERTA="false"; $env:ENABLE_PERPLEXITY="false"`.

Quick call:
```powershell
curl.exe -X POST http://localhost:8000/detect -H "Content-Type: application/json" -d "{\"text\": \"Paste at least ten words of text here so that the detector will accept the request.\"}"
```

## 5. Run the frontend
```powershell
cd ai-content-detector\frontend
npm install
npm run dev
```
Open http://localhost:5173 (requests to `/api/*` are proxied to port 8000).

## 6. Docker
```powershell
docker compose up --build
```
Frontend: http://localhost:8080, API: http://localhost:8000. `data/` and `results/` are mounted as volumes; the HF model cache persists in a named volume.

## 7. Tests
```powershell
cd backend
pytest
# optional: also run the tests that download and execute the real models
$env:RUN_MODEL_TESTS="1"; pytest
```
Tests check behaviour and output schema, not detection accuracy.

## 8. Dataset format
CSV / JSON / JSON-Lines / Hugging Face dataset, mapped to:

```
text,label,source,generator,domain
```
`label`: 0 = human, 1 = AI. See `data/raw/README.md` for provenance logging. Loading and splitting is in `backend/app/data.py`: exact-duplicate removal (texts with conflicting labels dropped), stratified random split, grouped split (keeps all rows of a `source` together), and **hold-out split** for unseen-generator / unseen-domain evaluation. `check_leakage` verifies there is no text overlap between splits.

## 9. Train the feature model
```powershell
cd backend
python -m scripts.train_feature_model --data ..\data\processed\dataset.csv
# unseen-generator evaluation:
python -m scripts.train_feature_model --data ..\data\processed\dataset.csv --holdout-generator gpt-4
```
This writes `data/processed/models/tfidf_lr.joblib`; the API picks it up on the next start and adds it as a fusion signal. (`joblib` files are pickles: only load files you created.)

## 10. Run experiments
```powershell
cd backend
python -m scripts.run_experiments --data ..\data\processed\dataset.csv --experiments A B C D --max-test 500
python -m scripts.run_experiments --data ..\data\processed\dataset.csv --experiments A B C D --holdout-generator gpt-4
```
| Exp | What |
|---|---|
| A | TF-IDF + Logistic Regression |
| B | Pretrained RoBERTa baseline |
| C | Linguistic + log-perplexity features + Logistic Regression |
| D | Fusion: prototype weights and weights learned on the validation split; also writes a candidate calibration report |
| E | Fine-tuned transformer checkpoint (`--experiments E --checkpoint <dir>`) |

Output in `results/`: `<exp>_metrics.json`, `results_summary.csv`, `<exp>_confusion.png`, `<exp>_roc.png`. A notebook version is in `notebooks/experiments.ipynb`. B/C/D run transformer models one text at a time on CPU; use `--max-train/--max-val/--max-test`.

**Activating learned artifacts** (copy from `results/` after you have inspected them): `fusion_config_fitted.json` → `data/processed/fusion_config.json`; a calibration report → `data/processed/calibration.json`; `robustness_report.json` → `data/processed/robustness_report.json`. The Trust Score status changes accordingly.

### Robustness
```powershell
python -m scripts.run_robustness --pairs ..\data\test\pairs.csv
```
`pairs.csv` columns: `original_text, modified_text, perturbation_type` (paraphrased_ai, human_edited_ai, unseen_generator, different_domain, ...). The modified texts must be produced by you (e.g. by a paraphraser, human editing, another LLM); the code does not fabricate them. `--synthetic` creates typo/shuffle pairs for smoke tests only. Reports prediction change, probability change and score change; the API exposes one pair at a time at `POST /robustness/compare`.

### Fine-tune DeBERTa (later; needs a dataset, GPU recommended)
```powershell
python -m training.train --data ..\data\processed\dataset.csv --output-dir ..\data\processed\models\deberta --epochs 3
```
Weighted cross-entropy for class imbalance, early stopping on validation F1, best-checkpoint saving, fixed seed, test metrics saved to `test_metrics.json`.

## 11. API
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | service info |
| GET | `/health` | status of models (not forced to load), DB |
| POST | `/detect` | `{"text": "..."}` → prediction, ai/human probability, detection score, trust score, signals, explanation, warnings |
| POST | `/explain` | `{"text", "method": "signals" \| "tfidf_terms" \| "occlusion" \| "integrated_gradients"}` |
| POST | `/robustness/compare` | `{"original_text", "modified_text", "perturbation_type"?}` → prediction/probability/score change |
| POST | `/train/status` | read-only: which trainable artifacts/results exist (never starts training) |
| POST | `/history` | `{"id": n}` → one record |
| POST | `/history/save` | save a result explicitly |
| POST | `/history/list` | `{"limit", "offset"}` |

Validation: empty text, fewer than `MIN_WORDS` (10), more than `MAX_CHARS` (20,000), and malformed JSON return 422 with a message. History stores a salted SHA-256 hash only, unless `STORE_RAW_TEXT=true`. Configuration is via environment variables (`.env.example`).

## 12. Limitations
- **No accuracy is established.** Until you run the experiments on a real dataset there is no evidence of performance.
- The RoBERTa baseline targets GPT-2-era text and is known to be weak on modern LLM output; paraphrasing and human editing degrade most detectors; non-native English writing can be falsely flagged.
- Perplexity, burstiness, TTR and repetition are weak, genre- and length-dependent signals. Low TTR does not prove AI; high perplexity does not prove human. TTR falls with text length for any author.
- Fusion weights, perplexity pivot and thresholds are prototype priors. Until validated, `ai_probability` is a score, not a calibrated probability.
- Trust Score is a heuristic reliability indicator labelled "prototype / not yet calibrated" until calibration **and** robustness artifacts exist; even then it must be re-validated.
- Language: English only (stop-word list, GPT-2/RoBERTa).
- Explanations describe what the model used; they do not prove who wrote the text. Integrated Gradients is optional, slow and not covered by automated tests.
- Sentence splitting is heuristic (abbreviations may split wrongly).
- Not suitable as sole evidence in academic-integrity or disciplinary decisions.

## 13. Research methodology
1. **Baselines**: A (TF-IDF+LR), B (pretrained RoBERTa, no training), C (linguistic+perplexity+LR).
2. **Proposed system**: D (fusion), then E (fine-tuned DeBERTa-v3) as the primary research model.
3. **Evaluation**: Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, plus abstention coverage for the three-state output; in-distribution vs **unseen generator** vs **unseen domain** splits; leakage checks.
4. **Calibration**: fit temperature on validation, report ECE/Brier on test.
5. **Robustness**: paraphrased AI, human-edited AI, unseen generator, different domain; report prediction/probability/score change and stability.
6. **Trust Score**: combine confidence, agreement, OOD flags, calibration and robustness; validate by checking whether higher trust corresponds to lower error on held-out data (not done yet).
7. **Reporting**: cite datasets and generators precisely; do not report numbers that were not produced by these scripts.

References (inspiration only, no code copied): Paresh-ch-Pothal/AI-Text-Detection; SelfCheckGPT (hallucination/consistency method, not used as the classifier); iyadalagha/ai-text-detector-app (idea of combining RoBERTa + perplexity + burstiness + TTR); *A Survey on Detection of LLMs-Generated Content*; Ghostbuster; RAID; DetectGPT; Fast-DetectGPT.

## 14. Future work
Fine-tuned DeBERTa as primary model; DetectGPT/Fast-DetectGPT-style curvature features; learned (stacked) fusion; temperature/Platt calibration and conformal abstention; adversarial robustness (paraphrase attacks, homoglyphs); SelfCheckGPT-style consistency as a *separate* hallucination module; SHAP/Integrated-Gradients in the UI; PostgreSQL repository; image and audio modules to complete the multimodal framework.

## Project layout
```
backend/app/  config, features, perplexity, detector, feature_model, scoring, trust_score,
              explainability, pipeline, robustness, data, evaluation, database, schemas, main
backend/scripts/  train_feature_model, run_experiments, run_robustness
backend/training/train.py   DeBERTa fine-tuning
backend/tests/  pytest suite
frontend/       Vite + React + TS + Tailwind dashboard
data/{raw,processed,test}/  datasets and artifacts (git-ignored)
notebooks/experiments.ipynb  results/  Dockerfile  docker-compose.yml
```
