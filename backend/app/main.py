"""FastAPI application. Run:  uvicorn app.main:app --reload  (from backend/)"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings
from .database import DetectionRepository, create_repository
from .features import preprocess_text, tokenize_words
from .pipeline import TextDetectionPipeline
from .robustness import compare_texts
from .schemas import (
    CompareRequest, DetectRequest, DetectResponse, ExplainRequest, ExplainResponse,
    HistoryGetRequest, HistoryListRequest, HistoryListResponse, HistoryRecord, HistorySaveRequest,
)

log = logging.getLogger("app")


def validate_text(text: str, s: Settings, field: str = "text") -> str:
    clean = preprocess_text(text)
    if not clean:
        raise HTTPException(422, f"{field} must not be empty")
    if len(text) > s.max_chars:
        raise HTTPException(422, f"{field} is too long ({len(text)} chars; maximum {s.max_chars})")
    n = len(tokenize_words(clean))
    if n < s.min_words:
        raise HTTPException(422, f"{field} is too short ({n} words; minimum {s.min_words})")
    return clean


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[TextDetectionPipeline] = None,
    repository: Optional[DetectionRepository] = None,
) -> FastAPI:
    s = settings or Settings.from_env()
    pipe = pipeline or TextDetectionPipeline.from_settings(s)
    repo = repository or create_repository(s.database_url, s.store_raw_text, s.hash_salt)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if s.preload_models:
            pipe.preload()
        yield

    app = FastAPI(title=s.app_name, version=__version__, lifespan=lifespan,
                  description="Text module of the Multimodal AI-Generated Content Detection framework.")
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"], allow_headers=["*"])
    app.state.settings, app.state.pipeline, app.state.repo = s, pipe, repo

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {"name": s.app_name, "version": __version__, "model_version": s.model_version, "docs": "/docs",
                "note": "Detection output is probabilistic; see /docs for the API."}

    @app.get("/health")
    def health() -> Dict[str, Any]:
        try:
            db = {"ok": True, "records": repo.count()}
        except Exception as e:
            db = {"ok": False, "error": str(e)}
        return {"status": "ok", "components": pipe.status(), "database": db}

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest) -> Dict[str, Any]:
        clean = validate_text(req.text, s)
        res = pipe.analyze(clean).to_dict()
        res["trust_score"] = res.pop("trust")
        hist_id = None
        if (s.auto_save_history if req.save_history is None else req.save_history):
            try:
                hist_id = repo.save(text=clean, prediction=res["prediction"], ai_probability=res["ai_probability"],
                                    human_probability=res["human_probability"], detection_score=res["detection_score"],
                                    trust_score=res["trust_score"]["value"], model_version=res["model_version"])
            except Exception:
                log.exception("could not save history")
        res["history_id"] = hist_id
        return res

    @app.post("/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest) -> Dict[str, Any]:
        clean = validate_text(req.text, s)
        try:
            return pipe.explain(clean, req.method)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(409, str(e))

    @app.post("/robustness/compare")
    def robustness_compare(req: CompareRequest) -> Dict[str, Any]:
        a = validate_text(req.original_text, s, "original_text")
        b = validate_text(req.modified_text, s, "modified_text")
        return compare_texts(pipe.analyze, a, b, req.perturbation_type).to_dict()

    @app.post("/train/status")
    def train_status() -> Dict[str, Any]:
        """Read-only report of which trainable artifacts exist. Never starts training."""
        fm_path = Path(s.feature_model_path)
        meta = pipe.feature_model.metadata if pipe.feature_model else None
        datasets = sorted(p.name for p in Path(s.processed_dir).glob("*") if p.suffix in {".csv", ".json", ".jsonl", ".parquet"})
        results = sorted(p.name for p in Path(s.results_dir).glob("*.json"))
        deberta = Path(s.processed_dir) / "models" / "deberta"
        return {
            "feature_model": {"trained": fm_path.exists(), "path": str(fm_path), "metadata": meta},
            "datasets_in_processed_dir": datasets,
            "experiment_result_files": results,
            "deberta_checkpoint_present": deberta.exists(),
            "calibration_loaded": pipe.calibrator is not None,
            "robustness_report_loaded": pipe.robustness_report is not None,
            "note": "Training is run from the command line (see README); this endpoint only reports status.",
        }

    @app.post("/history", response_model=HistoryRecord)
    def history_get(req: HistoryGetRequest) -> Dict[str, Any]:
        rec = repo.get(req.id)
        if rec is None:
            raise HTTPException(404, "record not found")
        return rec

    @app.post("/history/save")
    def history_save(req: HistorySaveRequest) -> Dict[str, Any]:
        clean = validate_text(req.text, s)
        rid = repo.save(text=clean, prediction=req.prediction, ai_probability=req.ai_probability,
                        human_probability=req.human_probability, detection_score=req.detection_score,
                        trust_score=req.trust_score, model_version=req.model_version)
        return {"id": rid}

    @app.post("/history/list", response_model=HistoryListResponse)
    def history_list(req: HistoryListRequest) -> Dict[str, Any]:
        return {"total": repo.count(), "items": repo.list(req.limit, req.offset)}

    return app


app = create_app()
