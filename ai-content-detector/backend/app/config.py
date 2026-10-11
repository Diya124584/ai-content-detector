"""Central configuration. Every tunable value is read from an environment
variable (or ``.env`` exported into the environment) with a documented default.

Nothing here is a "scientifically optimal" setting; defaults are prototype
choices that exist so the system runs end-to-end on a CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_opt_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Content Detector - Text Module"
    # Bump when the fusion logic / default weights change, so history rows can
    # be traced to the code that produced them.
    model_version: str = "text-fusion-prototype-0.1"

    # ---- models -------------------------------------------------------
    roberta_model_name: str = "openai-community/roberta-base-openai-detector"
    # Only needed if the checkpoint exposes generic labels (LABEL_0/LABEL_1).
    roberta_ai_label_index: Optional[int] = None
    perplexity_model_name: str = "gpt2"
    device: str = "cpu"  # "cpu", "cuda" or "auto"
    enable_roberta: bool = True
    enable_perplexity: bool = True
    preload_models: bool = False  # load at startup instead of first request
    max_roberta_chunks: int = 8  # bounds CPU cost for long documents
    max_perplexity_tokens: int = 2048

    # ---- optional artifacts (all may be absent) ------------------------
    feature_model_path: Path = PROJECT_ROOT / "data" / "processed" / "models" / "tfidf_lr.joblib"
    fusion_config_path: Path = PROJECT_ROOT / "data" / "processed" / "fusion_config.json"
    calibration_path: Path = PROJECT_ROOT / "data" / "processed" / "calibration.json"
    robustness_report_path: Path = PROJECT_ROOT / "data" / "processed" / "robustness_report.json"
    results_dir: Path = PROJECT_ROOT / "results"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # ---- input validation ---------------------------------------------
    min_words: int = 10  # below this we refuse to analyse
    reliable_words: int = 50  # below this we analyse but flag low reliability
    max_chars: int = 20_000

    # ---- database ------------------------------------------------------
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'detections.db').as_posix()}"
    store_raw_text: bool = False  # privacy default: only a hash is stored
    auto_save_history: bool = True
    hash_salt: str = ""

    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        origins = os.getenv("CORS_ORIGINS")
        return cls(
            model_version=os.getenv("MODEL_VERSION", d.model_version),
            roberta_model_name=os.getenv("ROBERTA_MODEL_NAME", d.roberta_model_name),
            roberta_ai_label_index=_env_opt_int("ROBERTA_AI_LABEL_INDEX"),
            perplexity_model_name=os.getenv("PERPLEXITY_MODEL_NAME", d.perplexity_model_name),
            device=os.getenv("DEVICE", d.device),
            enable_roberta=_env_bool("ENABLE_ROBERTA", d.enable_roberta),
            enable_perplexity=_env_bool("ENABLE_PERPLEXITY", d.enable_perplexity),
            preload_models=_env_bool("PRELOAD_MODELS", d.preload_models),
            max_roberta_chunks=_env_int("MAX_ROBERTA_CHUNKS", d.max_roberta_chunks),
            max_perplexity_tokens=_env_int("MAX_PERPLEXITY_TOKENS", d.max_perplexity_tokens),
            feature_model_path=_env_path("FEATURE_MODEL_PATH", d.feature_model_path),
            fusion_config_path=_env_path("FUSION_CONFIG_PATH", d.fusion_config_path),
            calibration_path=_env_path("CALIBRATION_PATH", d.calibration_path),
            robustness_report_path=_env_path("ROBUSTNESS_REPORT_PATH", d.robustness_report_path),
            results_dir=_env_path("RESULTS_DIR", d.results_dir),
            processed_dir=_env_path("PROCESSED_DIR", d.processed_dir),
            min_words=_env_int("MIN_WORDS", d.min_words),
            reliable_words=_env_int("RELIABLE_WORDS", d.reliable_words),
            max_chars=_env_int("MAX_CHARS", d.max_chars),
            database_url=os.getenv("DATABASE_URL", d.database_url),
            store_raw_text=_env_bool("STORE_RAW_TEXT", d.store_raw_text),
            auto_save_history=_env_bool("AUTO_SAVE_HISTORY", d.auto_save_history),
            hash_salt=os.getenv("HASH_SALT", d.hash_salt),
            cors_origins=tuple(o.strip() for o in origins.split(",")) if origins else d.cors_origins,
        )
