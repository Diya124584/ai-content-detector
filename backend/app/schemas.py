"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(..., description="Text to analyse")
    save_history: Optional[bool] = Field(None, description="Override the server's AUTO_SAVE_HISTORY setting")


class TrustScoreModel(BaseModel):
    value: float = Field(..., description="0-100 reliability estimate. NOT the AI probability.")
    status: str
    components: Dict[str, Optional[float]]
    missing_components: List[str]
    ood_flags: List[str]
    disclaimer: str


class ExplanationItem(BaseModel):
    factor: str
    direction: str
    strength: float
    contribution: float
    detail: str


class DetectResponse(BaseModel):
    prediction: Literal["AI-generated", "Human-written", "Uncertain"]
    ai_probability: float
    human_probability: float
    detection_score: float
    trust_score: TrustScoreModel
    signals: Dict[str, Optional[float]]
    signal_scores: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    explanation: List[ExplanationItem]
    warnings: List[str]
    features: Dict[str, float]
    components_active: List[str]
    model_version: str
    disclaimer: str
    history_id: Optional[int] = None


class ExplainRequest(BaseModel):
    text: str
    method: Literal["signals", "tfidf_terms", "occlusion", "integrated_gradients"] = "signals"


class ExplainResponse(BaseModel):
    method: str
    items: List[Dict[str, Any]]


class CompareRequest(BaseModel):
    original_text: str
    modified_text: str
    perturbation_type: Optional[str] = None


class HistoryGetRequest(BaseModel):
    id: int


class HistorySaveRequest(BaseModel):
    text: str
    prediction: str
    ai_probability: float = Field(..., ge=0, le=1)
    human_probability: float = Field(..., ge=0, le=1)
    detection_score: float = Field(..., ge=0, le=1)
    trust_score: Optional[float] = None
    model_version: str = "unspecified"


class HistoryListRequest(BaseModel):
    limit: int = Field(20, ge=1, le=200)
    offset: int = Field(0, ge=0)


class HistoryRecord(BaseModel):
    id: int
    text_hash: str
    prediction: str
    ai_probability: float
    human_probability: float
    detection_score: float
    trust_score: Optional[float]
    model_version: str
    created_at: str


class HistoryListResponse(BaseModel):
    total: int
    items: List[HistoryRecord]
