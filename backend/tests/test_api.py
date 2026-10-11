import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from .helpers import SAMPLE, make_pipeline


def _client(tmp_path, **kw):
    s = Settings(database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}", **kw)
    return TestClient(create_app(s, make_pipeline()))


def test_root_and_health(tmp_path):
    c = _client(tmp_path)
    assert c.get("/").status_code == 200
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["database"]["ok"]


def test_detect_schema_and_history_autosave(tmp_path):
    c = _client(tmp_path)
    r = c.post("/detect", json={"text": SAMPLE})
    assert r.status_code == 200
    d = r.json()
    assert d["prediction"] in {"AI-generated", "Human-written", "Uncertain"}
    assert {"ai_probability", "human_probability", "detection_score", "trust_score", "signals", "explanation"} <= set(d)
    assert d["trust_score"]["status"] == "prototype / not yet calibrated"
    assert d["history_id"] is not None
    lst = c.post("/history/list", json={"limit": 5}).json()
    assert lst["total"] == 1 and "text" not in lst["items"][0] and "raw_text" not in lst["items"][0]
    assert c.post("/history", json={"id": d["history_id"]}).json()["id"] == d["history_id"]


@pytest.mark.parametrize("payload,code", [({"text": ""}, 422), ({"text": "too short"}, 422), ({"text": "x " * 30000}, 422), ({}, 422), ({"text": 5}, 422)])
def test_validation_errors(tmp_path, payload, code):
    assert _client(tmp_path).post("/detect", json=payload).status_code == code


def test_malformed_json_does_not_crash(tmp_path):
    r = _client(tmp_path).post("/detect", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_explain_robustness_and_status(tmp_path):
    c = _client(tmp_path)
    assert c.post("/explain", json={"text": SAMPLE}).json()["method"] == "signals"
    assert c.post("/explain", json={"text": SAMPLE, "method": "tfidf_terms"}).status_code == 409
    r = c.post("/robustness/compare", json={"original_text": SAMPLE, "modified_text": SAMPLE + " Extra words appended here for testing."})
    assert r.status_code == 200 and "probability_change" in r.json()
    st = c.post("/train/status").json()
    assert st["feature_model"]["trained"] is False


def test_history_save_and_404(tmp_path):
    c = _client(tmp_path)
    r = c.post("/history/save", json={"text": SAMPLE, "prediction": "Uncertain", "ai_probability": 0.5, "human_probability": 0.5, "detection_score": 0.5})
    assert r.status_code == 200
    assert c.post("/history", json={"id": 12345}).status_code == 404
