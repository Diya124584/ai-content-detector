from app.database import SQLiteDetectionRepository, create_repository, hash_text
import pytest


def _save(repo, text="some text to hash"):
    return repo.save(text=text, prediction="Uncertain", ai_probability=0.5, human_probability=0.5,
                     detection_score=0.5, trust_score=42.0, model_version="test")


def test_save_get_list_count(tmp_path):
    repo = create_repository(f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    ids = [_save(repo, f"text {i}") for i in range(3)]
    assert repo.count() == 3
    rec = repo.get(ids[0])
    assert rec["prediction"] == "Uncertain" and rec["trust_score"] == 42.0
    assert [r["id"] for r in repo.list(limit=2)] == [ids[2], ids[1]]  # newest first
    assert repo.get(9999) is None


def test_raw_text_not_stored_by_default(tmp_path):
    repo = create_repository(f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    _save(repo, "secret essay content")
    import sqlite3
    row = sqlite3.connect(repo.path).execute("SELECT raw_text, text_hash FROM detections").fetchone()
    assert row[0] is None and "secret" not in row[1]


def test_raw_text_stored_when_enabled(tmp_path):
    repo = SQLiteDetectionRepository(tmp_path / "y.db", store_raw_text=True)
    repo.init()
    _save(repo, "keep me")
    import sqlite3
    assert sqlite3.connect(repo.path).execute("SELECT raw_text FROM detections").fetchone()[0] == "keep me"


def test_hash_is_stable_and_salted():
    assert hash_text("a  b") == hash_text("a b")
    assert hash_text("a", "s1") != hash_text("a", "s2")


def test_postgres_url_not_implemented():
    with pytest.raises(NotImplementedError):
        create_repository("postgresql://u:p@h/db")
