"""Persistence for detection history.

``DetectionRepository`` is the interface the API depends on. The SQLite
implementation uses only the stdlib; to move to PostgreSQL implement the same
interface (e.g. with psycopg or SQLAlchemy) and return it from
``create_repository`` for ``postgresql://`` URLs. Nothing else needs to change.

Privacy: only a salted SHA-256 hash of the text is stored unless
STORE_RAW_TEXT=true.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .features import preprocess_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT NOT NULL,
    prediction TEXT NOT NULL,
    ai_probability REAL NOT NULL,
    human_probability REAL NOT NULL,
    detection_score REAL NOT NULL,
    trust_score REAL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    raw_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at);
CREATE INDEX IF NOT EXISTS idx_detections_hash ON detections(text_hash);
"""
COLUMNS = ["id", "text_hash", "prediction", "ai_probability", "human_probability",
           "detection_score", "trust_score", "model_version", "created_at"]


def hash_text(text: str, salt: str = "") -> str:
    return hashlib.sha256((salt + preprocess_text(text)).encode("utf-8")).hexdigest()


class DetectionRepository(ABC):
    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def save(self, *, text: str, prediction: str, ai_probability: float, human_probability: float,
             detection_score: float, trust_score: Optional[float], model_version: str) -> int: ...

    @abstractmethod
    def get(self, record_id: int) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...


class SQLiteDetectionRepository(DetectionRepository):
    def __init__(self, path: str | Path, store_raw_text: bool = False, salt: str = "") -> None:
        self.path = str(path)
        self.store_raw_text = store_raw_text
        self.salt = salt

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)  # one short-lived connection per call: thread-safe
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def save(self, *, text: str, prediction: str, ai_probability: float, human_probability: float,
             detection_score: float, trust_score: Optional[float], model_version: str) -> int:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        raw = preprocess_text(text) if self.store_raw_text else None
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO detections (text_hash, prediction, ai_probability, human_probability, "
                "detection_score, trust_score, model_version, created_at, raw_text) VALUES (?,?,?,?,?,?,?,?,?)",
                (hash_text(text, self.salt), prediction, ai_probability, human_probability,
                 detection_score, trust_score, model_version, now, raw),
            )
            return int(cur.lastrowid)

    def get(self, record_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(f"SELECT {', '.join(COLUMNS)} FROM detections WHERE id = ?", (record_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        limit, offset = max(1, min(int(limit), 200)), max(0, int(offset))
        with self._conn() as c:
            rows = c.execute(f"SELECT {', '.join(COLUMNS)} FROM detections ORDER BY id DESC LIMIT ? OFFSET ?",
                             (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM detections").fetchone()[0])


def create_repository(database_url: str, store_raw_text: bool = False, salt: str = "") -> DetectionRepository:
    if database_url.startswith("sqlite:///"):
        repo = SQLiteDetectionRepository(database_url[len("sqlite:///"):], store_raw_text, salt)
        repo.init()
        return repo
    raise NotImplementedError(
        "Only sqlite:/// is implemented. For PostgreSQL, implement DetectionRepository "
        "(see this module's docstring) and return it here."
    )
