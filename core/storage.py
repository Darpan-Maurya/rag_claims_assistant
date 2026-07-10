import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppStorage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_redacted TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieval_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    conversation_id TEXT,
                    route TEXT NOT NULL,
                    query_redacted TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    conversation_id TEXT,
                    rating TEXT NOT NULL,
                    notes_redacted TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    event_type TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def ensure_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations(conversation_id, created_at) VALUES (?, ?)",
                (conversation_id, _utc_now()),
            )

    def add_message(self, conversation_id: str, role: str, content_redacted: str) -> None:
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(conversation_id, role, content_redacted, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, role, content_redacted, _utc_now()),
            )

    def add_retrieval_trace(
        self,
        request_id: str,
        conversation_id: Optional[str],
        route: str,
        query_redacted: str,
        filters: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_traces(
                    request_id, conversation_id, route, query_redacted,
                    filters_json, summary_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    conversation_id,
                    route,
                    query_redacted,
                    json.dumps(filters, default=str),
                    json.dumps(summary, default=str),
                    _utc_now(),
                ),
            )

    def add_feedback(
        self,
        request_id: str,
        rating: str,
        conversation_id: Optional[str] = None,
        notes_redacted: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback(request_id, conversation_id, rating, notes_redacted, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, conversation_id, rating, notes_redacted, _utc_now()),
            )

    def add_audit_event(
        self, event_type: str, details: Dict[str, Any], request_id: Optional[str] = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events(request_id, event_type, details_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (request_id, event_type, json.dumps(details, default=str), _utc_now()),
            )

    def counts(self) -> Dict[str, int]:
        tables = [
            "conversations",
            "messages",
            "retrieval_traces",
            "feedback",
            "audit_events",
            "ingestion_jobs",
        ]
        with self._connect() as conn:
            return {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

    def create_ingestion_job(self, job_id: str, options: Dict[str, Any]) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs(job_id, status, options_json, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, "queued", json.dumps(options, default=str), None, now, now),
            )

    def update_ingestion_job(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, error, _utc_now(), job_id),
            )

    def get_ingestion_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None
