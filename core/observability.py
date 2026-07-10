import json
import logging
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Iterator

from core.config import settings


_SENSITIVE_PATTERNS = [
    re.compile(r"\bPAT\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bCLM\d{3,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
]


def redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = value
    for pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        for key in ("request_id", "route", "status_code", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


logger = logging.getLogger("rag_claims")


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Counter[str] = Counter()
        self._latencies: Dict[str, list[float]] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe_latency(self, name: str, duration_ms: float) -> None:
        with self._lock:
            self._latencies.setdefault(name, []).append(duration_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latencies = {}
            for name, values in self._latencies.items():
                if not values:
                    continue
                ordered = sorted(values)
                latencies[name] = {
                    "count": len(values),
                    "avg_ms": round(sum(values) / len(values), 2),
                    "p95_ms": round(ordered[int((len(ordered) - 1) * 0.95)], 2),
                }
            return {"counters": dict(self._counters), "latencies": latencies}


metrics = MetricsRegistry()


@contextmanager
def timed_metric(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.observe_latency(name, duration_ms)


def safe_query_for_log(query: str) -> str:
    if settings.log_full_queries:
        return redact_text(query) or ""
    return f"len={len(query)}"
