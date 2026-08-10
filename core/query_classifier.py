from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib

from core.config import settings


BLOCKED_LABELS = {
    "BLOCK_PRIVATE_DATA",
    "BLOCK_PROMPT_INJECTION",
    "BLOCK_UNSUPPORTED_ADVICE",
}

ROUTE_LABELS = {
    "ANALYTICS",
    "CLAIMS_RAG",
    "DECISION_HELP",
    "LLM_ONLY",
    "WEB_SEARCH",
}


class QueryClassifierUnavailable(RuntimeError):
    """The local supervised routing model is unavailable or invalid."""


@dataclass(frozen=True)
class QueryClassification:
    label: str
    confidence: float
    model_version: str

    @property
    def is_blocked(self) -> bool:
        return self.label in BLOCKED_LABELS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "model_version": self.model_version,
        }


class QueryClassifier:
    """Loads a compact, locally trained intent-and-safety classifier."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or settings.query_classifier_model_path
        self._pipeline: Any = None
        self._model_version = "unknown"
        self._error: str | None = None
        self.reload()

    def reload(self) -> bool:
        self._pipeline = None
        self._error = None
        if not self.model_path.exists():
            self._error = f"missing:{self.model_path}"
            return False
        try:
            payload = joblib.load(self.model_path)
            pipeline = payload["pipeline"]
            classifier = pipeline.named_steps["classifier"]
            if not hasattr(classifier, "predict_proba"):
                raise ValueError("classifier does not expose predict_proba")
            self._pipeline = pipeline
            self._model_version = str(payload.get("model_version", "unknown"))
            return True
        except Exception as exc:
            self._error = f"invalid:{type(exc).__name__}"
            return False

    def classify(self, query: str) -> QueryClassification:
        if self._pipeline is None:
            raise QueryClassifierUnavailable(
                "Query classifier is not ready. Run `python -m training.train_query_classifier`."
            )
        probabilities = self._pipeline.predict_proba([query])[0]
        labels = self._pipeline.named_steps["classifier"].classes_
        best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
        return QueryClassification(
            label=str(labels[best_index]),
            confidence=float(probabilities[best_index]),
            model_version=self._model_version,
        )

    def readiness(self) -> Dict[str, Any]:
        return {
            "ready": self._pipeline is not None,
            "model_path": str(self.model_path),
            "model_version": self._model_version if self._pipeline is not None else None,
            "reason": self._error,
        }
