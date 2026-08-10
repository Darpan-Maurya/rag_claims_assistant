import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_list(value: str | None, default: List[str]) -> List[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RAG Claims Assistant")
    environment: str = os.getenv("APP_ENV", "development")
    api_key: str | None = os.getenv("RAG_API_KEY")
    cors_origins: List[str] = None  # type: ignore[assignment]
    max_query_chars: int = _as_int(os.getenv("MAX_QUERY_CHARS"), 1200)
    default_top_k: int = _as_int(os.getenv("DEFAULT_TOP_K"), 25)
    max_top_k: int = _as_int(os.getenv("MAX_TOP_K"), 50)
    min_dense_relevance_score: float = _as_float(
        os.getenv("MIN_DENSE_RELEVANCE_SCORE"), 0.20
    )

    raw_data_path: Path = Path(os.getenv("RAW_DATA_PATH", "data/raw/claims.csv"))
    processed_data_path: Path = Path(
        os.getenv("PROCESSED_DATA_PATH", "data/processed/claims_processed.parquet")
    )
    dataset_manifest_path: Path = Path(
        os.getenv("DATASET_MANIFEST_PATH", "data/processed/dataset_manifest.json")
    )
    metadata_path: Path = Path(os.getenv("RAG_METADATA_PATH", "rag/metadata.parquet"))
    chunk_metadata_path: Path = Path(os.getenv("CHUNK_METADATA_PATH", "rag/chunk_metadata.parquet"))
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", "data/app_state.sqlite3"))
    qdrant_url: str | None = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_local_path: Path = Path(os.getenv("QDRANT_LOCAL_PATH", "data/qdrant"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "claims_chunks")
    redis_url: str | None = os.getenv("REDIS_URL")
    query_cache_ttl_seconds: int = _as_int(os.getenv("QUERY_CACHE_TTL_SECONDS"), 300)
    enable_query_cache: bool = _as_bool(os.getenv("ENABLE_QUERY_CACHE"), True)

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model_name: str = os.getenv("GEMINI_MODEL_NAME", "models/gemini-flash-latest")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    query_classifier_model_path: Path = Path(
        os.getenv("QUERY_CLASSIFIER_MODEL_PATH", "models/query_classifier.joblib")
    )
    router_min_confidence: float = _as_float(os.getenv("ROUTER_MIN_CONFIDENCE"), 0.40)
    guardrail_min_confidence: float = _as_float(os.getenv("GUARDRAIL_MIN_CONFIDENCE"), 0.55)
    web_search_provider: str = os.getenv("WEB_SEARCH_PROVIDER", "disabled").strip().lower()
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    web_search_max_results: int = _as_int(os.getenv("WEB_SEARCH_MAX_RESULTS"), 5)
    web_search_timeout_seconds: int = _as_int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS"), 12)
    web_search_allowed_domains: List[str] = None  # type: ignore[assignment]
    enable_reranker: bool = _as_bool(os.getenv("ENABLE_RERANKER"), False)
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    log_full_queries: bool = _as_bool(os.getenv("LOG_FULL_QUERIES"), False)
    rbac_users: List[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(
            self,
            "cors_origins",
            _as_list(os.getenv("CORS_ORIGINS"), ["http://localhost:8501", "http://127.0.0.1:8501"]),
        )
        object.__setattr__(
            self,
            "rbac_users",
            _as_json(os.getenv("RAG_RBAC_USERS"), []),
        )
        object.__setattr__(
            self,
            "web_search_allowed_domains",
            _as_list(os.getenv("WEB_SEARCH_ALLOWED_DOMAINS"), []),
        )


settings = Settings()
