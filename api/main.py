import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analytics.claims_analytics import answer_analytics_query
from core.cache import QueryCache
from core.config import settings
from core.observability import (
    configure_logging,
    logger,
    metrics,
    redact_text,
    safe_query_for_log,
)
from core.security import (
    Principal,
    apply_row_level_access,
    authenticate_api_key,
    enforce_filter_access,
    visible_payers,
)
from core.storage import AppStorage
from etl.etl_pipeline import run_etl
from generate_mock_data import OUTPUT_PATH, generate_claims
from orchestrate.guardrails import check_input_guardrails, validate_grounded_answer
from orchestrate.router import classify_query, extract_filters
from rag.build_index import run as build_vector_index
from rag.llm_answer import answer_query_with_context
from rag.retriever import ClaimsRetriever

configure_logging()

# =====================
# INITIALIZE APP
# =====================
app = FastAPI(
    title="RAG-Powered Claims Assistant",
    description="Query insurance claims using natural language (RAG + LLM)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.increment("api.errors")
        logger.exception("Unhandled API error", extra={"request_id": request_id})
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    metrics.observe_latency("api.request", duration_ms)
    logger.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


# =====================
# LOAD RAG COMPONENTS
# =====================
storage = AppStorage()
query_cache = QueryCache()
retriever: Optional[ClaimsRetriever] = None


def load_retriever() -> Optional[ClaimsRetriever]:
    try:
        return ClaimsRetriever()
    except Exception as exc:
        logger.warning("retriever_not_ready", extra={"request_id": "startup"})
        storage.add_audit_event("retriever_not_ready", {"error": str(exc)})
        return None


try:
    retriever = load_retriever()
except Exception as e:
    logger.exception("startup_failed")
    raise RuntimeError(f"Failed to initialize app state: {e}")

# =====================
# REQUEST / RESPONSE SCHEMAS
# =====================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = settings.default_top_k
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    route: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    retrieval_summary: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    conversation_id: str
    request_id: str


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal["up", "down", "neutral"]
    conversation_id: Optional[str] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class IngestionRequest(BaseModel):
    regenerate_data: bool = False
    rebuild_index: bool = True


class IngestionResponse(BaseModel):
    job_id: str
    status: str


def require_principal(x_api_key: Optional[str] = Header(default=None)) -> Principal:
    principal = authenticate_api_key(x_api_key)
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return principal


def get_ready_retriever() -> ClaimsRetriever:
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever is not ready. Run an ingestion job and check /ready.",
        )
    return retriever


def require_metrics_principal(principal: Principal = Depends(require_principal)) -> Principal:
    if not principal.can_view_metrics:
        raise HTTPException(status_code=403, detail="Metrics require claims_manager or admin role")
    return principal


def require_ingestion_principal(principal: Principal = Depends(require_principal)) -> Principal:
    if not principal.can_ingest:
        raise HTTPException(status_code=403, detail="Ingestion requires claims_manager or admin role")
    return principal


def _validate_query_request(request: QueryRequest) -> None:
    if len(request.query) > settings.max_query_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Query is too long. Maximum is {settings.max_query_chars} characters.",
        )
    if request.top_k < 1 or request.top_k > settings.max_top_k:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must be between 1 and {settings.max_top_k}.",
        )


def _evidence_from_df(df) -> List[Dict[str, Any]]:
    evidence_fields = [
        "claim_id",
        "disease",
        "diagnosis_code",
        "speciality",
        "procedure_code",
        "claim_status",
        "denial_reason",
        "claim_amount",
        "allowed_amount",
        "paid_amount",
        "payer_name",
        "service_date",
        "retrieval_score",
        "dense_score",
        "lexical_score",
        "rerank_score",
        "chunk_id",
        "chunk_type",
        "matched_child_text",
        "context_window",
    ]
    rows: List[Dict[str, Any]] = []
    for _, row in df.head(settings.max_top_k).iterrows():
        item: Dict[str, Any] = {}
        for field in evidence_fields:
            if field in row and row[field] == row[field]:
                value = row[field]
                if hasattr(value, "item"):
                    value = value.item()
                item[field] = value.isoformat() if hasattr(value, "isoformat") else value
        rows.append(item)
    return rows


# =====================
# ROUTES
# =====================
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    retriever_ready = retriever.readiness() if retriever else {"ready": False, "reason": "not_loaded"}
    return {
        "status": "ready" if retriever_ready["ready"] else "not_ready",
        "retriever": retriever_ready,
        "cache": query_cache.readiness(),
        "gemini_configured": bool(settings.gemini_api_key),
        "storage": storage.counts(),
    }


@app.get("/metrics")
def metrics_endpoint(principal: Principal = Depends(require_metrics_principal)):
    return {
        "runtime": metrics.snapshot(),
        "storage": storage.counts(),
        "cache": query_cache.readiness(),
        "principal": {"user_id": principal.user_id, "role": principal.role},
    }


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
):
    request_id = request.state.request_id
    storage.add_feedback(
        request_id=payload.request_id,
        conversation_id=payload.conversation_id,
        rating=payload.rating,
        notes_redacted=redact_text(payload.notes),
    )
    storage.add_audit_event(
        "feedback_submitted",
        {
            "rating": payload.rating,
            "target_request_id": payload.request_id,
            "user_id": principal.user_id,
        },
        request_id=request_id,
    )
    metrics.increment("feedback.submitted")
    return FeedbackResponse(status="stored", request_id=payload.request_id)


@app.post("/ingestion/jobs", response_model=IngestionResponse)
def create_ingestion_job(
    payload: IngestionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    principal: Principal = Depends(require_ingestion_principal),
):
    job_id = str(uuid4())
    options = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    storage.create_ingestion_job(job_id, options)
    storage.add_audit_event(
        "ingestion_job_queued",
        {"job_id": job_id, "user_id": principal.user_id, **options},
        request_id=request.state.request_id,
    )
    background_tasks.add_task(run_ingestion_job, job_id, payload.regenerate_data, payload.rebuild_index)
    return IngestionResponse(job_id=job_id, status="queued")


@app.get("/ingestion/jobs/{job_id}")
def get_ingestion_job(
    job_id: str,
    principal: Principal = Depends(require_ingestion_principal),
):
    job = storage.get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job


def run_ingestion_job(job_id: str, regenerate_data: bool, rebuild_index: bool) -> None:
    global retriever
    try:
        storage.update_ingestion_job(job_id, "running")
        if regenerate_data:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            generate_claims().to_csv(OUTPUT_PATH, index=False)
        run_etl()
        if rebuild_index:
            build_vector_index()
        invalidated = query_cache.invalidate_all()
        retriever = load_retriever()
        if retriever is None:
            raise RuntimeError("Ingestion completed but retriever did not become ready.")
        storage.update_ingestion_job(job_id, "succeeded")
        storage.add_audit_event(
            "ingestion_job_succeeded",
            {"job_id": job_id, "cache_entries_invalidated": invalidated},
        )
    except Exception as exc:
        storage.update_ingestion_job(job_id, "failed", error=str(exc))
        storage.add_audit_event("ingestion_job_failed", {"job_id": job_id, "error": str(exc)})
        logger.exception("ingestion_job_failed", extra={"request_id": job_id})


@app.post("/query", response_model=QueryResponse)
def query_claims(
    payload: QueryRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
    active_retriever: ClaimsRetriever = Depends(get_ready_retriever),
):
    """
    Accepts a natural language query and returns a routed, evidence-grounded answer.
    """
    request_id = request.state.request_id
    conversation_id = payload.conversation_id or str(uuid4())
    _validate_query_request(payload)

    try:
        guardrail = check_input_guardrails(payload.query)
        storage.ensure_conversation(conversation_id)
        storage.add_message(conversation_id, "user", redact_text(payload.query) or "")

        if not guardrail.allowed:
            metrics.increment("route.BLOCKED")
            storage.add_audit_event(
                "guardrail_blocked",
                {"reason": guardrail.blocked_reason, "query": safe_query_for_log(payload.query)},
                request_id=request_id,
            )
            answer = guardrail.blocked_reason or "The request was blocked by safety guardrails."
            storage.add_message(conversation_id, "assistant", answer)
            return QueryResponse(
                answer=answer,
                route="BLOCKED",
                warnings=guardrail.warnings,
                conversation_id=conversation_id,
                request_id=request_id,
            )

        accessible_df = apply_row_level_access(active_retriever.df, principal)
        if accessible_df.empty:
            raise HTTPException(status_code=403, detail="No row-level access grants are available.")

        route = classify_query(payload.query)
        filters = enforce_filter_access(extract_filters(payload.query, accessible_df), principal)
        metrics.increment(f"route.{route}")
        allowed_payers = visible_payers(
            principal,
            active_retriever.df["payer_name"].dropna().astype(str).unique(),
        )
        cache_key = query_cache.make_key(
            user_id=principal.user_id,
            role=principal.role,
            allowed_payers=allowed_payers,
            query=payload.query,
            top_k=payload.top_k,
            route=route,
            filters=filters.to_dict(),
        )
        cached = query_cache.get(cache_key)
        if cached:
            metrics.increment("cache.hit")
            answer = cached["answer"]
            retrieval_summary = cached.get("retrieval_summary", {})
            retrieval_summary["cache"] = {"hit": True, "backend": "redis"}
            evidence = cached.get("evidence", [])
            warnings = cached.get("warnings", [])
            storage.add_message(conversation_id, "assistant", redact_text(answer) or "")
            storage.add_retrieval_trace(
                request_id=request_id,
                conversation_id=conversation_id,
                route=route,
                query_redacted=safe_query_for_log(payload.query),
                filters=filters.to_dict(),
                summary=retrieval_summary,
            )
            return QueryResponse(
                answer=answer,
                route=route,
                evidence=evidence,
                filters=filters.to_dict(),
                retrieval_summary=retrieval_summary,
                warnings=warnings,
                conversation_id=conversation_id,
                request_id=request_id,
            )

        metrics.increment("cache.miss")

        if route == "ANALYTICS":
            analytics = answer_analytics_query(payload.query, accessible_df, filters)
            answer = analytics["answer"]
            retrieval_summary = analytics["metrics"]
            retrieval_summary["rbac"] = {
                "role": principal.role,
                "allowed_payers": allowed_payers,
            }
            retrieval_summary["cache"] = {"hit": False}
            evidence = []
        else:
            details = active_retriever.retrieve_with_details(
                query=payload.query,
                k=payload.top_k,
                filters=filters,
                allowed_parent_indices=set(accessible_df.index.tolist()),
            )
            retrieved_df = details.results

            if retrieved_df.empty:
                answer = "Insufficient evidence: no relevant claims were found for the given query."
            elif details.retrieval_summary.get("top_score", 0.0) < settings.min_relevance_score:
                answer = "Insufficient evidence: retrieved claims were below the relevance threshold."
            else:
                answer = answer_query_with_context(
                    user_query=payload.query,
                    retrieved_df=retrieved_df,
                )
            retrieval_summary = details.retrieval_summary
            retrieval_summary["cache"] = {"hit": False}
            evidence = _evidence_from_df(retrieved_df)

        evidence_claim_ids = [str(item.get("claim_id")) for item in evidence if item.get("claim_id")]
        warnings = guardrail.warnings + validate_grounded_answer(answer, evidence_claim_ids, route)

        storage.add_message(conversation_id, "assistant", redact_text(answer) or "")
        storage.add_retrieval_trace(
            request_id=request_id,
            conversation_id=conversation_id,
            route=route,
            query_redacted=safe_query_for_log(payload.query),
            filters=filters.to_dict(),
            summary=retrieval_summary,
        )
        storage.add_audit_event(
            "query_processed",
            {"route": route, "warnings": warnings},
            request_id=request_id,
        )
        query_cache.set(
            cache_key,
            {
                "answer": answer,
                "route": route,
                "evidence": evidence,
                "filters": filters.to_dict(),
                "retrieval_summary": retrieval_summary,
                "warnings": warnings,
            },
        )

        return QueryResponse(
            answer=answer,
            route=route,
            evidence=evidence,
            filters=filters.to_dict(),
            retrieval_summary=retrieval_summary,
            warnings=warnings,
            conversation_id=conversation_id,
            request_id=request_id,
        )

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        metrics.increment("api.query_errors")
        logger.exception("query_failed", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
