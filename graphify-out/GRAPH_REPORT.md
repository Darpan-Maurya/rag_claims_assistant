# Graph Report - .  (2026-07-28)

## Corpus Check
- Corpus is ~10,903 words - fits in a single context window. You may not need a graph.

## Summary
- 235 nodes · 529 edges · 23 communities (20 shown, 3 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Advanced Retrieval
- Analytics and Evaluation
- Access Control
- Vector Database
- Persistent Storage
- Configuration and LLM
- Indexing Pipeline
- API Contracts
- API Runtime and Ingestion
- Observability
- ETL Validation
- Query Orchestration
- Query Cache
- Streamlit UI
- Core Package

## God Nodes (most connected - your core abstractions)
1. `ClaimsRetriever` - 35 edges
2. `Principal` - 29 edges
3. `query_claims()` - 22 edges
4. `AppStorage` - 20 edges
5. `QueryFilters` - 19 edges
6. `QueryCache` - 16 edges
7. `extract_filters()` - 14 edges
8. `QdrantVectorBackend` - 13 edges
9. `answer_analytics_query()` - 11 edges
10. `apply_filters()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `QueryRequest` --uses--> `QueryCache`  [INFERRED]
  api/main.py → core/cache.py
- `QueryRequest` --uses--> `Principal`  [INFERRED]
  api/main.py → core/security.py
- `QueryRequest` --uses--> `AppStorage`  [INFERRED]
  api/main.py → core/storage.py
- `QueryRequest` --uses--> `ClaimsRetriever`  [INFERRED]
  api/main.py → rag/retriever.py
- `QueryResponse` --uses--> `QueryCache`  [INFERRED]
  api/main.py → core/cache.py

## Import Cycles
- None detected.

## Communities (23 total, 3 thin omitted)

### Community 0 - "Advanced Retrieval"
Cohesion: 0.11
Nodes (15): evaluate_retrieval_quality(), RetrievalQuality, ClaimsGraphContext, DataFrame, Lightweight structured GraphRAG layer for claims. This is not a full LLM-…, expand_query(), ClaimsRetriever, DataFrame (+7 more)

### Community 1 - "Analytics and Evaluation"
Cohesion: 0.16
Nodes (22): answer_analytics_query(), approval_percentage(), Any, DataFrame, date, load_golden(), main(), check_input_guardrails() (+14 more)

### Community 2 - "Access Control"
Cohesion: 0.17
Nodes (13): require_ingestion_principal(), require_metrics_principal(), require_principal(), apply_row_level_access(), authenticate_api_key(), _configured_principals(), enforce_filter_access(), Principal (+5 more)

### Community 3 - "Vector Database"
Cohesion: 0.14
Nodes (8): Protocol, Any, ndarray, Path, QdrantVectorBackend, Interface for swapping Qdrant Local, Qdrant Cloud, or another vector DB., VectorSearchBackend, VectorSearchHit

### Community 4 - "Persistent Storage"
Cohesion: 0.25
Nodes (5): Connection, AppStorage, Any, Path, _utc_now()

### Community 5 - "Configuration and LLM"
Cohesion: 0.18
Nodes (11): _as_json(), _as_list(), Any, Settings, answer_query_with_context(), build_context_from_claims(), _fallback_evidence_answer(), DataFrame (+3 more)

### Community 6 - "Indexing Pipeline"
Cohesion: 0.24
Nodes (14): build_embeddings(), build_payloads(), build_vector_collection(), _jsonable(), load_data(), Any, DataFrame, ndarray (+6 more)

### Community 7 - "API Contracts"
Cohesion: 0.21
Nodes (13): create_ingestion_job(), FeedbackRequest, FeedbackResponse, IngestionRequest, IngestionResponse, QueryResponse, request_context_middleware(), submit_feedback() (+5 more)

### Community 8 - "API Runtime and Ingestion"
Cohesion: 0.27
Nodes (10): get_ingestion_job(), get_ready_retriever(), health_check(), load_retriever(), metrics_endpoint(), readiness_check(), run_ingestion_job(), choose_denial_reason() (+2 more)

### Community 9 - "Observability"
Cohesion: 0.20
Nodes (6): configure_logging(), JsonFormatter, MetricsRegistry, Any, timed_metric(), LogRecord

### Community 10 - "ETL Validation"
Cohesion: 0.38
Nodes (10): extract(), load(), DataFrame, run_etl(), transform(), validate_input(), write_manifest(), base_claims_df() (+2 more)

### Community 11 - "Query Orchestration"
Cohesion: 0.32
Nodes (8): _evidence_from_df(), Any, query_claims(), QueryRequest, Accepts a natural language query and returns a routed, evidence-grounded answer., _validate_query_request(), redact_text(), safe_query_for_log()

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ClaimsRetriever` connect `Advanced Retrieval` to `Analytics and Evaluation`, `Vector Database`, `Configuration and LLM`, `API Contracts`, `API Runtime and Ingestion`, `Query Orchestration`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `AppStorage` connect `Persistent Storage` to `API Runtime and Ingestion`, `Query Orchestration`, `API Contracts`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `QdrantVectorBackend` connect `Vector Database` to `Advanced Retrieval`, `Indexing Pipeline`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `ClaimsRetriever` (e.g. with `FeedbackRequest` and `FeedbackResponse`) actually correct?**
  _`ClaimsRetriever` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Principal` (e.g. with `FeedbackRequest` and `FeedbackResponse`) actually correct?**
  _`Principal` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `AppStorage` (e.g. with `FeedbackRequest` and `FeedbackResponse`) actually correct?**
  _`AppStorage` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `QueryFilters` (e.g. with `Principal` and `ClaimsRetriever`) actually correct?**
  _`QueryFilters` has 3 INFERRED edges - model-reasoned connections that need verification._