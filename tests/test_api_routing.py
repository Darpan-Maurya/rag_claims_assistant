from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from api import main as api_main
from core.security import Principal
from rag.web_search import WebSearchResult


class FakeClaimsData:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def dataframe(self) -> pd.DataFrame:
        return self.df

    def readiness(self):
        return {"ready": True, "rows": len(self.df), "source": "test", "reason": None}


class NoopStorage:
    def ensure_conversation(self, *_args, **_kwargs):
        return None

    def add_message(self, *_args, **_kwargs):
        return None

    def add_retrieval_trace(self, *_args, **_kwargs):
        return None

    def add_audit_event(self, *_args, **_kwargs):
        return None


class NoopCache:
    def make_key(self, **_kwargs):
        return "test-key"

    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None

    def readiness(self):
        return {"enabled": False}


def _claims_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["CLM1", "CLM2", "CLM3", "CLM4"],
            "claim_status": ["APPROVED", "DENIED", "DENIED", "PENDING"],
            "claim_amount": [100.0, 200.0, 300.0, 400.0],
            "denial_reason": ["", "Documentation", "Coverage", ""],
            "disease": ["Diabetes"] * 4,
            "speciality": ["Cardiology"] * 4,
            "payer_name": ["A"] * 4,
            "service_date": pd.to_datetime(["2024-01-01"] * 4),
        }
    )


def _client(monkeypatch):
    monkeypatch.setattr(api_main, "retriever", None)
    monkeypatch.setattr(api_main, "claims_data", FakeClaimsData(_claims_df()))
    monkeypatch.setattr(api_main, "storage", NoopStorage())
    monkeypatch.setattr(api_main, "query_cache", NoopCache())
    principal = Principal(user_id="test-admin", role="admin", allowed_payers=["*"])
    api_main.app.dependency_overrides[api_main.require_principal] = lambda: principal
    return TestClient(api_main.app)


def test_analytics_does_not_require_qdrant_retriever(monkeypatch):
    client = _client(monkeypatch)
    try:
        response = client.post("/query", json={"query": "What percentage of claims are denied?"})
    finally:
        api_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "ANALYTICS"
    assert "50.0% were denied" in payload["answer"]
    assert payload["filters"] == {}


def test_llm_only_does_not_require_claims_data_or_qdrant(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(api_main, "answer_general_query", lambda _query: "General explanation")
    try:
        response = client.post("/query", json={"query": "What is prior authorization?"})
    finally:
        api_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "LLM_ONLY"
    assert payload["answer"] == "General explanation"
    assert not payload["evidence"]


def test_rag_returns_503_when_retriever_is_unavailable(monkeypatch):
    client = _client(monkeypatch)
    try:
        response = client.post("/query", json={"query": "Show denied claims for diabetes"})
    finally:
        api_main.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Retriever is not ready" in response.json()["detail"]


def test_web_route_does_not_require_claims_data_or_qdrant(monkeypatch):
    client = _client(monkeypatch)

    class FakeWebSearch:
        provider = "tavily"
        is_available = True

        def search(self, _query):
            return [
                WebSearchResult(
                    title="CMS source",
                    url="https://www.cms.gov/example",
                    content="Public claims guidance",
                )
            ]

    monkeypatch.setattr(api_main, "web_search", FakeWebSearch())
    monkeypatch.setattr(api_main, "answer_web_query", lambda _query, _sources: "Web answer")
    try:
        response = client.post("/query", json={"query": "Find the latest CMS claims regulations"})
    finally:
        api_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "WEB_SEARCH"
    assert payload["answer"] == "Web answer"
    assert payload["evidence"][0]["url"] == "https://www.cms.gov/example"
    assert payload["retrieval_summary"]["router"]["data_source"] == "public_web"


def test_claim_record_request_routes_to_rag_with_symbolic_amount_filter(monkeypatch):
    client = _client(monkeypatch)
    captured = {}

    class FakeRetriever:
        def readiness(self):
            return {"ready": True}

        def retrieve_with_details(self, **kwargs):
            captured["filters"] = kwargs["filters"].to_dict()
            result = pd.DataFrame(
                {
                    "claim_id": ["CLM1"],
                    "claim_status": ["APPROVED"],
                    "claim_amount": [120000.0],
                    "disease": ["Diabetes"],
                    "payer_name": ["A"],
                }
            )
            return SimpleNamespace(
                results=result,
                retrieval_summary={
                    "top_rrf_score": 0.016,
                    "top_dense_score": 0.41,
                    "returned_count": 1,
                },
            )

    monkeypatch.setattr(api_main, "retriever", FakeRetriever())
    monkeypatch.setattr(api_main, "answer_query_with_context", lambda **_kwargs: "Evidence CLM1")
    query = "want to have claims which are approved and have claim amount > 100000"
    try:
        response = client.post("/query", json={"query": query})
    finally:
        api_main.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "RAG"
    assert payload["answer"] == "Evidence CLM1"
    assert captured["filters"] == {"claim_status": "APPROVED", "amount_min": 100000.0}
    assert payload["filters"] == {"claim_status": "APPROVED", "amount_min": 100000.0}
