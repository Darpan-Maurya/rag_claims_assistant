import pandas as pd

from rag.chunking import build_parent_child_chunks
from rag.corrective import evaluate_retrieval_quality
from rag.graph_context import ClaimsGraphContext
from rag.query_expansion import expand_query


def test_parent_child_sentence_window_chunks():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM00001"],
            "claim_text": [
                "Claim CLM00001 has diabetes. Claim status is DENIED. The denial reason was Pre-authorization missing."
            ],
        }
    )
    chunks = build_parent_child_chunks(df, window_size=1)
    assert "sentence_window_child" in set(chunks["chunk_type"])
    assert "parent_claim" in set(chunks["chunk_type"])
    assert chunks["parent_row_index"].nunique() == 1


def test_query_expansion_is_domain_aware():
    expansions = expand_query("denied diabetes claims")
    assert expansions
    assert any("diabetic" in item or "denial" in item for item in expansions)


def test_crag_quality_evaluator_marks_empty_as_incorrect():
    quality = evaluate_retrieval_quality({"returned_count": 0, "candidate_count": 0})
    assert quality.action == "incorrect"


def test_claims_graph_context_returns_relationships():
    df = pd.DataFrame(
        {
            "claim_id": ["CLM00001", "CLM00002"],
            "disease": ["Diabetes", "Diabetes"],
            "speciality": ["Cardiology", "Pulmonology"],
            "payer_name": ["A", "B"],
            "denial_reason": ["Pre-authorization missing", "Pre-authorization missing"],
            "network_status": ["IN_NETWORK", "IN_NETWORK"],
            "plan_type": ["PPO", "PPO"],
            "provider_type": ["Hospital", "Clinic"],
            "claim_status": ["DENIED", "DENIED"],
        }
    )
    graph = ClaimsGraphContext(df)
    related = graph.related_claims([0])
    assert related
    assert related[0]["claim_id"] == "CLM00002"
