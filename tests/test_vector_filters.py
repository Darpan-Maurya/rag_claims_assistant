import numpy as np
from qdrant_client.http import models

from rag.vector_backends import QdrantVectorBackend


def test_qdrant_dense_search_honors_payload_filter(tmp_path):
    backend = QdrantVectorBackend(
        collection_name="claims_test",
        url="",
        api_key=None,
        local_path=tmp_path / "qdrant",
    )
    backend.recreate_collection(vector_size=2)
    backend.upsert_vectors(
        np.array([[1.0, 0.0], [0.98, 0.02]], dtype="float32"),
        [
            {"claim_id": "CLM1", "payer_name": "MediPlus", "claim_status": "DENIED"},
            {"claim_id": "CLM2", "payer_name": "CareFirst", "claim_status": "DENIED"},
        ],
    )

    results = backend.search(
        np.array([1.0, 0.0], dtype="float32"),
        top_k=10,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="payer_name",
                    match=models.MatchValue(value="CareFirst"),
                )
            ]
        ),
    )

    assert [result.payload["claim_id"] for result in results] == ["CLM2"]
