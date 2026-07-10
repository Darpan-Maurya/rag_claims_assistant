from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.config import settings


@dataclass(frozen=True)
class VectorSearchHit:
    point_id: int
    score: float
    payload: Dict[str, Any]


class VectorSearchBackend(Protocol):
    """Interface for swapping Qdrant Local, Qdrant Cloud, or another vector DB."""

    def recreate_collection(self, vector_size: int) -> None:
        ...

    def upsert_vectors(
        self,
        embeddings: np.ndarray,
        payloads: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        ...

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[VectorSearchHit]:
        ...

    @property
    def size(self) -> int:
        ...

    def readiness(self) -> Dict[str, Any]:
        ...


class QdrantVectorBackend:
    def __init__(
        self,
        collection_name: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        local_path: Path | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.qdrant_collection
        self.url = url if url is not None else settings.qdrant_url
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        self.local_path = local_path or settings.qdrant_local_path
        if self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            self.mode = "remote"
        else:
            self.local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.local_path))
            self.mode = "local"

    def recreate_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        for field in [
            "claim_id",
            "parent_row_index",
            "payer_name",
            "claim_status",
            "disease",
            "speciality",
            "denial_reason",
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Payload indexes are an optimization. Collection creation should not fail
                # if a local/older Qdrant version does not support one field schema.
                pass

    def upsert_vectors(
        self,
        embeddings: np.ndarray,
        payloads: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        for start in range(0, len(payloads), batch_size):
            end = start + batch_size
            points = [
                models.PointStruct(
                    id=int(point_id),
                    vector=embeddings[point_id].tolist(),
                    payload=payload,
                )
                for point_id, payload in enumerate(payloads[start:end], start=start)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[VectorSearchHit]:
        query_vector = query_embedding.astype("float32").tolist()
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        except Exception:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        return [
            VectorSearchHit(
                point_id=int(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in points
        ]

    @property
    def size(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        try:
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                ).count
            )
        except Exception:
            collection = self.client.get_collection(self.collection_name)
            return int(collection.points_count or 0)

    def readiness(self) -> Dict[str, Any]:
        exists = self.client.collection_exists(self.collection_name)
        return {
            "backend": "qdrant",
            "mode": self.mode,
            "collection": self.collection_name,
            "collection_exists": exists,
            "points": self.size if exists else 0,
            "url": self.url or f"local:{self.local_path}",
        }
