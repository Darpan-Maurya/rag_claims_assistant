from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from core.config import settings
from rag.chunking import build_parent_child_chunks
from rag.vector_backends import QdrantVectorBackend

# =====================
# PATHS
# =====================
PROCESSED_DATA_PATH = settings.processed_data_path
METADATA_PATH = settings.metadata_path
CHUNK_METADATA_PATH = settings.chunk_metadata_path

# =====================
# CONFIG
# =====================
EMBEDDING_MODEL_NAME = settings.embedding_model_name


def load_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    if "claim_text" not in df.columns:
        raise ValueError("Expected 'claim_text' column in processed data")
    return df


def build_embeddings(texts, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # helps cosine similarity
    )
    return embeddings.astype("float32")


def save_metadata(df: pd.DataFrame, chunks: pd.DataFrame):
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Parent metadata for analytics and evidence rendering.
    df.to_parquet(METADATA_PATH, index=False)
    # Child metadata aligned to Qdrant point IDs.
    chunks.to_parquet(CHUNK_METADATA_PATH, index=False)


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_payloads(chunks: pd.DataFrame, parents: pd.DataFrame) -> list[Dict[str, Any]]:
    payloads: list[Dict[str, Any]] = []
    parent_fields = [
        "claim_id",
        "payer_name",
        "claim_status",
        "disease",
        "speciality",
        "denial_reason",
        "service_date",
        "claim_amount",
    ]
    for _, chunk in chunks.iterrows():
        parent = parents.iloc[int(chunk["parent_row_index"])]
        payload: Dict[str, Any] = {
            "chunk_id": _jsonable(chunk["chunk_id"]),
            "parent_row_index": int(chunk["parent_row_index"]),
            "chunk_type": _jsonable(chunk["chunk_type"]),
            "child_text": _jsonable(chunk["child_text"]),
            "window_text": _jsonable(chunk["window_text"]),
            "sentence_index": int(chunk["sentence_index"]),
        }
        for field in parent_fields:
            if field in parent:
                payload[field] = _jsonable(parent[field])
        payloads.append(payload)
    return payloads


def build_vector_collection(embeddings: np.ndarray, chunks: pd.DataFrame, parents: pd.DataFrame) -> None:
    backend = QdrantVectorBackend()
    backend.recreate_collection(vector_size=int(embeddings.shape[1]))
    backend.upsert_vectors(embeddings=embeddings, payloads=build_payloads(chunks, parents))


def run():
    print("📥 Loading processed data...")
    df = load_data()

    chunks = build_parent_child_chunks(df)
    texts = chunks["child_text"].tolist()
    print(f"✅ Loaded {len(df)} claims and {len(texts)} retrieval chunks for indexing.")

    print("🧠 Building embeddings...")
    embeddings = build_embeddings(texts, EMBEDDING_MODEL_NAME)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    print("📦 Building Qdrant vector collection...")
    build_vector_collection(embeddings, chunks, df)
    print(f"✅ Qdrant collection '{settings.qdrant_collection}' built with {len(chunks)} vectors.")

    print("💾 Saving index and metadata...")
    save_metadata(df, chunks)

    print("🎉 Done. Vector collection saved in Qdrant.")
    print(f"📄 Metadata saved at {METADATA_PATH}")
    print(f"📄 Chunk metadata saved at {CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    run()
