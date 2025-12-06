from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================
# PATHS
# =====================
PROCESSED_DATA_PATH = Path("data/processed/claims_processed.parquet")
INDEX_PATH = Path("rag/vector_store.faiss")
METADATA_PATH = Path("rag/metadata.parquet")

# =====================
# CONFIG
# =====================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; works well with normalized vectors
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, df: pd.DataFrame):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    # Save metadata aligned by row index
    df.to_parquet(METADATA_PATH, index=False)


def run():
    print("📥 Loading processed data...")
    df = load_data()

    texts = df["claim_text"].tolist()
    print(f"✅ Loaded {len(texts)} claims for indexing.")

    print("🧠 Building embeddings...")
    embeddings = build_embeddings(texts, EMBEDDING_MODEL_NAME)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"✅ Index built with {index.ntotal} vectors.")

    print("💾 Saving index and metadata...")
    save_index(index, df)

    print(f"🎉 Done. Index saved at {INDEX_PATH}")
    print(f"📄 Metadata saved at {METADATA_PATH}")


if __name__ == "__main__":
    run()
