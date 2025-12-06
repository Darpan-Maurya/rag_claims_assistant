from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("rag/vector_store.faiss")
METADATA_PATH = Path("rag/metadata.parquet")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

    print("📥 Loading index and metadata...")
    index = faiss.read_index(str(INDEX_PATH))
    df = pd.read_parquet(METADATA_PATH)

    print(f"Index vectors: {index.ntotal}, metadata rows: {len(df)}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Example dev query
    query = "denied claims for diabetes patients"
    print(f"\n🔎 Query: {query}")

    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype("float32"), k=5)

    print("\n🔝 Top 5 matches:")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        row = df.iloc[idx]
        print(f"\n#{rank} | score={score:.4f} | claim_id={row['claim_id']}")
        print(f"   disease={row['disease']}, status={row['claim_status']}")
        print(f"   denial_reason={row['denial_reason']}")


if __name__ == "__main__":
    main()
