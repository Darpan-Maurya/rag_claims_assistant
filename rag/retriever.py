from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("rag/vector_store.faiss")
METADATA_PATH = Path("rag/metadata.parquet")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ClaimsRetriever:
    """
    Wrapper around FAISS + metadata to retrieve relevant claims for a query.
    """

    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Index not found at {INDEX_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

        self.index = faiss.read_index(str(INDEX_PATH))
        self.df = pd.read_parquet(METADATA_PATH)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, k: int = 10) -> pd.DataFrame:
        """
        Returns a dataframe of top-k matching claims with a similarity score column.
        """
        q_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_emb.astype("float32"), k)

        indices = I[0]
        scores = D[0]

        results = self.df.iloc[indices].copy()
        results["similarity_score"] = scores
        return results
