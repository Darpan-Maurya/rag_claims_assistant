from pathlib import Path
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.observability import timed_metric
from orchestrate.router import QueryFilters, apply_filters, extract_filters
from rag.corrective import evaluate_retrieval_quality
from rag.graph_context import ClaimsGraphContext
from rag.query_expansion import expand_query
from rag.vector_backends import QdrantVectorBackend

BASE_DIR = Path(__file__).resolve().parent  # points to rag/ folder

METADATA_PATH = settings.metadata_path
EMBEDDING_MODEL_NAME = settings.embedding_model_name
CHUNK_METADATA_PATH = settings.chunk_metadata_path


@dataclass
class RetrievalDetails:
    route: str
    filters: Dict[str, object]
    retrieval_summary: Dict[str, object]
    results: pd.DataFrame


class ClaimsRetriever:
    """
    Hybrid retriever using metadata filters, Qdrant dense vector search,
    lexical BM25, reciprocal-rank fusion, and an optional cross-encoder reranker.
    """

    def __init__(self):
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
        if not CHUNK_METADATA_PATH.exists():
            raise FileNotFoundError(f"Chunk metadata not found at {CHUNK_METADATA_PATH}")

        self.vector_backend = QdrantVectorBackend()
        self.df = pd.read_parquet(METADATA_PATH)
        self.graph_context = ClaimsGraphContext(self.df)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chunk_df = self._load_chunk_metadata()
        self.uses_child_chunks = len(self.chunk_df) == self.vector_backend.size
        self.documents = self.chunk_df["child_text"].fillna("").astype(str).tolist()
        self._doc_tokens = [self._tokenize(text) for text in self.documents]
        self._doc_freq = self._build_doc_freq(self._doc_tokens)
        self._avg_doc_len = (
            sum(len(tokens) for tokens in self._doc_tokens) / max(1, len(self._doc_tokens))
        )
        self._reranker = None
        if settings.enable_reranker:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.reranker_model_name)

    def retrieve(self, query: str, k: int = 10) -> pd.DataFrame:
        """
        Returns a dataframe of top-k matching claims with a similarity score column.
        """
        return self.retrieve_with_details(query=query, k=k).results

    def retrieve_with_details(
        self,
        query: str,
        k: int = 10,
        filters: Optional[QueryFilters] = None,
        allowed_parent_indices: Optional[set[int]] = None,
    ) -> RetrievalDetails:
        filters = filters or extract_filters(query, self.df)
        return self._retrieve_internal(
            query=query,
            k=k,
            filters=filters,
            allow_correction=True,
            allowed_parent_indices=allowed_parent_indices,
        )

    def _retrieve_internal(
        self,
        query: str,
        k: int,
        filters: QueryFilters,
        allow_correction: bool,
        allowed_parent_indices: Optional[set[int]] = None,
    ) -> RetrievalDetails:
        filtered_df = apply_filters(self.df, filters)
        parent_indices = set(filtered_df.index.tolist())
        if allowed_parent_indices is not None:
            parent_indices = parent_indices.intersection(allowed_parent_indices)
        candidate_indices = self._candidate_chunk_indices(parent_indices)
        if not candidate_indices:
            empty = self.df.iloc[0:0].copy()
            return RetrievalDetails(
                route="RAG",
                filters=filters.to_dict(),
                retrieval_summary={
                    "candidate_count": 0,
                    "returned_count": 0,
                    "strategy": "metadata_filtered_hybrid",
                    "warning": "No rows matched metadata filters.",
                },
                results=empty,
            )

        query_variants = [query] + expand_query(query)
        with timed_metric("dense_retrieval"):
            dense_ranked = self._dense_search(query_variants, candidate_indices, k)
        with timed_metric("lexical_retrieval"):
            lexical_ranked = self._lexical_search(query_variants, candidate_indices, max(k * 4, 30))

        fused_scores = self._reciprocal_rank_fusion([dense_ranked, lexical_ranked])
        fused_indices = [
            index for index, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        ][: max(k * 4, k)]

        rerank_scores: Dict[int, float] = {}
        if self._reranker is not None and fused_indices:
            with timed_metric("rerank_retrieval"):
                pairs = [(query, self.documents[index]) for index in fused_indices]
                scores = self._reranker.predict(pairs)
                rerank_scores = {
                    index: float(score) for index, score in zip(fused_indices, scores)
                }
                fused_indices = sorted(
                    fused_indices,
                    key=lambda index: rerank_scores.get(index, 0.0),
                    reverse=True,
                )

        selected = fused_indices[: max(k * 4, k)]
        dense_scores = dict(dense_ranked)
        lexical_scores = dict(lexical_ranked)
        results = self._materialize_parent_results(
            selected,
            k=k,
            dense_scores=dense_scores,
            lexical_scores=lexical_scores,
            fused_scores=fused_scores,
            rerank_scores=rerank_scores,
        )

        summary = {
            "strategy": "parent_child_sentence_window_hybrid_rrf",
            "reranker_enabled": self._reranker is not None,
            "child_chunks_enabled": self.uses_child_chunks,
            "query_expansion_enabled": len(query_variants) > 1,
            "query_expansions": query_variants[1:],
            "candidate_count": int(len(parent_indices)),
            "candidate_chunk_count": int(len(candidate_indices)),
            "dense_candidates": int(len(dense_ranked)),
            "lexical_candidates": int(len(lexical_ranked)),
            "returned_count": int(len(results)),
            "top_score": float(results["retrieval_score"].max()) if not results.empty else 0.0,
        }
        parent_indices_for_graph = [int(index) for index in results.index.tolist()]
        summary["graph_context"] = self.graph_context.related_claims(parent_indices_for_graph)
        quality = evaluate_retrieval_quality(summary)
        summary["crag_action"] = quality.action
        summary["crag_confidence"] = round(quality.confidence, 3)
        summary["crag_reason"] = quality.reason

        if allow_correction and quality.action in {"ambiguous", "incorrect"} and filters.to_dict():
            relaxed_filters = QueryFilters()
            relaxed = self._retrieve_internal(
                query=query,
                k=k,
                filters=relaxed_filters,
                allow_correction=False,
                allowed_parent_indices=allowed_parent_indices,
            )
            if relaxed.results.empty:
                summary["corrective_retry"] = "no_improvement"
            elif relaxed.retrieval_summary.get("top_score", 0.0) > summary["top_score"]:
                relaxed.retrieval_summary["corrective_retry"] = "used_relaxed_filters"
                relaxed.retrieval_summary["original_filters"] = filters.to_dict()
                return relaxed

        return RetrievalDetails(
            route="RAG",
            filters=filters.to_dict(),
            retrieval_summary=summary,
            results=results,
        )

    def readiness(self) -> Dict[str, object]:
        return {
            "vector_backend": self.vector_backend.readiness(),
            "metadata_path": str(METADATA_PATH),
            "chunk_metadata_path": str(CHUNK_METADATA_PATH),
            "index_vectors": int(self.vector_backend.size),
            "metadata_rows": int(len(self.df)),
            "chunk_metadata_rows": int(len(self.chunk_df)),
            "child_chunks_enabled": self.uses_child_chunks,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_enabled": self._reranker is not None,
            "ready": bool(
                self.vector_backend.size == len(self.chunk_df)
                and len(self.df) > 0
                and len(self.chunk_df) > 0
            ),
        }

    def _dense_search(
        self, query_variants: List[str], candidate_indices: set[int], k: int
    ) -> List[tuple[int, float]]:
        q_emb = self.model.encode(query_variants, normalize_embeddings=True)
        search_k = self.vector_backend.size if len(candidate_indices) < len(self.chunk_df) else min(
            self.vector_backend.size, max(k * 8, 100)
        )

        best_scores: Dict[int, float] = {}
        for embedding in q_emb:
            for hit in self.vector_backend.search(embedding, search_k):
                if hit.point_id in candidate_indices:
                    best_scores[hit.point_id] = max(
                        best_scores.get(hit.point_id, 0.0),
                        hit.score,
                    )
        return sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[: max(k * 4, k)]

    def _lexical_search(
        self, query_variants: List[str], candidate_indices: set[int], limit: int
    ) -> List[tuple[int, float]]:
        variant_tokens = [self._tokenize(query) for query in query_variants]
        scores: List[tuple[int, float]] = []
        for index in candidate_indices:
            score = max(
                self._bm25_score(query_tokens, self._doc_tokens[index])
                for query_tokens in variant_tokens
            )
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]

    def _bm25_score(self, query_tokens: Iterable[str], doc_tokens: List[str]) -> float:
        if not doc_tokens:
            return 0.0
        token_counts = Counter(doc_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        doc_len = len(doc_tokens)
        for token in set(query_tokens):
            if token not in token_counts:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log(1 + (len(self._doc_tokens) - df + 0.5) / (df + 0.5))
            freq = token_counts[token]
            denom = freq + k1 * (1 - b + b * doc_len / max(1, self._avg_doc_len))
            score += idf * ((freq * (k1 + 1)) / denom)
        return score

    def _reciprocal_rank_fusion(
        self, ranked_lists: List[List[tuple[int, float]]], rank_constant: int = 60
    ) -> Dict[int, float]:
        fused: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (index, _) in enumerate(ranked, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (rank_constant + rank)
        return fused

    def _build_doc_freq(self, tokenized_docs: List[List[str]]) -> Dict[str, int]:
        doc_freq: Dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        return doc_freq

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _load_chunk_metadata(self) -> pd.DataFrame:
        chunks = pd.read_parquet(CHUNK_METADATA_PATH)
        if len(chunks) != self.vector_backend.size:
            raise RuntimeError(
                "Qdrant collection and chunk metadata are out of sync. "
                "Run `python3 -m rag.build_index` to rebuild ingestion artifacts."
            )
        return chunks

    def _candidate_chunk_indices(self, parent_indices: set[int]) -> set[int]:
        if self.uses_child_chunks:
            return set(
                self.chunk_df.index[
                    self.chunk_df["parent_row_index"].astype(int).isin(parent_indices)
                ].tolist()
            )
        return parent_indices

    def _materialize_parent_results(
        self,
        selected_chunk_indices: List[int],
        k: int,
        dense_scores: Dict[int, float],
        lexical_scores: Dict[int, float],
        fused_scores: Dict[int, float],
        rerank_scores: Dict[int, float],
    ) -> pd.DataFrame:
        best_by_parent: Dict[int, Dict[str, object]] = {}
        for chunk_index in selected_chunk_indices:
            chunk = self.chunk_df.iloc[chunk_index]
            parent_index = int(chunk["parent_row_index"])
            score = rerank_scores.get(chunk_index, fused_scores.get(chunk_index, 0.0))
            current = best_by_parent.get(parent_index)
            if current is None or float(score) > float(current["sort_score"]):
                best_by_parent[parent_index] = {
                    "chunk_index": chunk_index,
                    "sort_score": score,
                    "matched_child_text": chunk["child_text"],
                    "context_window": chunk["window_text"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk["chunk_type"],
                    "dense_score": dense_scores.get(chunk_index, 0.0),
                    "lexical_score": lexical_scores.get(chunk_index, 0.0),
                    "retrieval_score": fused_scores.get(chunk_index, 0.0),
                    "rerank_score": rerank_scores.get(chunk_index),
                }

        ordered = sorted(best_by_parent.items(), key=lambda item: item[1]["sort_score"], reverse=True)
        parent_indices = [parent_index for parent_index, _ in ordered[:k]]
        results = self.df.iloc[parent_indices].copy()
        for column in [
            "matched_child_text",
            "context_window",
            "chunk_id",
            "chunk_type",
            "dense_score",
            "lexical_score",
            "retrieval_score",
            "rerank_score",
        ]:
            results[column] = [best_by_parent[parent_index][column] for parent_index in parent_indices]
        results["similarity_score"] = results["dense_score"]
        return results
