from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RetrievalQuality:
    action: str
    confidence: float
    reason: str


def evaluate_retrieval_quality(summary: Dict[str, object]) -> RetrievalQuality:
    returned = int(summary.get("returned_count", 0) or 0)
    candidate_count = int(summary.get("candidate_count", 0) or 0)
    top_dense_score = float(summary.get("top_dense_score", 0.0) or 0.0)
    dense_candidates = int(summary.get("dense_candidates", 0) or 0)
    lexical_candidates = int(summary.get("lexical_candidates", 0) or 0)

    if candidate_count == 0 or returned == 0:
        return RetrievalQuality(
            action="incorrect",
            confidence=0.0,
            reason="No candidate evidence matched the query and filters.",
        )

    # Dense cosine similarity is a usable confidence proxy. RRF is deliberately
    # excluded because it is only a relative rank and has a tiny score range.
    confidence = max(0.0, min(1.0, (top_dense_score + 1.0) / 2.0))
    if dense_candidates > 0 and lexical_candidates > 0:
        confidence = min(1.0, confidence + 0.2)

    if confidence >= 0.45:
        return RetrievalQuality("correct", confidence, "Dense and/or lexical evidence looks usable.")
    if confidence >= 0.2:
        return RetrievalQuality("ambiguous", confidence, "Evidence exists but relevance is weak.")
    return RetrievalQuality("incorrect", confidence, "Evidence relevance is too weak.")
