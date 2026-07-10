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
    top_score = float(summary.get("top_score", 0.0) or 0.0)
    dense_candidates = int(summary.get("dense_candidates", 0) or 0)
    lexical_candidates = int(summary.get("lexical_candidates", 0) or 0)

    if candidate_count == 0 or returned == 0:
        return RetrievalQuality(
            action="incorrect",
            confidence=0.0,
            reason="No candidate evidence matched the query and filters.",
        )

    confidence = min(1.0, top_score * 12)
    if dense_candidates > 0 and lexical_candidates > 0:
        confidence = min(1.0, confidence + 0.2)

    if confidence >= 0.45:
        return RetrievalQuality("correct", confidence, "Dense and/or lexical evidence looks usable.")
    if confidence >= 0.2:
        return RetrievalQuality("ambiguous", confidence, "Evidence exists but relevance is weak.")
    return RetrievalQuality("incorrect", confidence, "Evidence relevance is too weak.")
