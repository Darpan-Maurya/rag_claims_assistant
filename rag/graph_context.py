from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


GRAPH_RELATION_FIELDS = [
    "disease",
    "speciality",
    "payer_name",
    "denial_reason",
    "network_status",
    "plan_type",
    "provider_type",
]


class ClaimsGraphContext:
    """
    Lightweight structured GraphRAG layer for claims.

    This is not a full LLM-extracted knowledge graph. It builds explainable
    claim-to-claim relationships from structured fields and is appropriate for
    this tabular claims domain.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self._field_value_to_claims: Dict[str, Dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        for row_index, row in self.df.iterrows():
            for field in GRAPH_RELATION_FIELDS:
                if field not in row or pd.isna(row[field]) or row[field] == "":
                    continue
                self._field_value_to_claims[field][str(row[field])].add(int(row_index))

    def related_claims(self, parent_indices: List[int], limit: int = 10) -> List[Dict[str, object]]:
        scores: Counter[int] = Counter()
        reasons: Dict[int, List[str]] = defaultdict(list)

        for parent_index in parent_indices:
            if parent_index < 0 or parent_index >= len(self.df):
                continue
            row = self.df.iloc[parent_index]
            for field in GRAPH_RELATION_FIELDS:
                value = row[field] if field in row and pd.notna(row[field]) else None
                if value is None or value == "":
                    continue
                for related_index in self._field_value_to_claims[field].get(str(value), set()):
                    if related_index == parent_index:
                        continue
                    scores[related_index] += 1
                    if len(reasons[related_index]) < 4:
                        reasons[related_index].append(f"{field}={value}")

        related = []
        for related_index, score in scores.most_common(limit):
            row = self.df.iloc[related_index]
            related.append(
                {
                    "claim_id": row["claim_id"],
                    "relationship_score": int(score),
                    "relationship_reasons": reasons[related_index],
                    "claim_status": row.get("claim_status"),
                    "disease": row.get("disease"),
                    "denial_reason": row.get("denial_reason"),
                }
            )
        return related
