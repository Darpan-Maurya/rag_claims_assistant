from typing import Dict, List


DOMAIN_EXPANSIONS: Dict[str, List[str]] = {
    "denied": ["rejected", "denial", "not approved"],
    "approved": ["paid", "accepted", "authorized"],
    "pending": ["under review", "in process"],
    "pre-authorization": ["prior authorization", "preauthorization", "auth missing"],
    "out-of-network": ["out of network", "non network provider"],
    "documentation": ["records", "clinical notes", "supporting documents"],
    "diabetes": ["diabetic", "endocrinology", "E11"],
    "hypertension": ["high blood pressure", "I10"],
    "cardiology": ["cardiac", "heart"],
    "pulmonology": ["respiratory", "lung"],
    "claim amount": ["billed amount", "allowed amount", "paid amount"],
}


def expand_query(query: str, max_expansions: int = 4) -> List[str]:
    lowered = query.lower()
    expansions: List[str] = []
    for trigger, alternatives in DOMAIN_EXPANSIONS.items():
        if trigger in lowered:
            for alternative in alternatives:
                expanded = f"{query} {alternative}"
                if expanded not in expansions:
                    expansions.append(expanded)
                if len(expansions) >= max_expansions:
                    return expansions
    return expansions
