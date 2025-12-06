def classify_query(query: str) -> str:
    q = query.lower()

    analytics_keywords = [
        "percentage", "percent"
    ]

    rag_keywords = [
        "claims", "denied", "approved",
        "last quarter", "reasons", "show me"
    ]

    decision_keywords = [
        "will my claim", "should i", "if i claim"
    ]

    if any(k in q for k in decision_keywords):
        return "DECISION_HELP"

    if any(k in q for k in analytics_keywords):
        return "ANALYTICS"

    if any(k in q for k in rag_keywords):
        return "RAG"

    return "LLM_ONLY"
