import json
from pathlib import Path

from core.query_classifier import QueryClassifier
from orchestrate.filters import extract_filters
from orchestrate.router import plan_query
from rag.retriever import ClaimsRetriever


GOLDEN_PATH = Path("eval/golden_queries.jsonl")


def load_golden():
    return [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()]


def main():
    retriever = ClaimsRetriever()
    classifier = QueryClassifier()
    rows = load_golden()
    route_hits = 0
    filter_hits = 0
    retrieval_nonempty = 0

    for row in rows:
        query = row["query"]
        route = plan_query(classifier.classify(query)).route
        filters = extract_filters(query, retriever.df).to_dict()
        route_hits += int(route == row["expected_route"])
        filter_hits += int(
            all(filters.get(key) == value for key, value in row.get("expected_filters", {}).items())
        )
        if route == "RAG":
            details = retriever.retrieve_with_details(query, k=10)
            retrieval_nonempty += int(not details.results.empty)

    print(
        json.dumps(
            {
                "queries": len(rows),
                "route_accuracy": route_hits / len(rows),
                "filter_accuracy": filter_hits / len(rows),
                "rag_nonempty_queries": retrieval_nonempty,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
