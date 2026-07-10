from rag.retriever import ClaimsRetriever


def main():
    print("📥 Loading Qdrant-backed retriever...")
    retriever = ClaimsRetriever()
    readiness = retriever.readiness()
    print(f"Vector backend: {readiness['vector_backend']}")
    print(f"Metadata rows: {readiness['metadata_rows']}")

    # Example dev query
    query = "denied claims for diabetes patients"
    print(f"\n🔎 Query: {query}")

    details = retriever.retrieve_with_details(query, k=5)

    print("\n🔝 Top 5 matches:")
    for rank, (_, row) in enumerate(details.results.iterrows(), start=1):
        print(f"\n#{rank} | score={row['retrieval_score']:.4f} | claim_id={row['claim_id']}")
        print(f"   disease={row['disease']}, status={row['claim_status']}")
        print(f"   denial_reason={row['denial_reason']}")
        print(f"   matched={row.get('matched_child_text', '')}")


if __name__ == "__main__":
    main()
