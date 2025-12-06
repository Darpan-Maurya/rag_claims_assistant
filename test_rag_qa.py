from rag.retriever import ClaimsRetriever
from rag.llm_answer import answer_query_with_context

def main():
    retriever = ClaimsRetriever()

    # Try a few realistic queries
    queries = [
        "Show me denied claims for diabetes patients last quarter",
        "What are the common denial reasons for cardiology claims?",
        "Total claim activity for hypertension patients",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"USER QUERY: {q}")
        print("=" * 80)

        retrieved = retriever.retrieve(q, k=25)
        print(f"Retrieved {len(retrieved)} candidate claims.")

        answer = answer_query_with_context(q, retrieved)
        print("\nASSISTANT ANSWER:\n")
        print(answer)
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main()
