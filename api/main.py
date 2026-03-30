from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.retriever import ClaimsRetriever
from rag.llm_answer import answer_query_with_context

# =====================
# INITIALIZE APP
# =====================
app = FastAPI(
    title="RAG-Powered Claims Assistant",
    description="Query insurance claims using natural language (RAG + LLM)",
    version="1.0.0",
)

# =====================
# LOAD RAG COMPONENTS
# =====================
try:
    retriever = ClaimsRetriever()
except Exception as e:
    raise RuntimeError(f"Failed to initialize retriever: {e}")

# =====================
# REQUEST / RESPONSE SCHEMAS
# =====================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 25


class QueryResponse(BaseModel):
    answer: str


# =====================
# ROUTES
# =====================
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_claims(request: QueryRequest):
    """
    Accepts a natural language query and returns a RAG-generated answer.
    """
    try:
        retrieved_df = retriever.retrieve(
            query=request.query,
            k=request.top_k
        )

        if retrieved_df.empty:
            return QueryResponse(
                answer="No relevant claims were found for the given query."
            )

        answer = answer_query_with_context(
            user_query=request.query,
            retrieved_df=retrieved_df,
        )

        return QueryResponse(answer=answer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
