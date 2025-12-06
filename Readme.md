RAG-Powered Insurance Claims Query Assistant
Overview

The RAG-Powered Insurance Claims Query Assistant is an AI-driven system that enables insurance payer staff to query structured claims data using natural language.
Instead of writing SQL or manually analyzing spreadsheets, users can ask questions such as:

“Show me denied claims for diabetes patients last quarter”

“What are the common denial reasons for cardiology claims?”

“What percentage of claims are approved?”

The system combines ETL pipelines, vector-based semantic retrieval (RAG), and Large Language Models (LLMs) to generate accurate, explainable, and data-grounded responses.

Key Features

Natural language querying over structured insurance claims data

Retrieval-Augmented Generation (RAG) using vector similarity search

Evidence-grounded answers with denial reasons, counts, and trends

FastAPI-based backend microservice

Streamlit-based interactive chatbot UI

Fully containerized using Docker and Docker Compose

Tech Stack

Language: Python

Backend: FastAPI, Uvicorn

LLM: Google Gemini

Vector Search: FAISS

Embeddings: Sentence-Transformers

Data Processing: Pandas, NumPy

UI: Streamlit

Deployment: Docker, Docker Compose

System Architecture (High Level)
User (Streamlit UI)
        ↓ HTTP
FastAPI Microservice (/query)
        ↓
FAISS Vector Retrieval (Top-K)
        ↓
LLM (Gemini) + Retrieved Context
        ↓
Natural Language Answer

Dataset Creation
Mock Data Design

Synthetic insurance claims data was generated to simulate real-world payer datasets.
Each record includes:

claim_id

patient_age

disease (e.g., Diabetes, Asthma, Hypertension)

speciality (e.g., Cardiology, Endocrinology)

claim_amount

claim_status (APPROVED / DENIED)

denial_reason (if denied)

service_date, submission_date

hospital_name, payer_name

Dataset Size

1,000–5,000 rows (configurable)

Contains a realistic mix of approved and denied claims

Multiple denial patterns (pre-authorization, coverage limits, documentation issues)

ETL Pipeline

The ETL process is an offline preprocessing step that prepares data for retrieval and analytics.

Extract

Raw claims loaded from CSV files using Pandas.

Transform

Missing values standardized

Date fields normalized

Structured claims converted into LLM-friendly narrative text, e.g.:

"Claim CLM0123 involves a patient with Diabetes treated under Endocrinology.
The claim amount was 45,000 INR and the claim was DENIED due to
pre-authorization missing."


This transformation significantly improves semantic retrieval quality.

Load

Processed data stored as Parquet files

Acts as input for:

FAISS index building

Analytics computations

Runtime retrieval

Embedding & Vector Indexing (RAG)

Narrative claim texts are converted into dense embeddings using Sentence-Transformers.

Embeddings are stored in a FAISS vector index for fast similarity search.

Claim metadata is stored alongside vectors for reconstruction after retrieval.

This enables semantic matching between user queries and relevant claims, even when exact keywords do not match.

Retrieval-Augmented Generation (RAG)

At query time:

User query is converted into an embedding.

FAISS retrieves the top-K most relevant claims.

Retrieved claims are formatted into a context window.

The LLM (Gemini) processes the query only using retrieved evidence.

The answer is generated with counts, summaries, and explanations.

This design:

Reduces hallucinations

Improves factual accuracy

Keeps responses auditable and explainable

FastAPI Backend

The backend is implemented as a stateless FastAPI microservice.

Endpoint
POST /query

Request
{
  "query": "Show me denied claims for diabetes patients last quarter",
  "top_k": 25
}

Response
{
  "answer": "Based on the provided claims data..."
}


The backend handles:

Request validation (Pydantic)

Vector-based retrieval

Context construction

LLM invocation

Error handling

Streamlit Chatbot UI

A lightweight Streamlit-based chat UI is used for user interaction.

Enables conversational querying

Displays answers in a chat format

Communicates with FastAPI via REST calls

The UI is intentionally decoupled from backend logic, enabling future replacement with a web or mobile frontend.

Deployment with Docker

The system is deployed as two independent containers:

FastAPI RAG Service

Streamlit UI

Run the system
docker-compose up --build

Access

Chat UI: http://localhost:8501

API Docs: http://localhost:8000/docs

This setup ensures:

Reproducible environment

Clear service separation

Easy portability to cloud platforms

Sample Queries

“Show denied claims for diabetes patients last quarter”

“What are the common denial reasons in cardiology?”

“Total claim activity for hypertension patients”

“Percentage of claims that are approved”

Design Principles

Separation of concerns: ETL, retrieval, inference, and UI are independent

Evidence-based AI: LLM responses are grounded in retrieved claims only

Scalable architecture: Can upgrade FAISS to a vector database without redesign

Microservice-first approach: Backend usable by any client

Future Scope

The system can be further enhanced with intelligent query routing and orchestration to improve performance and accuracy.

Intelligent Query Routing

Introduce a query router (router.py) to classify user queries into:

Analytics queries (counts, percentages)

RAG-based queries (claims, denial reasons)

LLM-only queries (conceptual questions)

Route analytics-heavy queries directly to Pandas/SQL logic instead of FAISS.

Analytics Optimization

Add an analytics module (claims_analytics.py) to compute:

Approval rates

Denial distributions

Time-based trends

This avoids unnecessary vector searches and reduces LLM context size.

Other Enhancements

Replace FAISS with a distributed vector database (Qdrant / Pinecone)

Add Redis caching for repeated queries

Integrate role-based access control (RBAC)

Support live data refresh and incremental ETL

Add evaluation metrics for retrieval quality