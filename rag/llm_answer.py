import os
from typing import List

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-flash-latest"



SYSTEM_INSTRUCTION = """
You are an AI assistant helping insurance payer staff analyze claims data.

Rules:
- Use ONLY the provided claims data.
- Do NOT hallucinate.
- Provide counts, trends, and key reasons clearly.
- If the data is insufficient, say so.
- Keep the answer concise and professional.
"""

def build_context_from_claims(df: pd.DataFrame, max_claims: int = 25) -> str:
    if df.empty:
        return "No relevant claims were retrieved."

    lines: List[str] = []
    for _, row in df.head(max_claims).iterrows():
        lines.append(
            f"Claim ID: {row['claim_id']}, "
            f"Disease: {row['disease']}, "
            f"Speciality: {row['speciality']}, "
            f"Status: {row['claim_status']}, "
            f"Denial Reason: {row['denial_reason'] or 'N/A'}, "
            f"Amount: {row['claim_amount']} INR, "
            f"Service Date: {row['service_date']}"
        )
    return "\n".join(lines)


def answer_query_with_context(user_query: str, retrieved_df: pd.DataFrame) -> str:
    context = build_context_from_claims(retrieved_df)

    prompt = f"""
{SYSTEM_INSTRUCTION}

User Query:
{user_query}

Relevant Claims Data:
{context}

Answer:
"""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    return response.text.strip()
