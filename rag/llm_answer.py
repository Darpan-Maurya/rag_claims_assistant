import os
from typing import List

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

from core.config import settings

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = settings.llm_model_name



SYSTEM_INSTRUCTION = """
You are an AI assistant helping insurance payer staff analyze claims data.

Rules:
- Use ONLY the provided claims data.
- Do NOT hallucinate.
- Provide counts, trends, and key reasons clearly.
- If the data is insufficient, say so.
- Keep the answer concise and professional.
- Cite claim IDs used as evidence.
- Do not provide medical, legal, or guaranteed coverage decisions.
"""


def _value(row: pd.Series, column: str, default: str = "N/A"):
    value = row[column] if column in row and pd.notna(row[column]) else default
    return value if value != "" else default


def build_context_from_claims(df: pd.DataFrame, max_claims: int = 25) -> str:
    if df.empty:
        return "No relevant claims were retrieved."

    lines: List[str] = []
    for _, row in df.head(max_claims).iterrows():
        context_window = _value(row, "context_window", "")
        matched_child = _value(row, "matched_child_text", "")
        lines.append(
            f"Claim ID: {_value(row, 'claim_id')}, "
            f"Disease: {_value(row, 'disease')}, "
            f"Diagnosis Code: {_value(row, 'diagnosis_code')}, "
            f"Speciality: {_value(row, 'speciality')}, "
            f"Procedure Code: {_value(row, 'procedure_code')}, "
            f"Status: {_value(row, 'claim_status')}, "
            f"Denial Reason: {_value(row, 'denial_reason')}, "
            f"Amount: {_value(row, 'claim_amount')} INR, "
            f"Allowed: {_value(row, 'allowed_amount')} INR, "
            f"Paid: {_value(row, 'paid_amount')} INR, "
            f"Payer: {_value(row, 'payer_name')}, "
            f"Network: {_value(row, 'network_status')}, "
            f"Prior Auth: {_value(row, 'prior_authorization_flag')}, "
            f"Service Date: {_value(row, 'service_date')}, "
            f"Retrieval Score: {_value(row, 'retrieval_score')}, "
            f"Matched Child Text: {matched_child}, "
            f"Sentence Window Context: {context_window}"
        )
    return "\n".join(lines)


def answer_query_with_context(user_query: str, retrieved_df: pd.DataFrame) -> str:
    if retrieved_df.empty:
        return "Insufficient evidence: no relevant claims were retrieved."

    if not settings.gemini_api_key:
        return _fallback_evidence_answer(user_query, retrieved_df)

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


def _fallback_evidence_answer(user_query: str, retrieved_df: pd.DataFrame) -> str:
    total = len(retrieved_df)
    status_counts = retrieved_df["claim_status"].value_counts().to_dict()
    claim_ids = retrieved_df["claim_id"].head(5).astype(str).tolist()
    reasons = {}
    if "denial_reason" in retrieved_df.columns:
        reasons = (
            retrieved_df[retrieved_df["claim_status"] == "DENIED"]["denial_reason"]
            .replace("", "N/A")
            .value_counts()
            .head(3)
            .to_dict()
        )
    answer = (
        f"Using retrieved evidence for query '{user_query}', I found {total} matching claims. "
        f"Status counts: "
        + ", ".join(f"{status}: {count}" for status, count in status_counts.items())
        + f". Evidence claim IDs: {', '.join(claim_ids)}."
    )
    if reasons:
        answer += " Top denial reasons: " + ", ".join(
            f"{reason}: {count}" for reason, count in reasons.items()
        ) + "."
    answer += " Gemini is not configured, so this is a deterministic evidence summary."
    return answer
