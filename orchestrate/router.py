import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Dict, Optional

import pandas as pd


ANALYTICS_KEYWORDS = [
    "percentage",
    "percent",
    "rate",
    "count",
    "how many",
    "total",
    "average",
    "avg",
    "sum",
    "trend",
    "breakdown",
    "common",
    "top",
]

RAG_KEYWORDS = [
    "claims",
    "denied",
    "approved",
    "pending",
    "last quarter",
    "reasons",
    "show me",
    "why",
    "examples",
    "evidence",
]

DECISION_KEYWORDS = [
    "will my claim",
    "should i",
    "if i claim",
    "guarantee",
]

KNOWN_DISEASES = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "COPD",
    "Coronary Artery Disease",
    "Covid-19",
]

KNOWN_SPECIALITIES = [
    "Endocrinology",
    "Cardiology",
    "Pulmonology",
    "General Medicine",
]

KNOWN_STATUSES = ["APPROVED", "DENIED", "PENDING"]

KNOWN_DENIAL_REASONS = [
    "Insufficient documentation",
    "Not medically necessary",
    "Pre-authorization missing",
    "Coverage limit exceeded",
    "Out-of-network provider",
]


@dataclass
class QueryFilters:
    claim_status: Optional[str] = None
    disease: Optional[str] = None
    speciality: Optional[str] = None
    payer_name: Optional[str] = None
    denial_reason: Optional[str] = None
    service_date_start: Optional[str] = None
    service_date_end: Optional[str] = None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def classify_query(query: str) -> str:
    q = query.lower()

    if any(k in q for k in DECISION_KEYWORDS):
        return "DECISION_HELP"

    if any(k in q for k in ANALYTICS_KEYWORDS):
        return "ANALYTICS"

    if any(k in q for k in RAG_KEYWORDS):
        return "RAG"

    return "RAG"


def extract_filters(query: str, df: Optional[pd.DataFrame] = None) -> QueryFilters:
    q = query.lower()
    filters = QueryFilters()

    for status in KNOWN_STATUSES:
        if status.lower() in q or status.lower().rstrip("ed") in q:
            filters.claim_status = status
            break

    for disease in KNOWN_DISEASES:
        if disease.lower() in q:
            filters.disease = disease
            break

    for speciality in KNOWN_SPECIALITIES:
        if speciality.lower() in q:
            filters.speciality = speciality
            break

    for reason in KNOWN_DENIAL_REASONS:
        if reason.lower() in q:
            filters.denial_reason = reason
            break

    if df is not None and "payer_name" in df.columns:
        for payer in sorted(df["payer_name"].dropna().astype(str).unique(), key=len, reverse=True):
            if payer.lower() in q:
                filters.payer_name = payer
                break

    amount_match = re.search(r"(?:over|above|greater than|more than)\s+([\d,]+(?:\.\d+)?)", q)
    if amount_match:
        filters.amount_min = float(amount_match.group(1).replace(",", ""))

    amount_match = re.search(r"(?:under|below|less than)\s+([\d,]+(?:\.\d+)?)", q)
    if amount_match:
        filters.amount_max = float(amount_match.group(1).replace(",", ""))

    if "last quarter" in q:
        reference = _reference_date(df)
        quarter_start, quarter_end = _previous_quarter(reference)
        filters.service_date_start = quarter_start.isoformat()
        filters.service_date_end = quarter_end.isoformat()

    return filters


def apply_filters(df: pd.DataFrame, filters: QueryFilters) -> pd.DataFrame:
    result = df
    filter_dict = filters.to_dict()
    if "claim_status" in filter_dict:
        result = result[result["claim_status"] == filters.claim_status]
    if "disease" in filter_dict:
        result = result[result["disease"] == filters.disease]
    if "speciality" in filter_dict:
        result = result[result["speciality"] == filters.speciality]
    if "payer_name" in filter_dict:
        result = result[result["payer_name"] == filters.payer_name]
    if "denial_reason" in filter_dict:
        result = result[result["denial_reason"] == filters.denial_reason]
    if "service_date_start" in filter_dict:
        result = result[pd.to_datetime(result["service_date"]) >= pd.to_datetime(filters.service_date_start)]
    if "service_date_end" in filter_dict:
        result = result[pd.to_datetime(result["service_date"]) <= pd.to_datetime(filters.service_date_end)]
    if "amount_min" in filter_dict:
        result = result[result["claim_amount"] >= filters.amount_min]
    if "amount_max" in filter_dict:
        result = result[result["claim_amount"] <= filters.amount_max]
    return result


def _reference_date(df: Optional[pd.DataFrame]) -> date:
    if df is not None and "service_date" in df.columns and not df.empty:
        return pd.to_datetime(df["service_date"]).max().date()
    return date.today()


def _previous_quarter(reference: date) -> tuple[date, date]:
    current_quarter = (reference.month - 1) // 3 + 1
    previous_quarter = current_quarter - 1
    year = reference.year
    if previous_quarter == 0:
        previous_quarter = 4
        year -= 1
    start_month = (previous_quarter - 1) * 3 + 1
    end_month = start_month + 2
    start = date(year, start_month, 1)
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, end_month + 1, 1) - timedelta(days=1)
    return start, end
