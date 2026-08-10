import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Dict, Optional

import pandas as pd


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
    negated_claim_status: Optional[str] = None
    disease: Optional[str] = None
    speciality: Optional[str] = None
    payer_name: Optional[str] = None
    denial_reason: Optional[str] = None
    service_date_start: Optional[str] = None
    service_date_end: Optional[str] = None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    network_status: Optional[str] = None
    plan_type: Optional[str] = None
    provider_type: Optional[str] = None
    member_state: Optional[str] = None
    appeal_status: Optional[str] = None
    prior_authorization_flag: Optional[bool] = None

    def to_dict(self) -> Dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def extract_filters(query: str, df: Optional[pd.DataFrame] = None) -> QueryFilters:
    """Extract concrete metadata filters after the model has selected a data path."""

    q = query.lower()
    filters = QueryFilters()

    for status in KNOWN_STATUSES:
        status_l = status.lower()
        if re.search(rf"\bnot\s+{re.escape(status_l)}\b", q):
            filters.negated_claim_status = status
            break
        if status_l in q:
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
            payer_pattern = rf"(?<!\w){re.escape(payer.lower())}(?!\w)"
            if re.search(payer_pattern, q):
                filters.payer_name = payer
                break

    if "out-of-network" in q or "out of network" in q:
        filters.network_status = "OUT_OF_NETWORK"
    elif "in-network" in q or "in network" in q:
        filters.network_status = "IN_NETWORK"

    if df is not None:
        for column in ("plan_type", "provider_type", "member_state", "appeal_status"):
            if column not in df.columns:
                continue
            for value in sorted(df[column].dropna().astype(str).unique(), key=len, reverse=True):
                if value and value.lower().replace("_", " ") in q:
                    setattr(filters, column, value)
                    break

    if any(
        phrase in q
        for phrase in (
            "without prior authorization",
            "without pre-authorization",
            "prior authorization missing",
            "pre-authorization missing",
            "missing authorization",
        )
    ):
        filters.prior_authorization_flag = False
    elif any(phrase in q for phrase in ("with prior authorization", "prior authorized")):
        filters.prior_authorization_flag = True

    minimum_match = re.search(
        r"(?:claim\s+)?amount\s*(?:>=|>)\s*([\d,]+(?:\.\d+)?)"
        r"|(?:over|above|greater than|more than)\s+([\d,]+(?:\.\d+)?)",
        q,
    )
    if minimum_match:
        value = next(group for group in minimum_match.groups() if group is not None)
        filters.amount_min = float(value.replace(",", ""))

    maximum_match = re.search(
        r"(?:claim\s+)?amount\s*(?:<=|<)\s*([\d,]+(?:\.\d+)?)"
        r"|(?:under|below|less than)\s+([\d,]+(?:\.\d+)?)",
        q,
    )
    if maximum_match:
        value = next(group for group in maximum_match.groups() if group is not None)
        filters.amount_max = float(value.replace(",", ""))

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
    if filters.negated_claim_status:
        result = result[result["claim_status"] != filters.negated_claim_status]
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
    if "network_status" in filter_dict:
        result = result[result["network_status"] == filters.network_status]
    if "plan_type" in filter_dict:
        result = result[result["plan_type"] == filters.plan_type]
    if "provider_type" in filter_dict:
        result = result[result["provider_type"] == filters.provider_type]
    if "member_state" in filter_dict:
        result = result[result["member_state"] == filters.member_state]
    if "appeal_status" in filter_dict:
        result = result[result["appeal_status"] == filters.appeal_status]
    if "prior_authorization_flag" in filter_dict:
        normalized = (
            result["prior_authorization_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        result = result[normalized == bool(filters.prior_authorization_flag)]
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
