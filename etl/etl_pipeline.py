import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import settings

# =====================
# PATHS
# =====================
RAW_DATA_PATH = settings.raw_data_path
PROCESSED_DATA_PATH = settings.processed_data_path
MANIFEST_PATH = settings.dataset_manifest_path

REQUIRED_COLUMNS = {
    "claim_id",
    "patient_id",
    "patient_age",
    "patient_gender",
    "disease",
    "speciality",
    "doctor_id",
    "hospital_name",
    "claim_amount",
    "claim_status",
    "denial_reason",
    "service_date",
    "submission_date",
    "payer_name",
}

OPTIONAL_DEFAULTS = {
    "diagnosis_code": "UNKNOWN",
    "procedure_code": "UNKNOWN",
    "plan_type": "UNKNOWN",
    "member_state": "UNKNOWN",
    "provider_type": "UNKNOWN",
    "allowed_amount": 0.0,
    "paid_amount": 0.0,
    "deductible": 0.0,
    "copay": 0.0,
    "network_status": "UNKNOWN",
    "prior_authorization_flag": False,
    "appeal_status": "NOT_APPEALED",
}

VALID_STATUSES = {"APPROVED", "DENIED", "PENDING"}
VALID_DENIAL_REASONS = {
    "",
    "Insufficient documentation",
    "Not medically necessary",
    "Pre-authorization missing",
    "Coverage limit exceeded",
    "Out-of-network provider",
}

# =====================
# EXTRACT
# =====================
def extract() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["claim_id"].duplicated().any():
        dupes = df.loc[df["claim_id"].duplicated(), "claim_id"].head(5).tolist()
        raise ValueError(f"Duplicate claim_id values found: {dupes}")

    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    df["claim_status"] = df["claim_status"].fillna("").astype(str).str.upper()
    invalid_statuses = sorted(set(df["claim_status"]) - VALID_STATUSES)
    if invalid_statuses:
        raise ValueError(f"Invalid claim_status values: {invalid_statuses}")

    df["denial_reason"] = df["denial_reason"].fillna("").astype(str)
    invalid_reasons = sorted(set(df["denial_reason"]) - VALID_DENIAL_REASONS)
    if invalid_reasons:
        raise ValueError(f"Invalid denial_reason values: {invalid_reasons}")

    numeric_columns = [
        "patient_age",
        "claim_amount",
        "allowed_amount",
        "paid_amount",
        "deductible",
        "copay",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Column {column} contains non-numeric values")

    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")
    if df["service_date"].isna().any() or df["submission_date"].isna().any():
        raise ValueError("service_date and submission_date must be valid dates")

    return df

# =====================
# TRANSFORM
# =====================
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_input(df.copy())
    # ---- Basic cleaning ----
    df["denial_reason"] = df["denial_reason"].fillna("")

    # ---- Create RAG-friendly text per claim ----
    def build_claim_text(row):
        text = (
            f"Claim {row['claim_id']} involves a patient aged {row['patient_age']} "
            f"with {row['disease']} diagnosis code {row['diagnosis_code']} treated under "
            f"{row['speciality']} using procedure code {row['procedure_code']} at "
            f"{row['hospital_name']} ({row['provider_type']}). "
            f"The member state is {row['member_state']} and plan type is {row['plan_type']}. "
            f"The claim amount was {row['claim_amount']} INR, allowed amount was "
            f"{row['allowed_amount']} INR, and paid amount was {row['paid_amount']} INR. "
            f"Network status is {row['network_status']} and prior authorization flag is "
            f"{row['prior_authorization_flag']}. "
            f"Claim status is {row['claim_status']}. "
        )

        if row["claim_status"] == "DENIED":
            text += (
                f"The denial reason was {row['denial_reason']} and appeal status is "
                f"{row['appeal_status']}. "
            )

        text += (
            f"The service was provided on {row['service_date'].date()} "
            f"and submitted on {row['submission_date'].date()} "
            f"to {row['payer_name']}."
        )
        return text

    df["claim_text"] = df.apply(build_claim_text, axis=1)

    return df

# =====================
# LOAD
# =====================
def load(df: pd.DataFrame):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    write_manifest(df)


def write_manifest(df: pd.DataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_name": "synthetic_claims",
        "schema_version": "2.0",
        "row_count": int(len(df)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phi_status": "synthetic_no_phi",
        "date_range": {
            "service_date_min": str(df["service_date"].min().date()),
            "service_date_max": str(df["service_date"].max().date()),
        },
        "columns": list(df.columns),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# =====================
# MAIN PIPELINE
# =====================
def run_etl():
    df = extract()
    df = transform(df)
    load(df)
    print(f"✅ ETL completed. Processed file saved at {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    run_etl()
