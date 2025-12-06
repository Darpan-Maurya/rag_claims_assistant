import pandas as pd
from pathlib import Path

# =====================
# PATHS
# =====================
RAW_DATA_PATH = Path("data/raw/claims.csv")
PROCESSED_DATA_PATH = Path("data/processed/claims_processed.parquet")

# =====================
# EXTRACT
# =====================
def extract() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)

# =====================
# TRANSFORM
# =====================
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # ---- Basic cleaning ----
    df["denial_reason"] = df["denial_reason"].fillna("")

    df["service_date"] = pd.to_datetime(df["service_date"])
    df["submission_date"] = pd.to_datetime(df["submission_date"])

    # ---- Create RAG-friendly text per claim ----
    def build_claim_text(row):
        text = (
            f"Claim {row['claim_id']} involves a patient aged {row['patient_age']} "
            f"with {row['disease']} treated under {row['speciality']} at "
            f"{row['hospital_name']}. "
            f"The claim amount was {row['claim_amount']} INR. "
            f"Claim status is {row['claim_status']}. "
        )

        if row["claim_status"] == "DENIED":
            text += f"The denial reason was {row['denial_reason']}. "

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
