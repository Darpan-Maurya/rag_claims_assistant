import pandas as pd
import pytest

from etl.etl_pipeline import transform


def base_claims_df():
    return pd.DataFrame(
        {
            "claim_id": ["CLM00001"],
            "patient_id": ["PAT0001"],
            "patient_age": [45],
            "patient_gender": ["F"],
            "disease": ["Diabetes"],
            "speciality": ["Endocrinology"],
            "doctor_id": ["DOC0001"],
            "hospital_name": ["City Care Hospital"],
            "claim_amount": [1000.0],
            "claim_status": ["APPROVED"],
            "denial_reason": [""],
            "service_date": ["2024-01-01"],
            "submission_date": ["2024-01-05"],
            "payer_name": ["MediPlus"],
        }
    )


def test_transform_adds_claim_text_and_optional_defaults():
    transformed = transform(base_claims_df())
    assert "claim_text" in transformed.columns
    assert "diagnosis_code" in transformed.columns
    assert "Claim CLM00001" in transformed.iloc[0]["claim_text"]


def test_transform_rejects_duplicate_claim_ids():
    df = pd.concat([base_claims_df(), base_claims_df()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate claim_id"):
        transform(df)
