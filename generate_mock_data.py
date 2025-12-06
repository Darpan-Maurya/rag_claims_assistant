import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# =====================
# CONFIG
# =====================
NUM_ROWS = 5000  # you can change to 1000–5000 later
OUTPUT_PATH = Path("data/raw/claims.csv")

# =====================
# DOMAIN VALUES
# =====================
diseases = [
    "Diabetes", "Hypertension", "Asthma",
    "COPD", "Coronary Artery Disease", "Covid-19"
]

specialities = [
    "Endocrinology", "Cardiology",
    "Pulmonology", "General Medicine"
]

claim_statuses = ["APPROVED", "DENIED", "PENDING"]

denial_reasons = [
    "Insufficient documentation",
    "Not medically necessary",
    "Pre-authorization missing",
    "Coverage limit exceeded",
    "Out-of-network provider"
]

hospitals = [
    "City Care Hospital",
    "Green Valley Clinic",
    "Sunrise Medical Center",
    "Metro Health"
]

payers = [
    "ABC Health Insurance",
    "SecureLife Health",
    "MediPlus",
    "CareFirst"
]

genders = ["M", "F"]

# =====================
# DATA GENERATION
# =====================
def generate_claims():
    rows = []
    start_date = datetime(2023, 1, 1)

    for i in range(NUM_ROWS):
        claim_status = np.random.choice(
            claim_statuses, p=[0.6, 0.3, 0.1]
        )

        denial_reason = ""
        if claim_status == "DENIED":
            denial_reason = random.choice(denial_reasons)

        service_date = start_date + timedelta(
            days=random.randint(0, 365)
        )
        submission_date = service_date + timedelta(
            days=random.randint(1, 30)
        )

        row = {
            "claim_id": f"CLM{i+1:05d}",
            "patient_id": f"PAT{random.randint(1, 800):04d}",
            "patient_age": random.randint(18, 90),
            "patient_gender": random.choice(genders),
            "disease": random.choice(diseases),
            "speciality": random.choice(specialities),
            "doctor_id": f"DOC{random.randint(1, 200):04d}",
            "hospital_name": random.choice(hospitals),
            "claim_amount": round(random.uniform(1000, 150000), 2),
            "claim_status": claim_status,
            "denial_reason": denial_reason,
            "service_date": service_date.date().isoformat(),
            "submission_date": submission_date.date().isoformat(),
            "payer_name": random.choice(payers),
        }

        rows.append(row)

    return pd.DataFrame(rows)

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_claims()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {len(df)} rows at {OUTPUT_PATH}")
