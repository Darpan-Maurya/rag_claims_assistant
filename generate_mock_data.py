import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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

diagnosis_codes = {
    "Diabetes": ["E11.9", "E11.65", "E10.9"],
    "Hypertension": ["I10", "I11.9", "I15.9"],
    "Asthma": ["J45.909", "J45.901", "J45.40"],
    "COPD": ["J44.9", "J44.1", "J44.0"],
    "Coronary Artery Disease": ["I25.10", "I25.119", "I20.9"],
    "Covid-19": ["U07.1", "J12.82", "Z86.16"],
}

specialities = [
    "Endocrinology", "Cardiology",
    "Pulmonology", "General Medicine"
]

procedure_codes = {
    "Endocrinology": ["83036", "95251", "99214"],
    "Cardiology": ["93000", "93306", "99215"],
    "Pulmonology": ["94010", "94640", "99214"],
    "General Medicine": ["99213", "80053", "85025"],
}

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
plan_types = ["HMO", "PPO", "EPO", "POS", "Medicare Advantage"]
provider_types = ["Hospital", "Clinic", "Specialist Group", "Ambulatory Center"]
member_states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"]
appeal_statuses = ["NOT_APPEALED", "APPEALED_PENDING", "APPEAL_UPHELD", "APPEAL_OVERTURNED"]


def choose_denial_reason(network_status: str, prior_authorization_flag: bool, amount: float) -> str:
    weighted_reasons = denial_reasons.copy()
    if network_status == "OUT_OF_NETWORK":
        weighted_reasons += ["Out-of-network provider"] * 4
    if not prior_authorization_flag:
        weighted_reasons += ["Pre-authorization missing"] * 4
    if amount > 100000:
        weighted_reasons += ["Coverage limit exceeded"] * 3
    return random.choice(weighted_reasons)

# =====================
# DATA GENERATION
# =====================
def generate_claims():
    rows = []
    start_date = datetime(2022, 1, 1)

    for i in range(NUM_ROWS):
        disease = random.choice(diseases)
        speciality = random.choice(specialities)
        claim_amount = round(random.uniform(1000, 150000), 2)
        network_status = np.random.choice(["IN_NETWORK", "OUT_OF_NETWORK"], p=[0.82, 0.18])
        prior_authorization_flag = bool(np.random.choice([True, False], p=[0.72, 0.28]))

        denial_risk = 0.17
        if network_status == "OUT_OF_NETWORK":
            denial_risk += 0.18
        if not prior_authorization_flag:
            denial_risk += 0.22
        if claim_amount > 100000:
            denial_risk += 0.08
        denial_risk = min(denial_risk, 0.72)
        pending_risk = 0.08
        approved_risk = max(0.05, 1 - denial_risk - pending_risk)

        claim_status = np.random.choice(
            claim_statuses, p=[approved_risk, denial_risk, pending_risk]
        )

        denial_reason = choose_denial_reason(
            network_status, prior_authorization_flag, claim_amount
        ) if claim_status == "DENIED" else ""

        service_date = start_date + timedelta(
            days=random.randint(0, 900)
        )
        submission_date = service_date + timedelta(
            days=random.randint(1, 30)
        )
        allowed_ratio = random.uniform(0.55, 0.95) if network_status == "IN_NETWORK" else random.uniform(0.25, 0.6)
        allowed_amount = round(claim_amount * allowed_ratio, 2)
        deductible = round(random.uniform(0, min(5000, allowed_amount * 0.2)), 2)
        copay = round(random.choice([0, 25, 50, 75, 100, 150]), 2)
        paid_amount = 0.0
        if claim_status == "APPROVED":
            paid_amount = round(max(0, allowed_amount - deductible - copay), 2)

        appeal_status = "NOT_APPEALED"
        if claim_status == "DENIED":
            appeal_status = np.random.choice(appeal_statuses, p=[0.58, 0.22, 0.12, 0.08])

        row = {
            "claim_id": f"CLM{i+1:05d}",
            "patient_id": f"PAT{random.randint(1, 800):04d}",
            "patient_age": random.randint(18, 90),
            "patient_gender": random.choice(genders),
            "member_state": random.choice(member_states),
            "disease": disease,
            "diagnosis_code": random.choice(diagnosis_codes[disease]),
            "speciality": speciality,
            "procedure_code": random.choice(procedure_codes[speciality]),
            "doctor_id": f"DOC{random.randint(1, 200):04d}",
            "hospital_name": random.choice(hospitals),
            "provider_type": random.choice(provider_types),
            "plan_type": random.choice(plan_types),
            "claim_amount": claim_amount,
            "allowed_amount": allowed_amount,
            "paid_amount": paid_amount,
            "deductible": deductible,
            "copay": copay,
            "network_status": network_status,
            "prior_authorization_flag": prior_authorization_flag,
            "claim_status": claim_status,
            "denial_reason": denial_reason,
            "appeal_status": appeal_status,
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
