import pandas as pd

from analytics.claims_analytics import answer_analytics_query
from orchestrate.router import QueryFilters


def test_analytics_answer_uses_deterministic_counts():
    df = pd.DataFrame(
        {
            "claim_status": ["APPROVED", "DENIED", "DENIED"],
            "claim_amount": [100.0, 200.0, 300.0],
            "denial_reason": ["", "Insufficient documentation", "Coverage limit exceeded"],
            "disease": ["Diabetes", "Diabetes", "Asthma"],
            "speciality": ["Cardiology", "Cardiology", "Pulmonology"],
            "payer_name": ["A", "A", "B"],
            "service_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    result = answer_analytics_query("What percentage approved?", df, QueryFilters())
    assert result["metrics"]["total_claims"] == 3
    assert result["metrics"]["approval_percentage"] == 33.33
    assert "33.33%" in result["answer"]
