from typing import Any, Dict

import pandas as pd

from orchestrate.router import QueryFilters, apply_filters


def approval_percentage(df):
    total = len(df)
    if total == 0:
        return 0.0
    approved = len(df[df["claim_status"] == "APPROVED"])
    return round((approved / total) * 100, 2)


def answer_analytics_query(query: str, df: pd.DataFrame, filters: QueryFilters) -> Dict[str, Any]:
    filtered = apply_filters(df, filters)
    q = query.lower()

    metrics: Dict[str, Any] = {
        "total_claims": int(len(filtered)),
        "filters": filters.to_dict(),
    }

    if filtered.empty:
        return {
            "answer": "No claims matched the requested filters, so no analytics could be calculated.",
            "metrics": metrics,
        }

    status_counts = filtered["claim_status"].value_counts().to_dict()
    metrics["status_counts"] = {str(key): int(value) for key, value in status_counts.items()}
    metrics["approval_percentage"] = approval_percentage(filtered)
    metrics["denial_percentage"] = round(
        (metrics["status_counts"].get("DENIED", 0) / len(filtered)) * 100, 2
    )
    metrics["total_claim_amount"] = round(float(filtered["claim_amount"].sum()), 2)
    metrics["average_claim_amount"] = round(float(filtered["claim_amount"].mean()), 2)

    if "denial_reason" in filtered.columns:
        reasons = (
            filtered[filtered["claim_status"] == "DENIED"]["denial_reason"]
            .replace("", "N/A")
            .value_counts()
            .head(5)
            .to_dict()
        )
        metrics["top_denial_reasons"] = {str(key): int(value) for key, value in reasons.items()}

    if "percentage" in q or "percent" in q or "rate" in q:
        answer = (
            f"Across {len(filtered)} matching claims, {metrics['approval_percentage']}% were approved "
            f"and {metrics['denial_percentage']}% were denied."
        )
    elif "reason" in q or "common" in q or "top" in q:
        reasons = metrics.get("top_denial_reasons", {})
        if reasons:
            answer = (
                f"Across {len(filtered)} matching claims, the top denial reasons are "
                + ", ".join(f"{reason}: {count}" for reason, count in reasons.items())
                + "."
            )
        else:
            answer = f"Across {len(filtered)} matching claims, no denied-claim reasons were found."
    elif "average" in q or "avg" in q:
        answer = (
            f"Across {len(filtered)} matching claims, the average claim amount is "
            f"{metrics['average_claim_amount']} INR."
        )
    elif "total" in q or "sum" in q:
        answer = (
            f"Across {len(filtered)} matching claims, the total claim amount is "
            f"{metrics['total_claim_amount']} INR."
        )
    else:
        answer = (
            f"I found {len(filtered)} matching claims. Status counts: "
            + ", ".join(f"{status}: {count}" for status, count in metrics["status_counts"].items())
            + "."
        )

    return {"answer": answer, "metrics": metrics}
