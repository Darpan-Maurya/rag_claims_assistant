from typing import Any, Dict

import re
import pandas as pd

from orchestrate.router import QueryFilters, apply_filters


def approval_percentage(df):
    total = len(df)
    if total == 0:
        return 0.0
    approved = len(df[df["claim_status"] == "APPROVED"])
    return round((approved / total) * 100, 2)


def answer_analytics_query(query: str, df: pd.DataFrame, filters: QueryFilters) -> Dict[str, Any]:
    # If the user asked for a negated status (e.g. "not approved"),
    # compute percentages relative to the full set matching other filters.
    if getattr(filters, "negated_claim_status", None):
        # build a copy of filters without any status constraints
        nf = QueryFilters(
            claim_status=None,
            disease=filters.disease,
            speciality=filters.speciality,
            payer_name=filters.payer_name,
            denial_reason=filters.denial_reason,
            service_date_start=filters.service_date_start,
            service_date_end=filters.service_date_end,
            amount_min=filters.amount_min,
            amount_max=filters.amount_max,
        )
        base_df = apply_filters(df, nf)
        q = query.lower()

        metrics = {
            "total_claims": int(len(base_df)),
            "filters": filters.to_dict(),
        }

        if base_df.empty:
            return {
                "answer": "No claims matched the requested filters, so no analytics could be calculated.",
                "metrics": metrics,
            }

        status = filters.negated_claim_status
        status_count = int(len(base_df[base_df["claim_status"] == status]))
        not_status_pct = round((1 - (status_count / len(base_df))) * 100, 2)

        metrics["status_counts"] = {str(key): int(value) for key, value in base_df["claim_status"].value_counts().to_dict().items()}
        metrics[f"not_{status.lower()}_percentage"] = not_status_pct

        answer = f"Across {len(base_df)} matching claims, {not_status_pct}% were not {status.lower()}."
        return {"answer": answer, "metrics": metrics}

    filtered = apply_filters(df, filters)
    q = query.lower()

    # support "top N" analytics (e.g. "top 10 claims with highest claim amount")
    top_n = None
    top_match = re.search(r"top\s+(\d+)", q)
    if top_match and ("highest" in q or "largest" in q or "claim amount" in q or "by amount" in q):
        try:
            top_n = int(top_match.group(1))
        except ValueError:
            top_n = None
    if top_n is not None and not filtered.empty:
        filtered = filtered.sort_values(by="claim_amount", ascending=False).head(top_n)

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
