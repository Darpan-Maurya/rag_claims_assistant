import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import pandas as pd

from orchestrate.filters import QueryFilters, apply_filters


@dataclass(frozen=True)
class AnalyticsIntent:
    metric: str
    target_status: Optional[str]
    negated_status: Optional[str]
    amount_column: str
    top_n: Optional[int]
    population_filters: QueryFilters


def approval_percentage(df: pd.DataFrame) -> float:
    total = len(df)
    if total == 0:
        return 0.0
    approved = len(df[df["claim_status"] == "APPROVED"])
    return round((approved / total) * 100, 2)


def derive_analytics_intent(query: str, filters: QueryFilters) -> AnalyticsIntent:
    """Separate the requested metric from filters defining its population.

    In "what percentage of claims are denied?", DENIED is the numerator, not
    a population filter. In "total amount of denied claims", it is a real
    population filter. Keeping that distinction explicit prevents 100% answers.
    """

    q = query.lower()
    top_match = re.search(r"\btop\s+(\d+)\b", q)
    top_n = (
        int(top_match.group(1))
        if top_match
        and any(token in q for token in ("highest", "largest", "claim amount", "by amount"))
        else None
    )

    if any(token in q for token in ("percentage", "percent", "rate")):
        metric = "percentage"
    elif re.search(r"\b(how many|count|number of|number)\b", q):
        metric = "count"
    elif top_n is not None:
        metric = "top_claims"
    elif any(token in q for token in ("denial reason", "denial reasons", "common reason", "common reasons")):
        metric = "denial_reasons"
    elif any(token in q for token in ("average", "avg", "mean")):
        metric = "average_amount"
    elif any(token in q for token in ("total", "sum")):
        metric = "total_amount" if any(
            token in q for token in ("amount", "cost", "paid", "allowed", "billed")
        ) else "count"
    else:
        metric = "status_breakdown"

    if "paid" in q and "paid_amount" in q or "paid amount" in q:
        amount_column = "paid_amount"
    elif "allowed" in q and "allowed amount" in q:
        amount_column = "allowed_amount"
    else:
        amount_column = "claim_amount"

    status_is_metric = metric in {"percentage", "count"} and (
        filters.claim_status is not None or filters.negated_claim_status is not None
    )
    population_filters = (
        replace(filters, claim_status=None, negated_claim_status=None)
        if status_is_metric
        else filters
    )

    return AnalyticsIntent(
        metric=metric,
        target_status=filters.claim_status if status_is_metric else None,
        negated_status=filters.negated_claim_status if status_is_metric else None,
        amount_column=amount_column,
        top_n=top_n,
        population_filters=population_filters,
    )


def _percentage(count: int, total: int) -> float:
    return round((count / total) * 100, 2) if total else 0.0


def _top_denial_reasons(df: pd.DataFrame) -> Dict[str, int]:
    if "denial_reason" not in df.columns:
        return {}
    reasons = (
        df[df["claim_status"] == "DENIED"]["denial_reason"]
        .replace("", "N/A")
        .value_counts()
        .head(5)
        .to_dict()
    )
    return {str(key): int(value) for key, value in reasons.items()}


def answer_analytics_query(
    query: str, df: pd.DataFrame, filters: QueryFilters
) -> Dict[str, Any]:
    intent = derive_analytics_intent(query, filters)
    filtered = apply_filters(df, intent.population_filters)

    if intent.top_n is not None and not filtered.empty:
        filtered = filtered.sort_values(by=intent.amount_column, ascending=False).head(intent.top_n)

    metrics: Dict[str, Any] = {
        "total_claims": int(len(filtered)),
        "requested_filters": filters.to_dict(),
        "population_filters": intent.population_filters.to_dict(),
        "metric": intent.metric,
        "metric_target": (
            f"not_{intent.negated_status.lower()}"
            if intent.negated_status
            else intent.target_status
        ),
        "amount_column": intent.amount_column,
    }

    if filtered.empty:
        return {
            "answer": "No claims matched the requested population filters, so no analytics could be calculated.",
            "metrics": metrics,
            "effective_filters": intent.population_filters.to_dict(),
        }

    status_counts = {
        str(key): int(value)
        for key, value in filtered["claim_status"].value_counts().to_dict().items()
    }
    metrics["status_counts"] = status_counts
    metrics["approval_percentage"] = approval_percentage(filtered)
    metrics["denial_percentage"] = _percentage(status_counts.get("DENIED", 0), len(filtered))
    metrics["total_claim_amount"] = round(float(filtered[intent.amount_column].sum()), 2)
    metrics["average_claim_amount"] = round(float(filtered[intent.amount_column].mean()), 2)
    metrics["top_denial_reasons"] = _top_denial_reasons(filtered)

    if intent.metric == "percentage":
        if intent.target_status:
            count = status_counts.get(intent.target_status, 0)
            percentage = _percentage(count, len(filtered))
            metrics["target_count"] = count
            metrics["target_percentage"] = percentage
            answer = (
                f"Across {len(filtered)} matching claims, {percentage}% were "
                f"{intent.target_status.lower()}."
            )
        elif intent.negated_status:
            status_count = status_counts.get(intent.negated_status, 0)
            percentage = _percentage(len(filtered) - status_count, len(filtered))
            metrics["target_count"] = len(filtered) - status_count
            metrics["target_percentage"] = percentage
            answer = (
                f"Across {len(filtered)} matching claims, {percentage}% were not "
                f"{intent.negated_status.lower()}."
            )
        else:
            answer = (
                f"Across {len(filtered)} matching claims, {metrics['approval_percentage']}% were approved "
                f"and {metrics['denial_percentage']}% were denied."
            )
    elif intent.metric == "count":
        if intent.target_status:
            count = status_counts.get(intent.target_status, 0)
            metrics["target_count"] = count
            answer = f"There are {count} {intent.target_status.lower()} claims out of {len(filtered)} matching claims."
        elif intent.negated_status:
            count = len(filtered) - status_counts.get(intent.negated_status, 0)
            metrics["target_count"] = count
            answer = f"There are {count} claims that are not {intent.negated_status.lower()} out of {len(filtered)} matching claims."
        else:
            answer = f"There are {len(filtered)} matching claims."
    elif intent.metric == "denial_reasons":
        reasons = metrics["top_denial_reasons"]
        if reasons:
            answer = (
                f"Across {len(filtered)} matching claims, the top denial reasons are "
                + ", ".join(f"{reason}: {count}" for reason, count in reasons.items())
                + "."
            )
        else:
            answer = f"Across {len(filtered)} matching claims, no denied-claim reasons were found."
    elif intent.metric == "average_amount":
        answer = (
            f"Across {len(filtered)} matching claims, the average {intent.amount_column.replace('_', ' ')} is "
            f"{metrics['average_claim_amount']} INR."
        )
    elif intent.metric == "total_amount":
        answer = (
            f"Across {len(filtered)} matching claims, the total {intent.amount_column.replace('_', ' ')} is "
            f"{metrics['total_claim_amount']} INR."
        )
    elif intent.metric == "top_claims":
        claim_ids = filtered.get("claim_id", pd.Series(dtype=str)).astype(str).tolist()
        metrics["top_claim_ids"] = claim_ids
        answer = (
            f"The top {len(filtered)} claims by {intent.amount_column.replace('_', ' ')} are "
            + ", ".join(claim_ids)
            + "."
        )
    else:
        answer = (
            f"Across {len(filtered)} matching claims, status counts are "
            + ", ".join(f"{status}: {count}" for status, count in status_counts.items())
            + "."
        )

    return {
        "answer": answer,
        "metrics": metrics,
        "effective_filters": intent.population_filters.to_dict(),
    }
