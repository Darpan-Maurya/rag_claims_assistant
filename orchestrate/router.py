from dataclasses import dataclass
from typing import Dict

from core.config import settings
from core.query_classifier import BLOCKED_LABELS, QueryClassification


@dataclass(frozen=True)
class DataSource:
    key: str
    route: str
    description: str
    cost_class: str
    latency_class: str


DATA_SOURCES = {
    "claims_analytics": DataSource(
        key="claims_analytics",
        route="ANALYTICS",
        description="Deterministic aggregations over authorized claims rows.",
        cost_class="low",
        latency_class="low",
    ),
    "claims_rag": DataSource(
        key="claims_rag",
        route="RAG",
        description="Authorized internal claim evidence through hybrid retrieval.",
        cost_class="medium",
        latency_class="medium",
    ),
    "public_web": DataSource(
        key="public_web",
        route="WEB_SEARCH",
        description="Current public information from an explicitly configured web provider.",
        cost_class="high",
        latency_class="high",
    ),
    "general_llm": DataSource(
        key="general_llm",
        route="LLM_ONLY",
        description="General explanation without accessing internal claims data.",
        cost_class="medium",
        latency_class="medium",
    ),
    "policy_refusal": DataSource(
        key="policy_refusal",
        route="DECISION_HELP",
        description="Unsupported individual coverage decision request.",
        cost_class="none",
        latency_class="low",
    ),
    "safety_block": DataSource(
        key="safety_classifier",
        route="BLOCKED",
        description="Model-classified unsafe or private-data request.",
        cost_class="none",
        latency_class="low",
    ),
}

LABEL_TO_SOURCE = {
    "ANALYTICS": "claims_analytics",
    "CLAIMS_RAG": "claims_rag",
    "DECISION_HELP": "policy_refusal",
    "LLM_ONLY": "general_llm",
    "WEB_SEARCH": "public_web",
}

LABEL_REASON_CODES = {
    "ANALYTICS": "deterministic_analytics",
    "CLAIMS_RAG": "single_source_retrieval",
    "DECISION_HELP": "individual_coverage_decision",
    "LLM_ONLY": "no_internal_data_access",
    "WEB_SEARCH": "live_public_information",
}


@dataclass(frozen=True)
class QueryPlan:
    route: str
    data_source: str
    reason_codes: tuple[str, ...]
    estimated_cost: str
    estimated_latency: str
    classifier: Dict[str, object]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "data_source": self.data_source,
            "reason_codes": list(self.reason_codes),
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency,
            "classifier": self.classifier,
            "warnings": list(self.warnings),
        }


def _plan_for(
    source_key: str,
    classification: QueryClassification,
    *reason_codes: str,
    warnings: tuple[str, ...] = (),
) -> QueryPlan:
    source = DATA_SOURCES[source_key]
    return QueryPlan(
        route=source.route,
        data_source=source.key,
        reason_codes=tuple(reason_codes),
        estimated_cost=source.cost_class,
        estimated_latency=source.latency_class,
        classifier=classification.to_dict(),
        warnings=warnings,
    )


def plan_query(
    classification: QueryClassification,
    *,
    web_search_available: bool = False,
) -> QueryPlan:
    """Map one local model prediction to one source with no fan-out."""

    if classification.label in BLOCKED_LABELS:
        return _plan_for("safety_block", classification, "model_safety_classification")

    if classification.confidence < settings.router_min_confidence:
        return _plan_for(
            "general_llm",
            classification,
            "low_classifier_confidence",
            warnings=(
                "The routing model was uncertain, so no internal or external data source was opened.",
            ),
        )

    if classification.label == "WEB_SEARCH" and not web_search_available:
        return _plan_for(
            "general_llm",
            classification,
            "web_provider_unavailable",
            warnings=("Live web search is not configured; answering without external sources.",),
        )

    source_key = LABEL_TO_SOURCE.get(classification.label, "general_llm")
    reason_code = LABEL_REASON_CODES.get(classification.label, "no_internal_data_access")
    return _plan_for(source_key, classification, "model_intent_classification", reason_code)
