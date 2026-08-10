from core.query_classifier import QueryClassifier
from orchestrate.router import plan_query


def _plan(query: str, *, web_search_available: bool = False):
    classification = QueryClassifier().classify(query)
    return plan_query(classification, web_search_available=web_search_available)


def test_planner_selects_low_cost_analytics_for_claim_aggregation():
    plan = _plan("What percentage of claims are denied?")

    assert plan.route == "ANALYTICS"
    assert plan.data_source == "claims_analytics"
    assert plan.estimated_cost == "low"


def test_planner_selects_internal_rag_for_claim_evidence():
    plan = _plan("Show denied diabetes claims from last quarter")

    assert plan.route == "RAG"
    assert plan.data_source == "claims_rag"
    assert "single_source_retrieval" in plan.reason_codes


def test_planner_uses_llm_only_for_general_explanations():
    plan = _plan("What is prior authorization?")

    assert plan.route == "LLM_ONLY"
    assert plan.data_source == "general_llm"
    assert "no_internal_data_access" in plan.reason_codes


def test_planner_selects_web_only_when_available_and_requested():
    query = "Find the latest CMS claims regulations"

    enabled_plan = _plan(query, web_search_available=True)
    disabled_plan = _plan(query, web_search_available=False)

    assert enabled_plan.route == "WEB_SEARCH"
    assert enabled_plan.data_source == "public_web"
    assert disabled_plan.route == "LLM_ONLY"
    assert "web_provider_unavailable" in disabled_plan.reason_codes


def test_planner_refuses_individual_coverage_decisions():
    plan = _plan("Will my claim be approved?")

    assert plan.route == "DECISION_HELP"
    assert plan.estimated_cost == "none"
