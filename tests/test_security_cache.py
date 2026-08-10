from core.cache import QueryCache
from core.security import Principal, apply_row_level_access, enforce_filter_access
from orchestrate.filters import QueryFilters

import pandas as pd
import pytest


def test_cache_key_separates_payer_scopes():
    cache = QueryCache()
    first = cache.make_key(
        user_id="u1",
        role="analyst",
        allowed_payers=["MediPlus"],
        query="denied claims",
        top_k=10,
        route="RAG",
        filters={},
    )
    second = cache.make_key(
        user_id="u2",
        role="analyst",
        allowed_payers=["CareFirst"],
        query="denied claims",
        top_k=10,
        route="RAG",
        filters={},
    )
    assert first != second


def test_row_level_access_filters_payers():
    df = pd.DataFrame({"payer_name": ["MediPlus", "CareFirst"], "claim_id": ["A", "B"]})
    principal = Principal(user_id="u1", role="analyst", allowed_payers=["MediPlus"])
    visible = apply_row_level_access(df, principal)
    assert visible["claim_id"].tolist() == ["A"]


def test_filter_access_rejects_disallowed_payer():
    principal = Principal(user_id="u1", role="analyst", allowed_payers=["MediPlus"])
    with pytest.raises(PermissionError):
        enforce_filter_access(QueryFilters(payer_name="CareFirst"), principal)
