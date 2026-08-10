import hmac
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from core.config import settings
from orchestrate.filters import QueryFilters


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    allowed_payers: List[str]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_ingest(self) -> bool:
        return self.role in {"admin", "claims_manager"}

    @property
    def can_view_metrics(self) -> bool:
        return self.role in {"admin", "claims_manager"}

    @property
    def can_access_all_payers(self) -> bool:
        return "*" in self.allowed_payers or self.is_admin


def _configured_principals() -> list[tuple[str, Principal]]:
    principals: list[tuple[str, Principal]] = []
    for item in settings.rbac_users:
        api_key = str(item.get("api_key", ""))
        if not api_key:
            continue
        principals.append(
            (
                api_key,
                Principal(
                    user_id=str(item.get("user_id", item.get("role", "user"))),
                    role=str(item.get("role", "viewer")),
                    allowed_payers=list(item.get("allowed_payers", [])) or [],
                ),
            )
        )
    if settings.api_key:
        principals.append(
            (
                settings.api_key,
                Principal(user_id="default-admin", role="admin", allowed_payers=["*"]),
            )
        )
    return principals


def authenticate_api_key(api_key: Optional[str]) -> Optional[Principal]:
    configured = _configured_principals()
    if not configured and settings.environment == "development":
        return Principal(user_id="local-dev", role="admin", allowed_payers=["*"])
    if not api_key:
        return None
    for configured_key, principal in configured:
        if hmac.compare_digest(configured_key, api_key):
            return principal
    return None


def enforce_filter_access(filters: QueryFilters, principal: Principal) -> QueryFilters:
    if principal.can_access_all_payers:
        return filters
    if not principal.allowed_payers:
        raise PermissionError("User has no payer access grants.")
    if filters.payer_name and filters.payer_name not in principal.allowed_payers:
        raise PermissionError("User is not allowed to access the requested payer.")
    return filters


def apply_row_level_access(df: pd.DataFrame, principal: Principal) -> pd.DataFrame:
    if principal.can_access_all_payers:
        return df
    if "payer_name" not in df.columns:
        return df.iloc[0:0]
    allowed = set(principal.allowed_payers)
    return df[df["payer_name"].isin(allowed)]


def visible_payers(principal: Principal, all_payers: Iterable[str]) -> List[str]:
    if principal.can_access_all_payers:
        return sorted(str(payer) for payer in all_payers)
    return sorted(principal.allowed_payers)
