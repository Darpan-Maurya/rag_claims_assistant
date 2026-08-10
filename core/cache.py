import hashlib
import json
from typing import Any, Dict, Optional

from core.config import settings

try:
    import redis
except ImportError:  # pragma: no cover - handled in runtime readiness
    redis = None


class QueryCache:
    def __init__(self) -> None:
        self.enabled = bool(settings.enable_query_cache and settings.redis_url and redis)
        self.prefix = "rag_claims:query:v2"
        self.client = None
        if self.enabled:
            self.client = redis.Redis.from_url(  # type: ignore[union-attr]
                settings.redis_url,
                decode_responses=True,
            )

    def make_key(
        self,
        *,
        user_id: str,
        role: str,
        allowed_payers: list[str],
        query: str,
        top_k: int,
        route: str,
        filters: Dict[str, Any],
    ) -> str:
        payload = {
            "user_id": user_id,
            "role": role,
            "allowed_payers": sorted(allowed_payers),
            "query": query.strip().lower(),
            "top_k": top_k,
            "route": route,
            "filters": filters,
            "collection": settings.qdrant_collection,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"{self.prefix}:{digest}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.client is None:
            return None
        try:
            value = self.client.get(key)
            if not value:
                return None
            return json.loads(value)
        except Exception:
            # Cache failure must degrade to a normal request, never a 500.
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.setex(
                key,
                settings.query_cache_ttl_seconds,
                json.dumps(value, default=str),
            )
        except Exception:
            return

    def invalidate_all(self) -> int:
        if not self.enabled or self.client is None:
            return 0
        try:
            deleted = 0
            for key in self.client.scan_iter(f"{self.prefix}:*"):
                deleted += int(self.client.delete(key))
            return deleted
        except Exception:
            return 0

    def readiness(self) -> Dict[str, Any]:
        if not settings.enable_query_cache:
            return {"enabled": False, "reason": "disabled_by_config"}
        if not settings.redis_url:
            return {"enabled": False, "reason": "REDIS_URL_not_configured"}
        if redis is None:
            return {"enabled": False, "reason": "redis_package_not_installed"}
        try:
            assert self.client is not None
            self.client.ping()
            return {
                "enabled": True,
                "backend": "redis",
                "ttl_seconds": settings.query_cache_ttl_seconds,
            }
        except Exception as exc:
            return {"enabled": False, "reason": str(exc)}
