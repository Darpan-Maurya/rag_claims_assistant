from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from core.config import settings
from core.observability import logger, metrics, timed_metric


class WebSearchError(RuntimeError):
    """A public-web provider could not return a safe search result."""


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    content: str
    score: float | None = None

    def to_evidence(self) -> Dict[str, Any]:
        return {
            "type": "public_web_source",
            "title": self.title,
            "url": self.url,
            "snippet": self.content,
            "score": self.score,
        }


class WebSearchService:
    """Explicit, allow-listable live web search adapter.

    The service is disabled unless Tavily is selected and its API key is supplied.
    It is never called for internal claims retrieval.
    """

    provider = "tavily"

    @property
    def is_available(self) -> bool:
        return settings.web_search_provider == self.provider and bool(settings.tavily_api_key)

    def readiness(self) -> Dict[str, Any]:
        if settings.web_search_provider == "disabled":
            return {"enabled": False, "reason": "disabled_by_config"}
        if settings.web_search_provider != self.provider:
            return {
                "enabled": False,
                "reason": f"unsupported_provider:{settings.web_search_provider}",
            }
        if not settings.tavily_api_key:
            return {"enabled": False, "reason": "TAVILY_API_KEY_not_configured"}
        return {
            "enabled": True,
            "provider": self.provider,
            "max_results": min(max(settings.web_search_max_results, 1), 10),
            "allowed_domains": settings.web_search_allowed_domains,
        }

    def search(self, query: str) -> List[WebSearchResult]:
        if not self.is_available:
            raise WebSearchError("Live web search is not configured.")

        payload: Dict[str, Any] = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": min(max(settings.web_search_max_results, 1), 10),
            "include_answer": False,
            "include_raw_content": False,
        }
        if settings.web_search_allowed_domains:
            payload["include_domains"] = settings.web_search_allowed_domains

        try:
            with timed_metric("web_search"):
                response = requests.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    timeout=max(settings.web_search_timeout_seconds, 1),
                )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            metrics.increment("web_search.errors")
            logger.warning(
                "web_search_failed",
                extra={"error_type": type(exc).__name__},
            )
            raise WebSearchError("Live web search provider is temporarily unavailable.") from exc
        except ValueError as exc:
            metrics.increment("web_search.errors")
            logger.warning(
                "web_search_invalid_response",
                extra={"error_type": type(exc).__name__},
            )
            raise WebSearchError("Live web search provider returned an invalid response.") from exc

        results: List[WebSearchResult] = []
        for item in body.get("results", []):
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            if not url or not title:
                continue
            score = item.get("score")
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    content=content[:2000],
                    score=float(score) if isinstance(score, (int, float)) else None,
                )
            )

        metrics.increment("web_search.requests")
        return results
