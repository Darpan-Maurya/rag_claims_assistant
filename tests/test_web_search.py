from types import SimpleNamespace

from rag import web_search


def test_tavily_search_builds_allowlisted_request(monkeypatch):
    monkeypatch.setattr(
        web_search,
        "settings",
        SimpleNamespace(
            web_search_provider="tavily",
            tavily_api_key="test-key",
            web_search_max_results=5,
            web_search_timeout_seconds=4,
            web_search_allowed_domains=["cms.gov"],
        ),
    )
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "CMS source",
                        "url": "https://www.cms.gov/example",
                        "content": "Current source content",
                        "score": 0.91,
                    }
                ]
            }

    def fake_post(url, json, timeout):
        captured.update({"url": url, "payload": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(web_search.requests, "post", fake_post)

    results = web_search.WebSearchService().search("latest CMS claims regulation")

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["payload"]["include_domains"] == ["cms.gov"]
    assert captured["timeout"] == 4
    assert results[0].to_evidence()["url"] == "https://www.cms.gov/example"
