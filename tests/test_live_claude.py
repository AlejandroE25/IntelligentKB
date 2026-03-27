"""Opt-in live integration tests for Anthropic Claude responses.

These tests are skipped by default and only run when:
- RUN_LIVE_CLAUDE_TESTS=1
- ANTHROPIC_API_KEY is set
"""

from __future__ import annotations

import os

import anthropic
import pytest


RUN_LIVE = os.environ.get("RUN_LIVE_CLAUDE_TESTS") == "1"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

pytestmark = pytest.mark.skipif(
    not RUN_LIVE or not API_KEY,
    reason="Set RUN_LIVE_CLAUDE_TESTS=1 and ANTHROPIC_API_KEY to run live Claude tests.",
)


def _article(article_id: str, content: str) -> dict[str, str]:
    return {
        "filename": f"{article_id}.html",
        "article_id": article_id,
        "title": "Campus Wi-Fi Setup",
        "keywords": "wifi securew2 illinoisnet",
        "content": content,
        "internal": "",
        "updated": "2025-01-01",
        "owner": "Networking Team",
    }


def test_live_ai_endpoint_returns_claude_response(monkeypatch):
    """Verify /ai returns a real Claude response (not a mock)."""
    from main import build_article_index, create_app
    import main as app_module

    # Keep this live test lightweight to limit token spend.
    monkeypatch.setattr(app_module, "MAX_TOKENS", 256)

    articles = [
        _article(
            "10001",
            "Use SecureW2 JoinNow. If login fails, reset NetID password and retry.",
        )
    ]
    vectorizer, matrix = build_article_index(articles)

    client = anthropic.Anthropic(api_key=API_KEY, max_retries=0)
    app = create_app(client, articles, vectorizer, matrix, contacts_text="")
    app.config["TESTING"] = True

    with app.test_client() as c:
        resp = c.post(
            "/ai",
            data={
                "query": "Wi-Fi login fails. Give concise steps in 1-2 sentences.",
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["error"] == ""
    assert isinstance(data["response"], str)
    assert data["response"].strip() != ""
