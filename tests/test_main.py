"""Unit tests for main.py"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_HTML = """\
<!DOCTYPE html>
<html lang="en-US">
<head>
  <title>Test Article Title</title>
  <meta name="keywords" content="ignored meta keywords">
</head>
<body>
  <header>Header noise</header>
  <nav>Nav noise</nav>
  <aside>Aside noise</aside>
  <script>script noise</script>
  <style>style noise</style>

  <!-- Article ID link -->
  <input onclick="window.open('resultc.php?action=7&amp;id=99999')">

  <!-- Keywords -->
  <span id="kb-page-keywords">wifi wireless test keyword</span>

  <!-- Main content -->
  <div id="kbcontent">This is the main article content. Step one. Step two.</div>

  <!-- Internal staff section -->
  <div class="kb-class-internal-site">Internal staff notes here.</div>

  <!-- Metadata (doc-attr) -->
  <div class="doc-attr updated">
    <div class="doc-attr-name">Updated:</div>
    <div class="doc-attr-value">2020-06-15</div>
  </div>
  <div class="doc-attr owner">
    <div class="doc-attr-name">Owned by:</div>
    <div class="doc-attr-value">wireless G. in University of Illinois Technology Services</div>
  </div>
  <div class="doc-attr id">
    <div class="doc-attr-name">Doc ID:</div>
    <div class="doc-attr-value">99999</div>
  </div>

  <!-- Feedback buttons (should be stripped) -->
  <button class="feedback-btn">Helpful</button>

  <footer>Footer noise</footer>
</body>
</html>
"""

FRESH_HTML = SAMPLE_HTML.replace("2020-06-15", "2025-01-01")

CONTACTS_HTML = """\
<!DOCTYPE html>
<html><head><title>Escalation Contacts</title></head>
<body>
  <div id="kbcontent">
    Networking Team: network@illinois.edu
    Help Desk: consult@illinois.edu
  </div>
</body>
</html>
"""


def _write_html(directory: Path, filename: str, content: str) -> Path:
    """Write HTML content to a file and return the path."""
    p = directory / filename
    p.write_text(content, encoding="utf-8")
    return p


def _make_mock_article(
    article_id: str = "99999",
    title: str = "Test Article",
    keywords: str = "wifi wireless",
    content: str = "Connect to IllinoisNet using SecureW2 JoinNow.",
    updated: str = "2025-01-01",
    owner: str = "Networking Team",
) -> dict:
    return {
        "filename": f"{article_id}.html",
        "article_id": article_id,
        "title": title,
        "keywords": keywords,
        "content": content,
        "internal": "",
        "updated": updated,
        "owner": owner,
    }


# ---------------------------------------------------------------------------
# _extract_doc_attr
# ---------------------------------------------------------------------------

class TestExtractDocAttr:
    def test_extracts_updated(self):
        from main import _extract_doc_attr
        soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
        assert _extract_doc_attr(soup, "updated") == "2020-06-15"

    def test_extracts_owner(self):
        from main import _extract_doc_attr
        soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
        result = _extract_doc_attr(soup, "owner")
        assert "wireless G." in result
        assert "Technology Services" in result

    def test_returns_empty_for_missing_class(self):
        from main import _extract_doc_attr
        soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
        assert _extract_doc_attr(soup, "nonexistent") == ""

    def test_returns_empty_on_bare_html(self):
        from main import _extract_doc_attr
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert _extract_doc_attr(soup, "updated") == ""


# ---------------------------------------------------------------------------
# _is_stale
# ---------------------------------------------------------------------------

class TestIsStale:
    def test_old_date_is_stale(self):
        from main import _is_stale
        assert _is_stale("2020-01-01") is True

    def test_recent_date_is_not_stale(self):
        from main import _is_stale
        assert _is_stale("2025-01-01") is False

    def test_empty_string_is_not_stale(self):
        from main import _is_stale
        assert _is_stale("") is False

    def test_invalid_date_is_not_stale(self):
        from main import _is_stale
        assert _is_stale("not-a-date") is False


# ---------------------------------------------------------------------------
# parse_article
# ---------------------------------------------------------------------------

class TestParseArticle:
    def test_extracts_title(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert result["title"] == "Test Article Title"

    def test_extracts_article_id(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert result["article_id"] == "99999"

    def test_extracts_keywords(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "wifi" in result["keywords"]
        assert "wireless" in result["keywords"]

    def test_extracts_main_content(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "main article content" in result["content"]

    def test_strips_header_footer_nav(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "Header noise" not in result["content"]
        assert "Footer noise" not in result["content"]
        assert "Nav noise" not in result["content"]

    def test_strips_feedback_buttons(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "Helpful" not in result["content"]

    def test_extracts_internal_section(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "Internal staff notes" in result["internal"]

    def test_internal_section_not_in_content(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "Internal staff notes" not in result["content"]

    def test_extracts_updated_date(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert result["updated"] == "2020-06-15"

    def test_extracts_owner(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "wireless G." in result["owner"]

    def test_doc_attr_not_in_content(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "article.html", SAMPLE_HTML)
        result = parse_article(p)
        assert "2020-06-15" not in result["content"]

    def test_filename_set(self, tmp_path):
        from main import parse_article
        p = _write_html(tmp_path, "myarticle.html", SAMPLE_HTML)
        result = parse_article(p)
        assert result["filename"] == "myarticle.html"

    def test_falls_back_to_body_when_no_kbcontent(self, tmp_path):
        from main import parse_article
        html = "<html><head><title>T</title></head><body><p>Fallback body text.</p></body></html>"
        p = _write_html(tmp_path, "fallback.html", html)
        result = parse_article(p)
        assert "Fallback body text" in result["content"]


# ---------------------------------------------------------------------------
# load_articles
# ---------------------------------------------------------------------------

class TestLoadArticles:
    def test_loads_html_files(self, tmp_path):
        from main import load_articles
        _write_html(tmp_path, "article1.html", SAMPLE_HTML)
        _write_html(tmp_path, "article2.html", FRESH_HTML)
        articles, contacts = load_articles(tmp_path)
        assert len(articles) == 2
        assert contacts == ""

    def test_separates_contacts_file(self, tmp_path):
        from main import load_articles
        _write_html(tmp_path, "article1.html", SAMPLE_HTML)
        _write_html(tmp_path, "contacts.html", CONTACTS_HTML)
        articles, contacts = load_articles(tmp_path)
        assert len(articles) == 1
        assert "network@illinois.edu" in contacts

    def test_contacts_excluded_from_articles(self, tmp_path):
        from main import load_articles
        _write_html(tmp_path, "contacts.html", CONTACTS_HTML)
        articles, _ = load_articles(tmp_path)
        assert len(articles) == 0

    def test_contacts_detection_is_case_insensitive(self, tmp_path):
        from main import load_articles
        # Use mixed-case filename with lowercase extension (glob matches *.html on Linux)
        _write_html(tmp_path, "Contacts.html", CONTACTS_HTML)
        articles, contacts = load_articles(tmp_path)
        assert len(articles) == 0
        assert contacts != ""

    def test_empty_directory_returns_empty(self, tmp_path):
        from main import load_articles
        articles, contacts = load_articles(tmp_path)
        assert articles == []
        assert contacts == ""


# ---------------------------------------------------------------------------
# build_article_index
# ---------------------------------------------------------------------------

class TestBuildArticleIndex:
    def test_returns_vectorizer_and_matrix(self):
        from main import build_article_index
        from sklearn.feature_extraction.text import TfidfVectorizer
        articles = [_make_mock_article("1"), _make_mock_article("2", content="VPN installation guide")]
        vectorizer, matrix = build_article_index(articles)
        assert isinstance(vectorizer, TfidfVectorizer)
        assert matrix.shape[0] == 2

    def test_matrix_row_count_matches_articles(self):
        from main import build_article_index
        articles = [_make_mock_article(str(i)) for i in range(5)]
        _, matrix = build_article_index(articles)
        assert matrix.shape[0] == 5


# ---------------------------------------------------------------------------
# select_relevant_articles
# ---------------------------------------------------------------------------

class TestSelectRelevantArticles:
    def _build(self, articles):
        from main import build_article_index
        return build_article_index(articles)

    def test_returns_top_k(self):
        from main import select_relevant_articles
        articles = [
            _make_mock_article("1", content="wifi wireless network IllinoisNet"),
            _make_mock_article("2", content="VPN Cisco AnyConnect installation"),
            _make_mock_article("3", content="password reset NetID"),
            _make_mock_article("4", content="email forwarding setup"),
        ]
        vectorizer, matrix = self._build(articles)
        results = select_relevant_articles("wifi connection problem", articles, vectorizer, matrix, top_k=2)
        assert len(results) == 2

    def test_most_relevant_article_ranked_first(self):
        from main import select_relevant_articles
        articles = [
            _make_mock_article("1", content="VPN Cisco installation Windows macOS"),
            _make_mock_article("2", content="wifi wireless IllinoisNet SecureW2 network"),
        ]
        vectorizer, matrix = self._build(articles)
        results = select_relevant_articles("wifi IllinoisNet", articles, vectorizer, matrix, top_k=2)
        assert results[0]["article_id"] == "2"

    def test_fallback_returns_all_when_no_match(self):
        from main import select_relevant_articles
        articles = [_make_mock_article("1"), _make_mock_article("2")]
        vectorizer, matrix = self._build(articles)
        # Query of only stop words produces zero vector -> fallback
        results = select_relevant_articles("the and or", articles, vectorizer, matrix)
        assert len(results) == len(articles)

    def test_top_k_clamped_to_article_count(self):
        from main import select_relevant_articles
        articles = [_make_mock_article("1")]
        vectorizer, matrix = self._build(articles)
        results = select_relevant_articles("wifi", articles, vectorizer, matrix, top_k=10)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_contains_system_prompt_header(self):
        from main import build_system_prompt, SYSTEM_PROMPT_HEADER
        result = build_system_prompt([_make_mock_article()])
        assert SYSTEM_PROMPT_HEADER in result

    def test_contains_article_content(self):
        from main import build_system_prompt
        article = _make_mock_article(content="Connect using SecureW2.")
        result = build_system_prompt([article])
        assert "Connect using SecureW2." in result

    def test_contains_article_id_and_url(self):
        from main import build_system_prompt, KB_BASE_URL
        article = _make_mock_article(article_id="90275")
        result = build_system_prompt([article])
        assert "90275" in result
        assert KB_BASE_URL in result

    def test_contains_updated_date(self):
        from main import build_system_prompt
        article = _make_mock_article(updated="2025-01-01")
        result = build_system_prompt([article])
        assert "2025-01-01" in result

    def test_stale_article_flagged(self):
        from main import build_system_prompt
        article = _make_mock_article(updated="2020-01-01")
        result = build_system_prompt([article])
        assert "POTENTIALLY STALE" in result

    def test_fresh_article_not_flagged(self):
        from main import build_system_prompt
        article = _make_mock_article(updated="2025-01-01")
        result = build_system_prompt([article])
        assert "POTENTIALLY STALE" not in result

    def test_contacts_block_appended_when_present(self):
        from main import build_system_prompt
        result = build_system_prompt([_make_mock_article()], contacts_text="network@illinois.edu")
        assert "ESCALATION CONTACTS" in result
        assert "network@illinois.edu" in result

    def test_contacts_block_absent_when_empty(self):
        from main import build_system_prompt
        result = build_system_prompt([_make_mock_article()], contacts_text="")
        # Check the section separator specifically — the header mentions "ESCALATION CONTACTS"
        # as a reference, but the actual block only appears when contacts_text is non-empty.
        assert "--- [ESCALATION CONTACTS] ---" not in result

    def test_multiple_articles_all_included(self):
        from main import build_system_prompt
        articles = [
            _make_mock_article("1", content="Article one content"),
            _make_mock_article("2", content="Article two content"),
        ]
        result = build_system_prompt(articles)
        assert "Article one content" in result
        assert "Article two content" in result


# ---------------------------------------------------------------------------
# _format_article_for_tool
# ---------------------------------------------------------------------------

class TestFormatArticleForTool:
    def test_contains_article_id(self):
        from main import _format_article_for_tool
        article = _make_mock_article(article_id="12345")
        result = _format_article_for_tool(article)
        assert "12345" in result

    def test_contains_content(self):
        from main import _format_article_for_tool
        article = _make_mock_article(content="Step one. Step two.")
        result = _format_article_for_tool(article)
        assert "Step one. Step two." in result

    def test_stale_flag_present_for_old_article(self):
        from main import _format_article_for_tool
        article = _make_mock_article(updated="2020-01-01")
        result = _format_article_for_tool(article)
        assert "POTENTIALLY STALE" in result

    def test_no_stale_flag_for_fresh_article(self):
        from main import _format_article_for_tool
        article = _make_mock_article(updated="2025-01-01")
        result = _format_article_for_tool(article)
        assert "POTENTIALLY STALE" not in result

    def test_owner_included(self):
        from main import _format_article_for_tool
        article = _make_mock_article(owner="Networking Team")
        result = _format_article_for_tool(article)
        assert "Networking Team" in result

    def test_falls_back_to_filename_when_no_id(self):
        from main import _format_article_for_tool
        article = _make_mock_article(article_id="")
        article["filename"] = "my_article.html"
        result = _format_article_for_tool(article)
        assert "my_article.html" in result


# ---------------------------------------------------------------------------
# handle_tool_call
# ---------------------------------------------------------------------------

class TestHandleToolCall:
    def _articles_and_index(self):
        from main import build_article_index
        articles = [
            _make_mock_article("1", content="wifi wireless IllinoisNet network SecureW2"),
            _make_mock_article("2", content="VPN Cisco AnyConnect installation guide"),
        ]
        vectorizer, matrix = build_article_index(articles)
        return articles, vectorizer, matrix

    def _make_tool_block(self, name: str, input_data: dict, tool_id: str = "tool_1"):
        tb = MagicMock()
        tb.name = name
        tb.input = input_data
        tb.id = tool_id
        return tb

    def test_search_articles_returns_content(self):
        from main import handle_tool_call
        articles, vectorizer, matrix = self._articles_and_index()
        tb = self._make_tool_block("search_articles", {"query": "wifi network"})
        result = handle_tool_call(tb, articles, vectorizer, matrix)
        assert "IllinoisNet" in result

    def test_search_articles_no_results_message(self):
        from main import handle_tool_call, build_article_index
        # Single article with content that won't match
        articles = [_make_mock_article("1", content="the and or")]
        vectorizer, matrix = build_article_index(articles)
        # TF-IDF with only stop words will zero-vector and fall back to all articles,
        # so test with a genuinely empty corpus response by mocking select_relevant_articles
        with patch("main.select_relevant_articles", return_value=[]):
            tb = self._make_tool_block("search_articles", {"query": "wifi"})
            result = handle_tool_call(tb, articles, vectorizer, matrix)
        assert "No relevant articles found" in result

    def test_get_article_by_id_returns_content(self):
        from main import handle_tool_call
        articles, vectorizer, matrix = self._articles_and_index()
        tb = self._make_tool_block("get_article", {"article_id": "1"})
        result = handle_tool_call(tb, articles, vectorizer, matrix)
        assert "IllinoisNet" in result

    def test_get_article_missing_id_returns_not_found(self):
        from main import handle_tool_call
        articles, vectorizer, matrix = self._articles_and_index()
        tb = self._make_tool_block("get_article", {"article_id": "99999"})
        result = handle_tool_call(tb, articles, vectorizer, matrix)
        assert "not found" in result.lower()

    def test_unknown_tool_returns_error_string(self):
        from main import handle_tool_call
        articles, vectorizer, matrix = self._articles_and_index()
        tb = self._make_tool_block("unknown_tool", {})
        result = handle_tool_call(tb, articles, vectorizer, matrix)
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------

class TestRunAgent:
    def _articles_and_index(self):
        from main import build_article_index
        articles = [_make_mock_article("1", content="Connect to wifi using SecureW2.")]
        vectorizer, matrix = build_article_index(articles)
        return articles, vectorizer, matrix

    def _make_text_response(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        return response

    def _make_tool_response(self, tool_name: str, tool_input: dict, tool_id: str = "t1"):
        tb = MagicMock()
        tb.type = "tool_use"
        tb.name = tool_name
        tb.input = tool_input
        tb.id = tool_id
        response = MagicMock()
        response.content = [tb]
        return response

    def test_returns_text_on_direct_response(self):
        from main import run_agent
        articles, vectorizer, matrix = self._articles_and_index()
        client = MagicMock()
        client.messages.create.return_value = self._make_text_response("Here are the steps.")
        result = run_agent(client, "system", [{"role": "user", "content": "wifi issue"}], articles, vectorizer, matrix)
        assert result == "Here are the steps."

    def test_handles_tool_call_then_text(self):
        from main import run_agent
        articles, vectorizer, matrix = self._articles_and_index()
        client = MagicMock()
        client.messages.create.side_effect = [
            self._make_tool_response("search_articles", {"query": "wifi"}),
            self._make_text_response("Based on the articles, here are the steps."),
        ]
        result = run_agent(client, "system", [{"role": "user", "content": "wifi issue"}], articles, vectorizer, matrix)
        assert result == "Based on the articles, here are the steps."
        assert client.messages.create.call_count == 2

    def test_returns_none_when_no_text_block(self):
        from main import run_agent
        articles, vectorizer, matrix = self._articles_and_index()
        client = MagicMock()
        empty_response = MagicMock()
        empty_response.content = []
        client.messages.create.return_value = empty_response
        result = run_agent(client, "system", [{"role": "user", "content": "issue"}], articles, vectorizer, matrix)
        assert result is None

    def test_session_conversation_not_mutated(self):
        from main import run_agent
        articles, vectorizer, matrix = self._articles_and_index()
        client = MagicMock()
        client.messages.create.return_value = self._make_text_response("Answer.")
        conversation = [{"role": "user", "content": "question"}]
        original_len = len(conversation)
        run_agent(client, "system", conversation, articles, vectorizer, matrix)
        assert len(conversation) == original_len


# ---------------------------------------------------------------------------
# create_app (Flask routes)
# ---------------------------------------------------------------------------

class TestCreateApp:
    def _make_app(self, assistant_response: str = "Troubleshooting steps here."):
        from main import create_app, build_article_index
        articles = [_make_mock_article("1", content="wifi SecureW2 IllinoisNet")]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = assistant_response
        mock_response = MagicMock()
        mock_response.content = [text_block]
        client.messages.create.return_value = mock_response

        app = create_app(client, articles, vectorizer, matrix, contacts_text="")
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret"
        return app

    def test_get_index_returns_200(self):
        app = self._make_app()
        with app.test_client() as c:
            response = c.get("/")
            assert response.status_code == 200

    def test_post_adds_user_and_assistant_to_conversation(self):
        app = self._make_app("Here are your steps.")
        with app.test_client() as c:
            response = c.post("/", data={"issue": "My wifi is not working"})
            assert response.status_code == 200
            data = response.data.decode()
            assert "My wifi is not working" in data
            assert "Here are your steps." in data

    def test_empty_post_does_not_add_to_conversation(self):
        app = self._make_app()
        with app.test_client() as c:
            c.post("/", data={"issue": "   "})
            # Session conversation should remain empty
            with c.session_transaction() as sess:
                assert sess.get("conversation", []) == []

    def test_clear_resets_conversation(self):
        app = self._make_app("Answer.")
        with app.test_client() as c:
            c.post("/", data={"issue": "wifi issue"})
            c.get("/clear")
            response = c.get("/")
            data = response.data.decode()
            assert "wifi issue" not in data

    def test_rate_limit_error_shows_error_message(self):
        from main import create_app, build_article_index
        import anthropic as anthropic_module
        articles = [_make_mock_article()]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        client.messages.create.side_effect = anthropic_module.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        app = create_app(client, articles, vectorizer, matrix)
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret"
        with app.test_client() as c:
            response = c.post("/", data={"issue": "wifi issue"})
            data = response.data.decode()
            assert "Rate limit" in data or "rate limit" in data.lower()
