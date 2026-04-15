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
# build number / CLI flag
# ---------------------------------------------------------------------------

class TestBuildNumber:
    def test_prefers_build_number_env(self, monkeypatch):
        from main import get_build_number

        monkeypatch.setenv("BUILD_NUMBER", "20260415.7")
        monkeypatch.setenv("GITHUB_SHA", "abcdef123456")
        assert get_build_number() == "20260415.7"

    def test_falls_back_to_github_sha(self, monkeypatch):
        from main import get_build_number

        monkeypatch.delenv("BUILD_NUMBER", raising=False)
        monkeypatch.setenv("GITHUB_SHA", "abcdef123456")
        assert get_build_number() == "abcdef1"

    def test_falls_back_to_git_commit(self, monkeypatch):
        from main import get_build_number

        monkeypatch.delenv("BUILD_NUMBER", raising=False)
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        with patch("main.subprocess.check_output", return_value="1a2b3c4\n"):
            assert get_build_number() == "1a2b3c4"

    def test_returns_unknown_when_commit_unavailable(self, monkeypatch):
        from main import get_build_number

        monkeypatch.delenv("BUILD_NUMBER", raising=False)
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        with patch("main.subprocess.check_output", side_effect=OSError):
            assert get_build_number() == "unknown"


class TestMainBuildNumberFlag:
    def test_build_number_flag_prints_and_returns_early(self):
        import main as main_module

        args = type("Args", (), {"build_number": True})()
        with (
            patch.object(main_module, "_parse_args", return_value=args),
            patch.object(main_module, "get_build_number", return_value="build-42"),
            patch("builtins.print") as mock_print,
            patch.object(main_module, "load_dotenv") as mock_load_dotenv,
        ):
            main_module.main()

        mock_print.assert_called_once_with("build-42")
        mock_load_dotenv.assert_not_called()


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
        second = (
            FRESH_HTML
            .replace("resultc.php?action=7&amp;id=99999", "resultc.php?action=7&amp;id=99998")
            .replace("<div class=\"doc-attr-value\">99999</div>", "<div class=\"doc-attr-value\">99998</div>")
        )
        _write_html(tmp_path, "article2.html", second)
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

    def test_deduplicates_same_article_id_across_htm_and_html(self, tmp_path):
        from main import load_articles

        html_htm = SAMPLE_HTML.replace("Test Article Title", "Test Article HTM")
        html_html = SAMPLE_HTML.replace("Test Article Title", "Test Article HTML")

        _write_html(tmp_path, "dup_article.htm", html_htm)
        _write_html(tmp_path, "dup_article.html", html_html)

        articles, _ = load_articles(tmp_path)
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article HTML"


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
# select_display_articles
# ---------------------------------------------------------------------------

class TestSelectDisplayArticles:
    def _build(self, articles):
        from main import build_article_index
        return build_article_index(articles)

    def test_returns_tuples_of_article_and_score(self):
        from main import select_display_articles
        articles = [
            _make_mock_article("1", content="wifi wireless network IllinoisNet"),
            _make_mock_article("2", content="VPN Cisco AnyConnect installation"),
        ]
        vectorizer, matrix = self._build(articles)
        results = select_display_articles("wifi connection", articles, vectorizer, matrix, display_k=2)
        assert len(results) == 2
        for article, score in results:
            assert isinstance(article, dict)
            assert isinstance(score, float)

    def test_most_relevant_ranked_first(self):
        from main import select_display_articles
        articles = [
            _make_mock_article("1", content="VPN Cisco installation Windows macOS"),
            _make_mock_article("2", content="wifi wireless IllinoisNet SecureW2 network"),
        ]
        vectorizer, matrix = self._build(articles)
        results = select_display_articles("wifi IllinoisNet", articles, vectorizer, matrix, display_k=2)
        assert results[0][0]["article_id"] == "2"

    def test_scores_descending(self):
        from main import select_display_articles
        articles = [
            _make_mock_article("1", content="wifi wireless network IllinoisNet"),
            _make_mock_article("2", content="VPN Cisco AnyConnect installation"),
            _make_mock_article("3", content="password reset NetID account"),
        ]
        vectorizer, matrix = self._build(articles)
        results = select_display_articles("wifi IllinoisNet", articles, vectorizer, matrix, display_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_display_k_limits_results(self):
        from main import select_display_articles
        articles = [_make_mock_article(str(i), content=f"article {i} content topic") for i in range(6)]
        vectorizer, matrix = self._build(articles)
        results = select_display_articles("content topic", articles, vectorizer, matrix, display_k=3)
        assert len(results) == 3

    def test_fallback_on_zero_vector(self):
        from main import select_display_articles
        articles = [_make_mock_article("1"), _make_mock_article("2")]
        vectorizer, matrix = self._build(articles)
        results = select_display_articles("the and or", articles, vectorizer, matrix, display_k=5)
        assert all(score == 0.0 for _, score in results)


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

    def test_post_shows_query_and_ai_response(self):
        app = self._make_app("Here are your steps.")
        with app.test_client() as c:
            response = c.post("/", data={"query": "My wifi is not working"})
            assert response.status_code == 200
            data = response.data.decode()
            assert "My wifi is not working" in data
            assert "Here are your steps." in data

    def test_empty_post_shows_empty_state(self):
        app = self._make_app()
        with app.test_client() as c:
            response = c.post("/", data={"query": "   "})
            assert response.status_code == 200
            data = response.data.decode()
            # No results shown — left panel should show the empty-state placeholder
            assert "Enter a support question above to get started" in data

    def test_post_missing_query_field_shows_empty_state(self):
        """A POST with no 'query' field must not show any results."""
        app = self._make_app()
        with app.test_client() as c:
            response = c.post("/", data={})
            assert response.status_code == 200
            data = response.data.decode()
            assert "Enter a support question above to get started" in data

    def test_get_shows_empty_panels(self):
        app = self._make_app()
        with app.test_client() as c:
            response = c.get("/")
            data = response.data.decode()
            assert "Enter a support question above to get started" in data

    def test_refine_returns_200(self):
        app = self._make_app("Refined answer.")
        with app.test_client() as c:
            response = c.post("/refine", data={
                "original_query": "wifi issue",
                "refinement": "user is on macOS",
            })
            assert response.status_code == 200

    def test_refine_shows_combined_response(self):
        app = self._make_app("Refined answer.")
        with app.test_client() as c:
            response = c.post("/refine", data={
                "original_query": "wifi issue",
                "refinement": "user is on macOS",
            })
            data = response.data.decode()
            assert "Refined answer." in data

    def test_refine_preserves_original_query_in_search_bar(self):
        app = self._make_app("Refined answer.")
        with app.test_client() as c:
            response = c.post("/refine", data={
                "original_query": "wifi issue",
                "refinement": "user is on macOS",
            })
            data = response.data.decode()
            assert "wifi issue" in data

    def test_refine_missing_original_query_shows_empty_state(self):
        app = self._make_app()
        with app.test_client() as c:
            response = c.post("/refine", data={"refinement": "extra context"})
            assert response.status_code == 200
            data = response.data.decode()
            assert "Enter a support question above to get started" in data

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
            response = c.post("/", data={"query": "wifi issue"})
            data = response.data.decode()
            assert "Rate limit" in data or "rate limit" in data.lower()


# ---------------------------------------------------------------------------
# GET /search  (JSON endpoint)
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    def _make_app(self):
        from main import create_app, build_article_index
        articles = [
            _make_mock_article("1", content="wifi wireless IllinoisNet SecureW2", keywords="wifi wireless"),
            _make_mock_article("2", content="VPN Cisco AnyConnect installation", keywords="vpn cisco"),
        ]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        app = create_app(client, articles, vectorizer, matrix)
        app.config["TESTING"] = True
        return app

    def test_empty_query_returns_empty_list(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search?q=")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["articles"] == []
            assert data["count"] == 0

    def test_missing_query_param_returns_empty_list(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] == 0

    def test_matching_query_returns_articles(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search?q=wifi+wireless")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] > 0
            assert any(a["article_id"] == "1" for a in data["articles"])

    def test_article_fields_present(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search?q=wifi")
            data = resp.get_json()
            assert data["count"] > 0
            article = data["articles"][0]
            for field in ("article_id", "title", "owner", "updated", "is_stale",
                          "keywords", "excerpt", "excerpt_truncated",
                          "badge_label", "badge_class", "in_ai", "url"):
                assert field in article, f"Missing field: {field}"

    def test_in_ai_flag_true_for_first_top_k(self):
        from main import TOP_K_ARTICLES
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search?q=wifi")
            data = resp.get_json()
            # First TOP_K_ARTICLES results should have in_ai=True
            for i, a in enumerate(data["articles"]):
                if i < TOP_K_ARTICLES:
                    assert a["in_ai"] is True
                else:
                    assert a["in_ai"] is False

    def test_badge_label_is_valid(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search?q=wifi")
            data = resp.get_json()
            for a in data["articles"]:
                assert a["badge_label"] in ("High", "Medium", "Low")

    def test_top_result_promoted_to_high_when_gap_is_clear(self):
        from main import create_app, build_article_index

        articles = [
            _make_mock_article("1", title="Duo Enrollment", content="duo enrollment setup"),
            _make_mock_article("2", title="Duo Troubleshooting", content="duo troubleshooting"),
        ]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        app = create_app(client, articles, vectorizer, matrix)
        app.config["TESTING"] = True

        mocked_display = [
            (articles[0], 0.18),
            (articles[1], 0.14),
        ]
        with patch("main.select_display_articles", return_value=mocked_display):
            with app.test_client() as c:
                resp = c.get("/search?q=duo+enrollment")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["articles"][0]["badge_label"] == "High"
        assert data["articles"][0]["badge_class"] == "badge-high"

    def test_top_result_not_promoted_without_overlap(self):
        from main import create_app, build_article_index

        articles = [
            _make_mock_article("1", title="VPN Setup", keywords="vpn cisco", content="vpn install"),
            _make_mock_article("2", title="Duo Enrollment", keywords="duo mfa", content="duo setup"),
        ]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        app = create_app(client, articles, vectorizer, matrix)
        app.config["TESTING"] = True

        mocked_display = [
            (articles[0], 0.18),
            (articles[1], 0.14),
        ]
        with patch("main.select_display_articles", return_value=mocked_display):
            with app.test_client() as c:
                resp = c.get("/search?q=duo+enrollment")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["articles"][0]["badge_label"] == "Medium"
        assert data["articles"][0]["badge_class"] == "badge-medium"

    def test_returns_json_content_type(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/search?q=wifi")
            assert "application/json" in resp.content_type


# ---------------------------------------------------------------------------
# POST /ai  (JSON endpoint)
# ---------------------------------------------------------------------------

class TestAiEndpoint:
    def _make_app(self, assistant_response: str = "AI troubleshooting answer."):
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
        app = create_app(client, articles, vectorizer, matrix)
        app.config["TESTING"] = True
        return app

    def test_missing_query_returns_error(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.post("/ai", data={})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["error"]
            assert data["response"] is None

    def test_valid_query_returns_response(self):
        app = self._make_app("Here are the troubleshooting steps.")
        with app.test_client() as c:
            resp = c.post("/ai", data={"query": "wifi issue"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["response"] == "Here are the troubleshooting steps."
            assert data["error"] == ""

    def test_refinement_combined_into_prompt(self):
        """The Claude client must be called once and receive the combined query."""
        from main import create_app, build_article_index
        articles = [_make_mock_article("1", content="wifi IllinoisNet")]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Refined steps."
        mock_resp = MagicMock()
        mock_resp.content = [text_block]
        client.messages.create.return_value = mock_resp
        app = create_app(client, articles, vectorizer, matrix)
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.post("/ai", data={"query": "wifi issue", "refinement": "user is on macOS"})
            data = resp.get_json()
            assert data["response"] == "Refined steps."
        # Verify the prompt sent to Claude contains both parts
        _, call_kwargs = client.messages.create.call_args
        messages = call_kwargs["messages"]
        combined = messages[0]["content"]
        assert "wifi issue" in combined
        assert "user is on macOS" in combined

    def test_footer_contains_article_count(self):
        app = self._make_app("Steps here.")
        with app.test_client() as c:
            resp = c.post("/ai", data={"query": "wifi"})
            data = resp.get_json()
            assert "article" in data["footer"]

    def test_rate_limit_error_returns_json_error(self):
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
        with app.test_client() as c:
            resp = c.post("/ai", data={"query": "wifi issue"})
            data = resp.get_json()
            assert "rate limit" in data["error"].lower()
            assert data["response"] is None

    def test_returns_json_content_type(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.post("/ai", data={"query": "wifi"})
            assert "application/json" in resp.content_type


# ---------------------------------------------------------------------------
# GET /brave  (Brave Search JSON endpoint)
# ---------------------------------------------------------------------------

class TestBraveEndpoint:
    def _make_app(self, brave_api_key: str = ""):
        from main import create_app, build_article_index
        articles = [_make_mock_article("1", content="wifi wireless IllinoisNet")]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        app = create_app(client, articles, vectorizer, matrix, brave_api_key=brave_api_key)
        app.config["TESTING"] = True
        return app

    def test_no_api_key_returns_unavailable(self):
        app = self._make_app(brave_api_key="")
        with app.test_client() as c:
            resp = c.get("/brave?q=wifi+problem")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["results"] == []
            assert data["available"] is False

    def test_empty_query_returns_empty_with_no_key(self):
        app = self._make_app(brave_api_key="")
        with app.test_client() as c:
            resp = c.get("/brave?q=")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["results"] == []
            assert data["available"] is False

    def test_empty_query_with_key_returns_empty(self):
        app = self._make_app(brave_api_key="test-key")
        with app.test_client() as c:
            resp = c.get("/brave?q=")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["results"] == []

    def test_with_api_key_calls_fetch_brave_results(self):
        from unittest.mock import patch
        app = self._make_app(brave_api_key="test-key")
        mock_results = [
            {
                "title": "Someone with a similar issue",
                "url": "https://example.com/forum",
                "description": "Describes the same wifi problem.",
            }
        ]
        with patch("main.fetch_brave_results", return_value=mock_results):
            with app.test_client() as c:
                resp = c.get("/brave?q=wifi+not+working")
                data = resp.get_json()
                assert data["available"] is True
                assert len(data["results"]) == 1
                assert data["results"][0]["title"] == "Someone with a similar issue"

    def test_returns_json_content_type(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/brave?q=wifi")
            assert "application/json" in resp.content_type

    def test_brave_section_in_html_when_key_set(self):
        """When brave_api_key is set, the template renders the brave-section div."""
        from main import create_app, build_article_index
        articles = [_make_mock_article("1", content="wifi wireless")]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        app = create_app(client, articles, vectorizer, matrix, brave_api_key="test-key")
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.get("/")
            body = resp.data.decode()
            assert 'id="brave-section"' in body
            assert "Users with Similar Issues" in body

    def test_brave_section_absent_when_no_key(self):
        """Without a brave_api_key the template does not render the brave-section div."""
        from main import create_app, build_article_index
        articles = [_make_mock_article("1", content="wifi wireless")]
        vectorizer, matrix = build_article_index(articles)
        client = MagicMock()
        app = create_app(client, articles, vectorizer, matrix, brave_api_key="")
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.get("/")
            body = resp.data.decode()
            # The div should not be rendered (the JS string literal is still present)
            assert 'id="brave-section"' not in body


# ---------------------------------------------------------------------------
# fetch_brave_results  (unit tests)
# ---------------------------------------------------------------------------

class TestFetchBraveResults:
    def test_returns_empty_list_on_url_error(self):
        import urllib.error
        from unittest.mock import patch
        from main import fetch_brave_results

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            results = fetch_brave_results("test query", "test-key")
        assert results == []

    def test_returns_empty_list_on_os_error(self):
        from unittest.mock import patch
        from main import fetch_brave_results

        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            results = fetch_brave_results("test query", "test-key")
        assert results == []

    def test_parses_web_results(self):
        import json as json_mod
        from unittest.mock import MagicMock, patch
        from main import fetch_brave_results

        payload = {
            "web": {
                "results": [
                    {"title": "Title A", "url": "https://a.com", "description": "Desc A"},
                    {"title": "Title B", "url": "https://b.com", "description": ""},
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json_mod.dumps(payload).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = fetch_brave_results("test query", "test-key")

        assert len(results) == 2
        assert results[0] == {"title": "Title A", "url": "https://a.com", "description": "Desc A"}
        assert results[1] == {"title": "Title B", "url": "https://b.com", "description": ""}

    def test_returns_empty_list_for_missing_web_key(self):
        import json as json_mod
        from unittest.mock import MagicMock, patch
        from main import fetch_brave_results

        mock_resp = MagicMock()
        mock_resp.read.return_value = json_mod.dumps({}).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = fetch_brave_results("test query", "test-key")
        assert results == []

    def test_respects_count_parameter(self):
        """The count query parameter should be forwarded to the Brave API URL."""
        import json as json_mod
        from unittest.mock import MagicMock, call, patch
        from main import fetch_brave_results

        mock_resp = MagicMock()
        mock_resp.read.return_value = json_mod.dumps({"web": {"results": []}}).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            fetch_brave_results("wifi problem", "test-key", count=3)
            args, _ = mock_open.call_args
            req = args[0]
            assert "count=3" in req.full_url


# ---------------------------------------------------------------------------
# GET /articles  (articles listing page)
# ---------------------------------------------------------------------------

class TestArticlesEndpoint:
    def _make_app(self, article_list=None):
        from main import create_app, build_article_index
        if article_list is None:
            article_list = [
                _make_mock_article("1", title="Campus Wi-Fi Guide", keywords="wifi wireless"),
                _make_mock_article("2", title="VPN Setup", keywords="vpn cisco", updated="2020-01-01"),
                _make_mock_article("3", title="MFA Troubleshooting", keywords="mfa"),
            ]
        vectorizer, matrix = build_article_index(article_list)
        client = MagicMock()
        app = create_app(client, article_list, vectorizer, matrix)
        app.config["TESTING"] = True
        return app

    def test_returns_200(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            assert resp.status_code == 200

    def test_returns_html_content_type(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            assert "text/html" in resp.content_type

    def test_all_article_titles_present(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            body = resp.data.decode()
            assert "Campus Wi-Fi Guide" in body
            assert "VPN Setup" in body
            assert "MFA Troubleshooting" in body

    def test_article_ids_present(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            body = resp.data.decode()
            # Article IDs appear in the id-cell table column
            assert 'class="col-id id-cell">1<' in body
            assert 'class="col-id id-cell">2<' in body
            assert 'class="col-id id-cell">3<' in body

    def test_stale_badge_shown_for_old_article(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            body = resp.data.decode()
            # Article "2" has updated="2020-01-01" which is stale
            assert "stale-badge" in body

    def test_article_count_in_header(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            body = resp.data.decode()
            assert "3 articles indexed" in body

    def test_empty_articles_list(self):
        from main import create_app
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Use a dummy vectorizer that won't be called for an empty article list
        vectorizer = MagicMock(spec=TfidfVectorizer)
        vectorizer.transform.return_value = np.zeros((0, 1))
        matrix = np.zeros((0, 1))
        client = MagicMock()
        app = create_app(client, [], vectorizer, matrix)
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.get("/articles")
            assert resp.status_code == 200
            body = resp.data.decode()
            assert "No articles are currently indexed" in body

    def test_articles_sorted_alphabetically(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            body = resp.data.decode()
            # "Campus Wi-Fi Guide" comes before "MFA", which comes before "VPN"
            campus_pos = body.index("Campus Wi-Fi Guide")
            mfa_pos = body.index("MFA Troubleshooting")
            vpn_pos = body.index("VPN Setup")
            assert campus_pos < mfa_pos < vpn_pos

    def test_back_to_search_link_present(self):
        app = self._make_app()
        with app.test_client() as c:
            resp = c.get("/articles")
            body = resp.data.decode()
            assert 'href="/"' in body
