"""
KB AI Search Agent
==================
A locally-run web app where a help desk consultant describes a support problem
in plain English and receives relevant troubleshooting steps drawn from local KB
articles. TF-IDF pre-filtering is used to select only the most relevant articles
for each query before sending them to Claude, reducing context size and cost.

Search intelligence upgrades are available via feature flags (see search_enhancement.py):
  FEATURE_HYBRID_RETRIEVAL      – TF-IDF + semantic (LSA/sentence-transformers) + RRF fusion
  FEATURE_QUERY_NORMALIZATION   – synonym expansion, optional typo correction
  FEATURE_ADAPTIVE_TOPK         – score-gap based adaptive top-k for Claude context
  FEATURE_SEARCH_CACHE          – LRU result cache with TTL

All flags default to False so the current behaviour is preserved unless opted in.
"""

from __future__ import annotations

import json
import logging
import os
import argparse
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
from flask import Flask, jsonify, render_template_string, request

# Optional search-enhancement module – imported at module level so tests can
# monkeypatch its FeatureFlags.  Falls back gracefully if the file is missing.
try:
    from search_enhancement import (
        FeatureFlags,
        HybridRetriever,
        SearchTimings,
        adaptive_top_k,
    )
    _SEARCH_ENHANCEMENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SEARCH_ENHANCEMENT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTICLES_DIR = Path(__file__).parent / "articles"
CONTACTS_FILENAME = "contacts.html"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

# Guardrail: limit tool-loop turns so a single user request cannot spiral into
# many Claude API calls when tool_use chains become long.
MAX_AGENT_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "3"))

# Soft prompt budget controls to reduce input token spend.
MAX_ARTICLE_CHARS_IN_PROMPT = int(os.environ.get("MAX_ARTICLE_CHARS_IN_PROMPT", "2500"))
MAX_CONTACTS_CHARS_IN_PROMPT = int(os.environ.get("MAX_CONTACTS_CHARS_IN_PROMPT", "1500"))

# Maximum number of articles sent to Claude per query.
# Raising this improves recall at the cost of larger context windows.
TOP_K_ARTICLES = max(1, int(os.environ.get("TOP_K_ARTICLES", "3")))

# Number of articles shown in the left search-results panel (independent of TOP_K_ARTICLES).
DISPLAY_K_ARTICLES = 10

KB_BASE_URL = "https://answers.uillinois.edu/illinois/internal"

# Articles last updated more than this many years ago are flagged as potentially stale.
STALE_ARTICLE_YEARS = 2

# TF-IDF cosine similarity thresholds for relevancy labels shown in the left panel.
RELEVANCE_HIGH = 0.20
RELEVANCE_MEDIUM = 0.08

# Brave Search integration: show "Users with Similar Issues" box when fewer than
# BRAVE_MIN_HIGH_CONF articles receive a "High" relevancy badge.
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_MIN_HIGH_CONF = 1   # threshold (exclusive) of High-confidence KB articles
BRAVE_RESULT_COUNT = 5    # number of Brave Search results to request

SYSTEM_PROMPT_HEADER = (
    "You are a help desk assistant for University of Illinois Technology Services.\n"
    "Resolve support issues using ONLY the Knowledge Base articles provided. Do not use outside knowledge.\n"
    "\n"
    "## Response Format\n"
    "Structure every response with these sections:\n"
    "\n"
    "**Issue:** One sentence identifying the problem.\n"
    "\n"
    "**Steps:**\n"
    "1. First step. [Article XXXXX](url)\n"
    "2. Second step. [Article XXXXX](url)\n"
    "\n"
    "**If Unresolved:** Ask 1-2 targeted clarifying questions before suggesting escalation.\n"
    "\n"
    "## Rules\n"
    "- Use only information from the provided KB articles. Do not add general knowledge.\n"
    "- Use plain language, but preserve exact names of systems, tools, buttons, and settings as written in the articles.\n"
    "- Each response is independent; do not assume any prior context.\n"
    f"- If an article was last updated more than {STALE_ARTICLE_YEARS} years ago, note it inline "
    "(e.g., 'Note: this article was last updated in YYYY - verify these steps are still current.').\n"
    "- If articles conflict, prefer the more recently updated one and note the discrepancy.\n"
    "- If the provided articles lack enough information, use the available tools to search for or "
    "fetch additional articles before responding.\n"
    "\n"
    "## Escalation\n"
    "Only suggest escalation as a last resort - after clarifying questions have not resolved the issue.\n"
    "Use the [ESCALATION CONTACTS] section at the end of this prompt to route appropriately.\n"
)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_articles",
        "description": (
            "Search the local Knowledge Base for articles relevant to a query. "
            "Use this when the pre-loaded articles don't contain enough information to answer the issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or a description of what information you need.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_article",
        "description": (
            "Fetch a specific KB article by its numeric article ID. "
            "Use this when the KB content references a specific article ID that you don't already have."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "The numeric article ID (e.g. '90275').",
                }
            },
            "required": ["article_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# HTML Parsing
# ---------------------------------------------------------------------------

def _extract_doc_attr(soup: BeautifulSoup, attr_class: str) -> str:
    """Extract the value text from a doc-attr div with a specific secondary class."""
    for div in soup.find_all("div", class_="doc-attr"):
        if attr_class in div.get("class", []):
            value_div = div.find("div", class_="doc-attr-value")
            if value_div:
                return value_div.get_text(strip=True)
    return ""


def _is_stale(updated: str) -> bool:
    """Return True if the article's updated date is older than STALE_ARTICLE_YEARS."""
    if not updated:
        return False
    try:
        updated_date = date.fromisoformat(updated)
        return (date.today().year - updated_date.year) > STALE_ARTICLE_YEARS
    except ValueError:
        return False


def parse_article(path: Path) -> dict[str, str]:
    """Parse a UW-Madison-style KB HTML article and return a dict with:
       - filename:   base name of the file
       - article_id: numeric KB article ID
       - title:      page title
       - keywords:   keywords string
       - content:    cleaned plain-text main content
       - internal:   internal-staff section text (may be empty)
       - updated:    last-updated date string (e.g. '2025-02-04'), may be empty
       - owner:      owning team/group string, may be empty
    """
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # 1. Extract metadata from doc-attr BEFORE stripping it
    updated = _extract_doc_attr(soup, "updated")
    owner = _extract_doc_attr(soup, "owner")

    # 2. Strip noise elements before any text extraction
    for tag in soup.find_all(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    for tag in soup.find_all(class_="doc-attr"):
        tag.decompose()
    # Feedback / analytics buttons
    for tag in soup.find_all(class_="feedback-btn"):
        tag.decompose()

    # 2. Extract internal-staff section (then remove it from the tree)
    internal_text = ""
    internal_div = soup.find("div", class_="kb-class-internal-site")
    if internal_div:
        internal_text = internal_div.get_text(separator="\n", strip=True)
        internal_div.decompose()

    # 3. Extract article ID from the "Show changes" link (resultc.php?action=7&id=XXXXX)
    article_id = ""
    id_match = re.search(r'resultc\.php\?action=7&(?:amp;)?id=(\d+)', html)
    if id_match:
        article_id = id_match.group(1)

    # 4. Extract title
    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else path.stem

    # 4. Extract keywords
    keywords_span = soup.find("span", id="kb-page-keywords")
    keywords = keywords_span.text.strip() if keywords_span else ""

    # 5. Extract main content
    content_div = soup.find("div", id="kbcontent")
    if content_div:
        content = content_div.get_text(separator="\n", strip=True)
    else:
        # Fall back to body text if no kbcontent div
        body = soup.find("body")
        content = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

    return {
        "filename": path.name,
        "article_id": article_id,
        "title": title,
        "keywords": keywords,
        "content": content,
        "internal": internal_text,
        "updated": updated,
        "owner": owner,
    }


def load_articles(articles_dir: Path) -> tuple[list[dict[str, str]], str]:
    """Load and parse all .htm and .html files from the given directory.

    Returns ``(articles, contacts_text)`` where ``contacts_text`` is the parsed
    content of ``contacts.html`` (if present) and ``articles`` contains everything
    else.  The contacts file is excluded from the TF-IDF index.
    """
    articles_by_key: dict[str, dict[str, str]] = {}
    contacts_text = ""
    for ext in ("*.htm", "*.html"):
        for filepath in sorted(articles_dir.glob(ext)):
            if filepath.name.lower() == CONTACTS_FILENAME:
                contacts_text = parse_article(filepath)["content"]
            else:
                parsed = parse_article(filepath)
                article_id = parsed.get("article_id", "").strip()
                dedupe_key = article_id if article_id else parsed["filename"].lower()

                existing = articles_by_key.get(dedupe_key)
                if existing is None:
                    articles_by_key[dedupe_key] = parsed
                    continue

                # Prefer .html over .htm when both represent the same article.
                existing_is_htm = existing["filename"].lower().endswith(".htm")
                parsed_is_html = parsed["filename"].lower().endswith(".html")
                if existing_is_htm and parsed_is_html:
                    articles_by_key[dedupe_key] = parsed

    articles = list(articles_by_key.values())
    return articles, contacts_text

# ---------------------------------------------------------------------------
# TF-IDF Article Index
# ---------------------------------------------------------------------------

def build_article_index(
    articles: list[dict[str, str]],
) -> tuple[TfidfVectorizer, np.ndarray]:
    """Build a TF-IDF index over all KB articles.

    Each article is represented by its title, keywords, and content combined
    so that all three fields contribute to relevance scoring.

    Returns a ``(vectorizer, doc_matrix)`` tuple.  The document matrix is
    pre-computed here so it does not need to be recomputed on every query.
    """
    corpus = [
        f"{a['title']} {a['keywords']} {a['content']}"
        for a in articles
    ]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    doc_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, doc_matrix


def select_relevant_articles(
    query: str,
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
    top_k: int = TOP_K_ARTICLES,
) -> list[dict[str, str]]:
    """Return the *top_k* articles most relevant to *query* using TF-IDF cosine similarity.

    Falls back to returning all articles if the query produces an all-zero
    vector (e.g. it contains only stop-words not present in the corpus).
    """
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, doc_matrix).flatten()

    # If every score is zero the query had no useful terms; return all articles.
    if not np.any(scores):
        return articles

    top_indices = np.argsort(scores)[::-1][: min(top_k, len(articles))]
    return [articles[i] for i in top_indices]


def select_display_articles(
    query: str,
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
    display_k: int = DISPLAY_K_ARTICLES,
) -> list[tuple[dict[str, str], float]]:
    """Return the top *display_k* articles with their raw cosine similarity scores.

    Returns a list of ``(article, score)`` tuples ordered by relevance descending.
    Falls back to all articles (with score 0.0) when the query produces an
    all-zero vector (e.g. only stop-words).
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_matrix).flatten()

    if not np.any(scores):
        return [(a, 0.0) for a in articles[:display_k]]

    top_indices = np.argsort(scores)[::-1][: min(display_k, len(articles))]
    return [(articles[i], float(scores[i])) for i in top_indices]


def fetch_brave_results(query: str, api_key: str, count: int = BRAVE_RESULT_COUNT) -> list[dict]:
    """Query the Brave Search API and return a list of web results.

    Each result is a dict with keys: ``title``, ``url``, ``description``.
    Returns an empty list if the API call fails for any reason.
    """
    params = urllib.parse.urlencode({"q": query, "count": count})
    url = f"{BRAVE_SEARCH_API_URL}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        logging.warning("Brave Search request failed: %s", exc)
        return []

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("description", ""),
        }
        for r in data.get("web", {}).get("results", [])
    ]


def build_system_prompt(articles: list[dict[str, str]], contacts_text: str = "") -> str:
    """Construct the full system prompt with all KB article content."""
    lines = [SYSTEM_PROMPT_HEADER, "", "--- KNOWLEDGE BASE ---", ""]
    for article in articles:
        article_id = article["article_id"]
        article_url = f"{KB_BASE_URL}/{article_id}" if article_id else ""
        label = f"Article {article_id} ({article_url})" if article_id else article["filename"]
        lines.append(f"[ARTICLE: {label}]")
        if article.get("updated"):
            stale_flag = " [POTENTIALLY STALE]" if _is_stale(article["updated"]) else ""
            lines.append(f"Last updated: {article['updated']}{stale_flag}")
        if article.get("owner"):
            lines.append(f"Owner: {article['owner']}")
        if article["keywords"]:
            lines.append(f"Keywords: {article['keywords']}")
        lines.append("---")
        content = article["content"]
        if len(content) > MAX_ARTICLE_CHARS_IN_PROMPT:
            content = f"{content[:MAX_ARTICLE_CHARS_IN_PROMPT]}\n\n[TRUNCATED]"
        lines.append(content)
        lines.append("")  # blank separator between articles
    if contacts_text:
        trimmed_contacts = contacts_text[:MAX_CONTACTS_CHARS_IN_PROMPT]
        if len(contacts_text) > MAX_CONTACTS_CHARS_IN_PROMPT:
            trimmed_contacts += "\n\n[TRUNCATED]"
        lines += ["", "--- [ESCALATION CONTACTS] ---", "", trimmed_contacts, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agentic Tool Handling
# ---------------------------------------------------------------------------

def _format_article_for_tool(article: dict[str, str]) -> str:
    """Format a single article as a tool result string."""
    article_id = article["article_id"]
    url = f"{KB_BASE_URL}/{article_id}" if article_id else ""
    label = f"Article {article_id} ({url})" if article_id else article["filename"]
    lines = [f"[ARTICLE: {label}]"]
    if article.get("updated"):
        stale_flag = " [POTENTIALLY STALE]" if _is_stale(article["updated"]) else ""
        lines.append(f"Last updated: {article['updated']}{stale_flag}")
    if article.get("owner"):
        lines.append(f"Owner: {article['owner']}")
    lines.append("---")
    tool_content = article["content"]
    if len(tool_content) > MAX_ARTICLE_CHARS_IN_PROMPT:
        tool_content = f"{tool_content[:MAX_ARTICLE_CHARS_IN_PROMPT]}\n\n[TRUNCATED]"
    lines.append(tool_content)
    return "\n".join(lines)


def handle_tool_call(
    tool_block,
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
) -> str:
    """Dispatch a single tool call and return the result string."""
    if tool_block.name == "search_articles":
        query = tool_block.input["query"]
        results = select_relevant_articles(query, articles, vectorizer, doc_matrix)
        if not results:
            return "No relevant articles found for that query."
        return "\n\n".join(_format_article_for_tool(a) for a in results)

    if tool_block.name == "get_article":
        article_id = str(tool_block.input["article_id"]).strip()
        article = next((a for a in articles if a["article_id"] == article_id), None)
        if not article:
            return f"Article {article_id} was not found in the local KB."
        return _format_article_for_tool(article)

    return f"Unknown tool: {tool_block.name}"


def run_agent(
    client: anthropic.Anthropic,
    system_prompt: str,
    conversation: list[dict],
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
) -> str | None:
    """Run the agentic loop, handling tool calls mid-response.

    ``conversation`` is the full message history up to and including the latest
    user turn.  Tool call / result turns are appended to a local copy and are
    NOT persisted to the session — only the final text reply is returned.
    """
    messages = list(conversation)

    for _ in range(MAX_AGENT_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
            tools=TOOLS,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            text_blocks = [b for b in response.content if hasattr(b, "text")]
            return text_blocks[0].text if text_blocks else None

        # Append assistant turn (contains tool_use blocks) and process results
        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": handle_tool_call(tb, articles, vectorizer, doc_matrix),
            }
            for tb in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

    return None


# ---------------------------------------------------------------------------
# Web Application
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KB AI Search</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #262626;
      color: #cccccc;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }

    /* ── HEADER ── */
    #page-header {
      background: #262626;
      padding: 8px 12px;
      flex-shrink: 0;
      border-bottom: 2px solid #555;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    #page-header h1 {
      font-size: 15px;
      font-weight: bold;
      font-style: italic;
      color: #7aaee8;
      margin: 0;
      white-space: nowrap;
      flex-shrink: 0;
    }
    #page-header .nav-link {
      color: #7aaee8;
      font-size: 12px;
      text-decoration: none;
      white-space: nowrap;
      flex-shrink: 0;
      padding: 4px 6px;
      border: 1px solid #444;
    }
    #page-header .nav-link:hover { color: #aaccff; border-color: #7aaee8; }
    .build-badge {
      color: #666;
      font-size: 11px;
      white-space: nowrap;
      flex-shrink: 0;
      margin-left: auto;
    }
    #search-form {
      display: flex;
      flex: 1;
      gap: 6px;
      align-items: center;
    }
    #search-input {
      flex: 1;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #1e1e1e;
      color: #cccccc;
      border: 2px inset #555;
      padding: 4px 6px;
      height: 30px;
    }
    #search-input:focus { outline: none; border-color: #7aaee8; }
    .btn-submit {
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #3a3a3a;
      color: #cccccc;
      border: 2px outset #666;
      padding: 3px 14px;
      height: 30px;
      cursor: pointer;
      white-space: nowrap;
    }
    .btn-submit:hover { background: #484848; }
    .btn-submit:active { border-style: inset; }

    /* ── TWO-COLUMN BODY ── */
    #main-body {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── LEFT PANEL ── */
    #left-panel {
      width: 50%;
      border-right: 2px solid #555;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .panel-header {
      background: #3c5070;
      color: #fff;
      font-weight: bold;
      font-size: 12px;
      padding: 3px 8px;
      flex-shrink: 0;
    }
    #results-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .empty-state {
      color: #777;
      font-size: 13px;
      font-style: italic;
      margin: auto;
      text-align: center;
    }

    /* Article result cards */
    .result-card {
      background: #2e2e2e;
      border: 1px solid #444;
      padding: 6px 8px;
    }
    .result-card.not-in-ai {
      border-color: #3a3a3a;
      opacity: 0.75;
    }
    .card-title-row {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 2px;
    }
    .card-title {
      font-weight: bold;
      color: #cccccc;
      text-decoration: none;
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .card-title:hover { color: #aaccff; text-decoration: underline; }
    .card-id {
      color: #777;
      font-size: 11px;
      white-space: nowrap;
    }
    .relevancy-badge {
      font-size: 11px;
      font-weight: bold;
      padding: 1px 5px;
      border-radius: 2px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .badge-high   { background: #1a3a1a; color: #6ec86e; border: 1px solid #3a6a3a; }
    .badge-medium { background: #3a3010; color: #c8b050; border: 1px solid #6a5020; }
    .badge-low    { background: #2e2e2e; color: #888;    border: 1px solid #444; }
    .card-meta {
      color: #888;
      font-size: 11px;
      margin-bottom: 2px;
    }
    .stale-badge {
      background: #3a2a10;
      color: #c89040;
      border: 1px solid #6a4a20;
      font-size: 10px;
      padding: 0 4px;
      border-radius: 2px;
      margin-left: 4px;
    }
    .card-keywords {
      color: #999;
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 2px;
    }
    .card-excerpt {
      color: #aaa;
      font-size: 12px;
      line-height: 1.4;
    }
    .not-in-ai-note {
      color: #666;
      font-size: 10px;
      font-style: italic;
      margin-top: 3px;
    }

    /* ── RIGHT PANEL ── */
    #right-panel {
      width: 50%;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #ai-content {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
    }
    #ai-body {
      flex: 1;
      white-space: pre-wrap;
      word-wrap: break-word;
      line-height: 1.5;
      font-size: 13px;
      color: #cccccc;
    }
    #ai-body a { color: #7aaee8; text-decoration: underline; }
    #ai-body a:hover { color: #aaccff; }
    .ai-loading {
      color: #888;
      font-style: italic;
    }
    .ai-error {
      color: #d08080;
    }
    #ai-footer {
      color: #777;
      font-size: 11px;
      margin-top: 8px;
      padding-top: 6px;
      border-top: 1px solid #444;
    }
    #refine-area {
      flex-shrink: 0;
      padding: 6px 8px;
      border-top: 1px solid #444;
    }
    #refine-form {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    #refine-form label {
      color: #999;
      font-size: 12px;
      white-space: nowrap;
    }
    #refine-input {
      flex: 1;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 12px;
      background: #1e1e1e;
      color: #cccccc;
      border: 1px inset #555;
      padding: 3px 5px;
      height: 26px;
    }
    #refine-input:focus { outline: none; border-color: #7aaee8; }
    .btn-refine {
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #3a3a3a;
      color: #cccccc;
      border: 2px outset #666;
      padding: 1px 8px;
      height: 26px;
      cursor: pointer;
    }
    .btn-refine:hover { background: #484848; }
    .btn-refine:active { border-style: inset; }

    /* ── BRAVE SEARCH SECTION ── */
    #brave-section {
      flex-shrink: 0;
      max-height: 220px;
      overflow-y: auto;
      padding: 6px 8px;
      border-top: 1px dashed #444;
    }
    .brave-header {
      font-size: 11px;
      font-weight: bold;
      color: #888;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .brave-result {
      background: #2a2a2a;
      border: 1px solid #3a3a3a;
      padding: 5px 8px;
      margin-bottom: 6px;
    }
    .brave-result-title {
      font-size: 12px;
      color: #7aaee8;
      text-decoration: none;
      display: block;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .brave-result-title:hover { color: #aaccff; text-decoration: underline; }
    .brave-result-url {
      font-size: 10px;
      color: #666;
      margin-top: 1px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .brave-result-desc {
      font-size: 11px;
      color: #aaa;
      margin-top: 3px;
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <!-- HEADER -->
  <div id="page-header">
    <h1>KB AI Search</h1>
    <a class="nav-link" href="/articles">Browse Articles</a>
    <span class="build-badge">Build {{ build_number }}</span>
    <form id="search-form" method="post" action="/">
      <input type="text" id="search-input" name="query"
             value="{{ query | e }}"
             placeholder="Describe the support issue..."
             autofocus>
      <button type="submit" class="btn-submit">Submit</button>
    </form>
  </div>

  <!-- MAIN BODY -->
  <div id="main-body">

    <!-- LEFT: Search Results -->
    <div id="left-panel">
      <div class="panel-header" id="results-header">SEARCH RESULTS{% if display_articles is not none %} ({{ display_articles | length }} found){% endif %}</div>
      <div id="results-list">
        {% if display_articles is none %}
          <p class="empty-state">Enter a support question above to get started.</p>
        {% elif display_articles | length == 0 %}
          <p class="empty-state">No articles found for this query.</p>
        {% else %}
          {% for article, score in display_articles %}
            {% set badge_label, badge_class = classify_badge(score, loop.index0, display_scores, query, article) %}
            {% set in_ai = loop.index0 < top_k %}
            <div class="result-card{% if not in_ai %} not-in-ai{% endif %}">
              <div class="card-title-row">
                {% if article.article_id %}
                  <a class="card-title" href="{{ kb_base_url }}/{{ article.article_id }}"
                     title="{{ article.title }}">{{ article.title }}</a>
                {% else %}
                  <span class="card-title" title="{{ article.title }}">{{ article.title }}</span>
                {% endif %}
                <span class="card-id">#{{ article.article_id }}</span>
                <span class="relevancy-badge {{ badge_class }}">{{ badge_label }}</span>
              </div>
              <div class="card-meta">
                {% if article.owner %}{{ article.owner }}{% endif %}
                {% if article.updated %} · Updated: {{ article.updated }}
                  {% if article.updated | is_stale %}<span class="stale-badge">⚠ Last updated {{ article.updated[:4] }}</span>{% endif %}
                {% endif %}
              </div>
              {% if article.keywords %}
                <div class="card-keywords">Keywords: {{ article.keywords }}</div>
              {% endif %}
              {% if article.content %}
                <div class="card-excerpt">{{ article.content[:200] }}{% if article.content | length > 200 %}…{% endif %}</div>
              {% endif %}
              {% if not in_ai %}
                <div class="not-in-ai-note">Not included in AI response</div>
              {% endif %}
            </div>
          {% endfor %}
        {% endif %}
      </div>
    </div>

    <!-- RIGHT: AI Troubleshooting -->
    <div id="right-panel">
      <div class="panel-header">AI TROUBLESHOOTING</div>
      <div id="ai-content">
        {% if ai_response is none and display_articles is none %}
          <p class="empty-state">AI troubleshooting steps will appear here after a search.</p>
        {% elif ai_error %}
          <div id="ai-body" class="ai-error">{{ ai_error }}</div>
        {% elif ai_response %}
          <div id="ai-body">{{ ai_response | e }}</div>
          {% if ai_footer %}
            <div id="ai-footer">{{ ai_footer }}</div>
          {% endif %}
        {% else %}
          <p class="empty-state">No response received.</p>
        {% endif %}
      </div>
      {% if brave_available %}
        <!-- Brave Search: populated by JS when KB confidence is low; lives outside
             #ai-content so AI response updates never wipe it. -->
        <div id="brave-section" style="display:none">
          <div class="brave-header">🔍 Users with Similar Issues / Potential Solutions</div>
          <div id="brave-list"></div>
        </div>
      {% endif %}
      {% if ai_response and not ai_error %}
        <div id="refine-area">
      {% else %}
        <div id="refine-area" style="display:none">
      {% endif %}
          <form id="refine-form" method="post" action="/refine">
            <input type="hidden" name="original_query" value="{{ query | e }}">
            <label for="refine-input">Refine:</label>
            <input type="text" id="refine-input" name="refinement"
                   placeholder="Add context or clarify...">
            <button type="submit" class="btn-refine">→</button>
          </form>
        </div>

    </div>

  </div>

  <script>
    // ── Utilities ────────────────────────────────────────────────────────────
    function esc(s) {
      return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function buildCard(a) {
      const notInAiClass = a.in_ai ? '' : ' not-in-ai';
      const titleHtml = a.url
        ? `<a class="card-title" href="${esc(a.url)}" title="${esc(a.title)}">${esc(a.title)}</a>`
        : `<span class="card-title" title="${esc(a.title)}">${esc(a.title)}</span>`;
      const staleHtml = a.is_stale
        ? ` <span class="stale-badge">⚠ Last updated ${esc(String(a.updated).slice(0, 4))}</span>`
        : '';
      const metaHtml = [
        a.owner ? esc(a.owner) : '',
        a.updated ? ` · Updated: ${esc(a.updated)}${staleHtml}` : '',
      ].join('');
      const kwHtml = a.keywords
        ? `<div class="card-keywords">Keywords: ${esc(a.keywords)}</div>` : '';
      const exHtml = a.excerpt
        ? `<div class="card-excerpt">${esc(a.excerpt)}${a.excerpt_truncated ? '…' : ''}</div>` : '';
      const noteHtml = a.in_ai ? '' : '<div class="not-in-ai-note">Not included in AI response</div>';
      return `<div class="result-card${notInAiClass}">
  <div class="card-title-row">${titleHtml}<span class="card-id">#${esc(a.article_id)}</span><span class="relevancy-badge ${esc(a.badge_class)}">${esc(a.badge_label)}</span></div>
  <div class="card-meta">${metaHtml}</div>${kwHtml}${exHtml}${noteHtml}
</div>`;
    }

    // ── DOM refs ─────────────────────────────────────────────────────────────
    const searchInput   = document.getElementById('search-input');
    const searchForm    = document.getElementById('search-form');
    const resultsHeader = document.getElementById('results-header');
    const resultsList   = document.getElementById('results-list');
    const aiContent     = document.getElementById('ai-content');
    const refineArea    = document.getElementById('refine-area');
    const refineForm    = document.getElementById('refine-form');
    const refineInput   = document.getElementById('refine-input');
    const refineQueryIn = refineForm ? refineForm.querySelector('[name="original_query"]') : null;
    const submitBtn     = searchForm ? searchForm.querySelector('.btn-submit') : null;
    const refineBtn     = refineForm ? refineForm.querySelector('.btn-refine') : null;
    const braveSection  = document.getElementById('brave-section');
    const braveList     = document.getElementById('brave-list');

    // ── State ─────────────────────────────────────────────────────────────────
    let searchTimer = null;
    let lastSearchQuery = '';
    let aiInFlight = false;
    let activeAiController = null;

    function setAiBusy(isBusy) {
      if (submitBtn) submitBtn.disabled = isBusy;
      if (refineBtn) refineBtn.disabled = isBusy;
    }

    // ── Brave Search ──────────────────────────────────────────────────────────
    const BRAVE_MIN_HIGH_CONF = {{ brave_min_high_conf }};

    function renderBraveResults(results) {
      if (!braveSection || !braveList) return;
      if (!results || results.length === 0) {
        braveSection.style.display = 'none';
        return;
      }
      braveList.innerHTML = results.map(r => {
        const titleHtml = r.url
          ? `<a class="brave-result-title" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>`
          : `<span class="brave-result-title">${esc(r.title)}</span>`;
        const urlHtml  = r.url  ? `<div class="brave-result-url">${esc(r.url)}</div>`         : '';
        const descHtml = r.description ? `<div class="brave-result-desc">${esc(r.description)}</div>` : '';
        return `<div class="brave-result">${titleHtml}${urlHtml}${descHtml}</div>`;
      }).join('');
      braveSection.style.display = '';
    }

    async function doBraveSearch(q) {
      if (!braveSection) return;
      try {
        const resp = await fetch(`/brave?q=${encodeURIComponent(q)}`);
        const data = await resp.json();
        if (data.available) {
          renderBraveResults(data.results);
        } else {
          braveSection.style.display = 'none';
        }
      } catch (_) {
        if (braveSection) braveSection.style.display = 'none';
      }
    }

    function checkBraveSearch(articles, q) {
      if (!braveSection) return;
      const highCount = articles.filter(a => a.badge_label === 'High').length;
      if (highCount < BRAVE_MIN_HIGH_CONF) {
        doBraveSearch(q);
      } else {
        braveSection.style.display = 'none';
      }
    }

    // ── Live search (debounced 300 ms) ────────────────────────────────────────
    async function doSearch(q) {
      if (!q) {
        resultsHeader.textContent = 'SEARCH RESULTS';
        resultsList.innerHTML = '<p class="empty-state">Enter a support question above to get started.</p>';
        lastSearchQuery = '';
        if (braveSection) braveSection.style.display = 'none';
        return;
      }
      try {
        const resp = await fetch(`/search?q=${encodeURIComponent(q)}`);
        const data = await resp.json();
        resultsHeader.textContent = `SEARCH RESULTS (${data.count} found)`;
        resultsList.innerHTML = data.articles.length === 0
          ? '<p class="empty-state">No articles found for this query.</p>'
          : data.articles.map(buildCard).join('');
        lastSearchQuery = q;
        // Nudge AI panel if it still shows the initial empty state
        const aiEmpty = aiContent.querySelector('.empty-state');
        if (aiEmpty && data.count > 0) {
          aiEmpty.textContent = 'Press Submit for AI troubleshooting steps.';
        }
        // Trigger Brave search when KB confidence is low
        checkBraveSearch(data.articles, q);
      } catch (err) {
        // Show a visible error in the left panel rather than failing silently
        resultsHeader.textContent = 'SEARCH RESULTS';
        resultsList.innerHTML = '<p class="empty-state">Could not load results. Check your connection and try again.</p>';
      }
    }

    searchInput.addEventListener('input', function () {
      clearTimeout(searchTimer);
      const q = this.value.trim();
      searchTimer = setTimeout(() => doSearch(q), 300);
    });

    // ── Async AI response ─────────────────────────────────────────────────────
    function showAiLoading() {
      aiContent.innerHTML = '<p class="empty-state" style="color:#888;font-style:italic">Generating troubleshooting steps\u2026</p>';
      if (refineArea) refineArea.style.display = 'none';
    }

    function showAiResult(data, query) {
      if (data.error) {
        aiContent.innerHTML = `<div id="ai-body" class="ai-error">${esc(data.error)}</div>`;
      } else if (data.response) {
        let html = `<div id="ai-body">${esc(data.response)}</div>`;
        if (data.footer) html += `<div id="ai-footer">${esc(data.footer)}</div>`;
        aiContent.innerHTML = html;
        if (refineQueryIn) refineQueryIn.value = query;
        if (refineInput)   refineInput.value   = '';
        if (refineArea)    refineArea.style.display = '';
      } else {
        aiContent.innerHTML = '<p class="empty-state">No response received.</p>';
      }
      // Sync brave-section visibility using server-computed article confidence.
      // This ensures brave shows/hides correctly even when doSearch was skipped.
      if (braveSection && typeof data.high_conf_count !== 'undefined') {
        if (data.high_conf_count < BRAVE_MIN_HIGH_CONF) {
          doBraveSearch(query);
        } else {
          braveSection.style.display = 'none';
        }
      }
    }

    async function doAi(query, refinement) {
      if (activeAiController) {
        activeAiController.abort();
      }
      const controller = new AbortController();
      activeAiController = controller;
      aiInFlight = true;
      setAiBusy(true);
      showAiLoading();
      const body = new FormData();
      body.append('query', query);
      if (refinement) body.append('refinement', refinement);
      try {
        const resp = await fetch('/ai', { method: 'POST', body, signal: controller.signal });
        const data = await resp.json();
        showAiResult(data, query);
      } catch (err) {
        if (err && err.name === 'AbortError') return;
        aiContent.innerHTML = `<div id="ai-body" class="ai-error">Request failed: ${esc(String(err))}</div>`;
      } finally {
        if (activeAiController === controller) {
          activeAiController = null;
        }
        aiInFlight = false;
        setAiBusy(false);
      }
    }

    // ── Form submit → fire search then AI concurrently ────────────────────────
    searchForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      const q = searchInput.value.trim();
      if (!q) return;
      clearTimeout(searchTimer);
      // Fetch results synchronously first (so user sees them before AI loads)
      if (lastSearchQuery !== q) {
        await doSearch(q);
      }
      // AI call fires without blocking — user can already read articles
      if (aiInFlight) return;
      doAi(q);
    });

    // ── Refine submit → async AI only (left panel unchanged) ─────────────────
    if (refineForm) {
      refineForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const origQuery  = refineQueryIn ? refineQueryIn.value.trim() : '';
        const refinement = refineInput   ? refineInput.value.trim()   : '';
        if (!origQuery) return;
        if (aiInFlight) return;
        doAi(origQuery, refinement);
      });
    }

    // ── On page load: check server-rendered badges (e.g. after /refine) ───────
    window.addEventListener('DOMContentLoaded', function () {
      const q = searchInput ? searchInput.value.trim() : '';
      if (q && braveSection) {
        const highBadges = document.querySelectorAll('.badge-high').length;
        if (highBadges < BRAVE_MIN_HIGH_CONF) {
          doBraveSearch(q);
        }
      }
    });
  </script>
</body>
</html>
"""


ARTICLES_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KB Articles – KB AI Search</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #262626;
      color: #cccccc;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }

    /* ── HEADER ── */
    #page-header {
      background: #262626;
      padding: 8px 12px;
      flex-shrink: 0;
      border-bottom: 2px solid #555;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    #page-header h1 {
      font-size: 15px;
      font-weight: bold;
      font-style: italic;
      color: #7aaee8;
      margin: 0;
      white-space: nowrap;
      flex-shrink: 0;
    }
    #page-header .nav-link {
      color: #7aaee8;
      font-size: 12px;
      text-decoration: none;
      white-space: nowrap;
      flex-shrink: 0;
      padding: 4px 6px;
      border: 1px solid #444;
    }
    #page-header .nav-link:hover { color: #aaccff; border-color: #7aaee8; }
    .header-count {
      color: #888;
      font-size: 12px;
      margin-left: auto;
    }
    .build-badge {
      color: #666;
      font-size: 11px;
      white-space: nowrap;
      flex-shrink: 0;
    }

    /* ── CONTENT ── */
    #content {
      padding: 12px 16px;
      flex: 1;
    }
    .page-title {
      font-size: 14px;
      font-weight: bold;
      color: #7aaee8;
      margin: 0 0 12px 0;
    }

    /* ── TABLE ── */
    table {
      width: 100%;
      border-collapse: collapse;
    }
    thead th {
      background: #3c5070;
      color: #fff;
      font-size: 12px;
      font-weight: bold;
      padding: 5px 8px;
      text-align: left;
      white-space: nowrap;
    }
    tbody tr:nth-child(even) { background: #2a2a2a; }
    tbody tr:hover { background: #303040; }
    td {
      padding: 5px 8px;
      border-bottom: 1px solid #3a3a3a;
      vertical-align: top;
    }
    .col-title { width: 30%; }
    .col-id    { width: 8%; white-space: nowrap; }
    .col-owner { width: 20%; }
    .col-updated { width: 12%; white-space: nowrap; }
    .col-keywords { width: 30%; }
    .article-link {
      color: #7aaee8;
      text-decoration: none;
    }
    .article-link:hover { color: #aaccff; text-decoration: underline; }
    .stale-badge {
      background: #3a2a10;
      color: #c89040;
      border: 1px solid #6a4a20;
      font-size: 10px;
      padding: 0 4px;
      border-radius: 2px;
      margin-left: 4px;
      white-space: nowrap;
    }
    .id-cell {
      color: #777;
      font-size: 11px;
    }
    .keywords-cell {
      color: #999;
      font-size: 11px;
    }
    .empty-state {
      color: #777;
      font-style: italic;
    }
  </style>
</head>
<body>
  <!-- HEADER -->
  <div id="page-header">
    <h1>KB AI Search</h1>
    <a class="nav-link" href="/">← Back to Search</a>
    <span class="header-count">{{ article_count }} article{{ 's' if article_count != 1 else '' }} indexed</span>
    <span class="build-badge">Build {{ build_number }}</span>
  </div>

  <!-- CONTENT -->
  <div id="content">
    <p class="page-title">All Indexed Articles</p>
    {% if articles %}
    <table>
      <thead>
        <tr>
          <th class="col-title">Title</th>
          <th class="col-id">Article ID</th>
          <th class="col-owner">Owner</th>
          <th class="col-updated">Updated</th>
          <th class="col-keywords">Keywords</th>
        </tr>
      </thead>
      <tbody>
        {% for article in articles %}
        <tr>
          <td class="col-title">
            {% if article.article_id %}
              <a class="article-link" href="{{ kb_base_url }}/{{ article.article_id }}" target="_blank">{{ article.title }}</a>
            {% else %}
              {{ article.title }}
            {% endif %}
          </td>
          <td class="col-id id-cell">{{ article.article_id or '—' }}</td>
          <td class="col-owner">{{ article.owner or '' }}</td>
          <td class="col-updated">
            {{ article.updated or '' }}
            {% if article.updated and article.updated | is_stale %}
              <span class="stale-badge">⚠ {{ article.updated[:4] }}</span>
            {% endif %}
          </td>
          <td class="col-keywords keywords-cell">{{ article.keywords or '' }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
      <p class="empty-state">No articles are currently indexed.</p>
    {% endif %}
  </div>
</body>
</html>
"""


def create_app(
    client: anthropic.Anthropic,
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
    contacts_text: str = "",
    retriever: Optional["HybridRetriever"] = None,  # type: ignore[name-defined]
    brave_api_key: str = "",
) -> Flask:
    """Create and configure the Flask application.

    For each incoming query the TF-IDF index is used to select:
    - DISPLAY_K_ARTICLES articles for the left panel (with scores for relevancy labels)
    - TOP_K_ARTICLES articles to pass to Claude for the right panel

    When *retriever* is provided (and its feature flags are enabled) the
    hybrid retrieval path is used instead of direct TF-IDF cosine similarity.

    Claude may fetch additional articles mid-response via tool calls handled
    by the agentic loop.
    """
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)

    # Resolve the build number once at startup so every request uses the same value.
    _build_number = get_build_number()

    # Build article lookup map once for O(1) access in route helpers
    _id_to_article: dict[str, dict] = {a["article_id"]: a for a in articles}

    def _is_stale_filter(updated: str) -> bool:
        return _is_stale(updated)

    app.jinja_env.filters["is_stale"] = _is_stale_filter

    def _token_root(token: str) -> str:
      for suffix in ("ments", "ment", "ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and (len(token) - len(suffix)) >= 4:
          return token[: -len(suffix)]
      return token

    def _keyword_roots(text: str) -> set[str]:
      stop = {
        "how", "do", "i", "in", "to", "the", "a", "an", "for", "and",
        "or", "of", "on", "with", "my", "is", "are", "can", "you", "me",
        "help", "please", "issue", "problem", "not", "from", "at", "it",
      }
      tokens = re.findall(r"[a-z0-9]+", text.lower())
      roots: set[str] = set()
      for token in tokens:
        if len(token) < 3 or token in stop:
          continue
        roots.add(_token_root(token))
      return roots

    def _has_keyword_overlap(query: str, article: dict[str, str]) -> bool:
      query_roots = _keyword_roots(query)
      if not query_roots:
        return False

      article_text = (
        f"{article.get('title', '')} "
        f"{article.get('keywords', '')} "
        f"{article.get('content', '')[:1200]}"
      )
      article_roots = _keyword_roots(article_text)
      overlap = query_roots & article_roots

      if len(overlap) >= 2:
        return True
      if len(overlap) == 1 and len(query_roots) <= 2:
        return True
      return False

    def _classify_relevance_badge(
      score: float,
      rank: int,
      all_scores: list[float],
      query_text: str,
      article: dict[str, str],
    ) -> tuple[str, str]:
      """Classify relevance badges with overlap-aware near-exact promotion."""
      if score >= RELEVANCE_HIGH:
        return "High", "badge-high"

      if rank == 0 and score >= 0.16:
        second = all_scores[1] if len(all_scores) > 1 else 0.0
        if (score - second) >= 0.03 and _has_keyword_overlap(query_text, article):
          return "High", "badge-high"

      if score >= RELEVANCE_MEDIUM:
        return "Medium", "badge-medium"

      return "Low", "badge-low"

    def _render(
        query: str = "",
        display_articles=None,
        ai_response: str | None = None,
        ai_footer: str = "",
        ai_error: str = "",
    ):
        display_scores = [score for _, score in display_articles] if display_articles else []
        return render_template_string(
            HTML_TEMPLATE,
            query=query,
            display_articles=display_articles,
            display_scores=display_scores,
            ai_response=ai_response,
            ai_footer=ai_footer,
            ai_error=ai_error,
            top_k=TOP_K_ARTICLES,
            relevance_high=RELEVANCE_HIGH,
            relevance_medium=RELEVANCE_MEDIUM,
            classify_badge=_classify_relevance_badge,
            kb_base_url=KB_BASE_URL,
            brave_min_high_conf=BRAVE_MIN_HIGH_CONF,
            brave_available=bool(brave_api_key),
            build_number=_build_number,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers (enhanced path when retriever is provided)
    # ------------------------------------------------------------------

    def _get_display_articles(
        query: str,
    ) -> list[tuple[dict[str, str], float]]:
        """Return display articles as (article, score) pairs.

        Uses the hybrid retriever when available; otherwise falls back to the
        original TF-IDF cosine similarity implementation.
        """
        if retriever is not None:
            id_score_pairs, _ = retriever.retrieve(
                query, articles, vectorizer, doc_matrix, top_k=DISPLAY_K_ARTICLES
            )
            return [
                (_id_to_article[aid], score)
                for aid, score in id_score_pairs
                if aid in _id_to_article
            ]
        return select_display_articles(query, articles, vectorizer, doc_matrix)

    def _get_top_articles_for_claude(
        display: list[tuple[dict[str, str], float]],
    ) -> list[dict[str, str]]:
        """Select the articles to pass to Claude.

        With FEATURE_ADAPTIVE_TOPK enabled the selection boundary is determined
        by score gaps rather than the fixed TOP_K_ARTICLES constant.
        """
        if (
            _SEARCH_ENHANCEMENT_AVAILABLE
            and retriever is not None
            and retriever.flags.adaptive_topk
        ):
            scores = [s for _, s in display]
            k = adaptive_top_k(
                scores,
                min_k=TOP_K_ARTICLES,
                max_k=min(DISPLAY_K_ARTICLES, len(display)),
            )
            return [a for a, _ in display[:k]]
        return [a for a, _ in display[:TOP_K_ARTICLES]]

    # ------------------------------------------------------------------

    def _run_claude(query: str, top_articles: list[dict[str, str]]) -> tuple[str | None, str, str]:
        """Run Claude for the given query and top articles.

        Returns ``(ai_response, ai_footer, ai_error)``.
        """
        system_prompt = build_system_prompt(top_articles, contacts_text)
        messages = [{"role": "user", "content": query}]
        ai_response = None
        ai_footer = ""
        ai_error = ""
        try:
            ai_response = run_agent(client, system_prompt, messages, articles, vectorizer, doc_matrix)
            if ai_response is None:
                ai_error = "No text response received from the assistant."
            else:
                # Build footer: count articles and find most recent date
                n = len(top_articles)
                dates = [a["updated"] for a in top_articles if a.get("updated")]
                most_recent = max(dates) if dates else None
                ai_footer = f"Generated from {n} article{'s' if n != 1 else ''}"
                if most_recent:
                    ai_footer += f" · Most recent: {most_recent}"
        except anthropic.RateLimitError:
            ai_error = (
                "Rate limit reached. The request was too large or too many requests were "
                "made in a short period. Please wait a moment and try again."
            )
        except anthropic.APIError as exc:
            ai_error = f"API error: {exc}"
        return ai_response, ai_footer, ai_error

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return _render()

        query = request.form.get("query", "").strip()
        if not query:
            return _render()

        # Select articles for both panels in one pass
        display = _get_display_articles(query)
        top_articles = _get_top_articles_for_claude(display)

        ai_response, ai_footer, ai_error = _run_claude(query, top_articles)
        return _render(
            query=query,
            display_articles=display,
            ai_response=ai_response,
            ai_footer=ai_footer,
            ai_error=ai_error,
        )

    @app.route("/refine", methods=["POST"])
    def refine():
        original_query = request.form.get("original_query", "").strip()
        refinement = request.form.get("refinement", "").strip()

        if not original_query:
            return _render()

        # Re-compute display articles from the original query (left panel unchanged)
        display = _get_display_articles(original_query)
        top_articles = _get_top_articles_for_claude(display)

        # Build combined prompt for Claude
        if refinement:
            combined_query = f"[Original query]: {original_query} / [Refinement]: {refinement}"
        else:
            combined_query = original_query

        ai_response, ai_footer, ai_error = _run_claude(combined_query, top_articles)
        return _render(
            query=original_query,
            display_articles=display,
            ai_response=ai_response,
            ai_footer=ai_footer,
            ai_error=ai_error,
        )

    @app.route("/search")
    def search():
        """Return JSON article data for the left panel (live / debounced search)."""
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"articles": [], "count": 0})

        display = _get_display_articles(q)
        display_scores = [score for _, score in display]
        result = []
        for i, (article, score) in enumerate(display):
            badge_label, badge_class = _classify_relevance_badge(score, i, display_scores, q, article)
            result.append({
                "article_id": article["article_id"],
                "title": article["title"],
                "owner": article.get("owner", ""),
                "updated": article.get("updated", ""),
                "is_stale": _is_stale(article.get("updated", "")),
                "keywords": article.get("keywords", ""),
                "excerpt": article.get("content", "")[:200],
                "excerpt_truncated": len(article.get("content", "")) > 200,
                "badge_label": badge_label,
                "badge_class": badge_class,
                "in_ai": i < TOP_K_ARTICLES,
                "url": f"{KB_BASE_URL}/{article['article_id']}" if article["article_id"] else "",
            })

        resp = jsonify({"articles": result, "count": len(result)})
        return resp

    @app.route("/brave")
    def brave():
        """Return Brave Search web results for low-confidence queries.

        Called by the frontend when fewer than BRAVE_MIN_HIGH_CONF articles
        have a "High" relevancy badge.  Returns JSON::

            {
              "results":   [{"title": ..., "url": ..., "description": ...}, ...],
              "available": true | false   # false when no API key is configured
            }
        """
        q = request.args.get("q", "").strip()
        if not q or not brave_api_key:
            return jsonify({"results": [], "available": bool(brave_api_key)})
        results = fetch_brave_results(q, brave_api_key)
        return jsonify({"results": results, "available": True})

    @app.route("/ai", methods=["POST"])
    def ai():
        """Run the Claude agent and return JSON {response, footer, error, high_conf_count}."""
        query = request.form.get("query", "").strip()
        refinement = request.form.get("refinement", "").strip()
        if not query:
            return jsonify({"response": None, "footer": "", "error": "No query provided.", "high_conf_count": 0})

        combined = (
            f"[Original query]: {query} / [Refinement]: {refinement}"
            if refinement
            else query
        )
        # Derive top articles from the original (un-refined) query
        display = _get_display_articles(query)
        display_scores = [score for _, score in display]
        high_conf_count = sum(
            1 for i, (article, score) in enumerate(display)
            if _classify_relevance_badge(score, i, display_scores, query, article)[0] == "High"
        )
        top_articles = _get_top_articles_for_claude(display)
        ai_response, ai_footer, ai_error = _run_claude(combined, top_articles)
        return jsonify({"response": ai_response, "footer": ai_footer, "error": ai_error, "high_conf_count": high_conf_count})

    @app.route("/articles")
    def articles_index():
        """Render a page listing all indexed KB articles."""
        sorted_articles = sorted(articles, key=lambda a: a.get("title", "").lower())
        return render_template_string(
            ARTICLES_TEMPLATE,
            articles=sorted_articles,
            article_count=len(sorted_articles),
            kb_base_url=KB_BASE_URL,
            build_number=_build_number,
        )

    return app


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IntelligentKB web server.")
    parser.add_argument(
        "--build-number",
        action="store_true",
        help="Print the most recent build number and exit.",
    )
    return parser.parse_args()


def get_build_number() -> str:
    """Return a build identifier from CI env vars or the local git commit."""
    env_build = os.environ.get("BUILD_NUMBER", "").strip()
    if env_build:
        return env_build

    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha[:7]

    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if commit_sha:
            return commit_sha
    except (subprocess.SubprocessError, OSError):
        pass

    return "unknown"


def main() -> None:
    args = _parse_args()
    if args.build_number:
        print(get_build_number())
        return

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Create a .env file with ANTHROPIC_API_KEY=your_key_here, "
            "or set the environment variable directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not ARTICLES_DIR.is_dir():
        print(
            f"Error: Articles directory not found at '{ARTICLES_DIR}'.\n"
            "Create an 'articles/' directory containing .htm or .html KB article files.",
            file=sys.stderr,
        )
        sys.exit(1)

    articles, contacts_text = load_articles(ARTICLES_DIR)
    if not articles:
        print(
            f"Warning: No .htm or .html files found in '{ARTICLES_DIR}'.\n"
            "Add KB articles to the articles/ directory and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(articles)} KB article(s):")
    for a in articles:
        print(f"    {a['filename']}")

    vectorizer, doc_matrix = build_article_index(articles)

    # Optionally initialise the hybrid retriever (no-op if all flags are False)
    retriever = None
    if _SEARCH_ENHANCEMENT_AVAILABLE:
        from search_enhancement import FeatureFlags, HybridRetriever
        flags = FeatureFlags()
        if flags.any_enabled():
            retriever = HybridRetriever(flags)
            retriever.build(articles, vectorizer, doc_matrix)
            print(f"Search enhancements active: {flags}")

    client = anthropic.Anthropic(
        api_key=api_key,
        max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "1")),
    )
    brave_api_key = os.environ.get("BRAVE_API_KEY", "")
    if brave_api_key:
        print("Brave Search integration enabled.")
    app = create_app(client, articles, vectorizer, doc_matrix, contacts_text, retriever, brave_api_key)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nStarting web server at http://{host}:{port}/")
    print("Press Ctrl+C to stop.\n")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
