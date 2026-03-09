"""
KB AI Search Agent
==================
A locally-run web app where a help desk consultant describes a support problem
in plain English and receives relevant troubleshooting steps drawn from local KB
articles. TF-IDF pre-filtering is used to select only the most relevant articles
for each query before sending them to Claude, reducing context size and cost.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
from datetime import date
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
from flask import Flask, redirect, render_template_string, request, session, url_for

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTICLES_DIR = Path(__file__).parent / "articles"
CONTACTS_FILENAME = "contacts.html"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048

# Maximum number of articles sent to Claude per query.
# Raising this improves recall at the cost of larger context windows.
TOP_K_ARTICLES = 3

KB_BASE_URL = "https://answers.uillinois.edu/illinois/internal"

# Articles last updated more than this many years ago are flagged as potentially stale.
STALE_ARTICLE_YEARS = 2

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
    "- Reference prior messages in the conversation to avoid repeating information already covered.\n"
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
    articles = []
    contacts_text = ""
    for ext in ("*.htm", "*.html"):
        for filepath in sorted(articles_dir.glob(ext)):
            if filepath.name.lower() == CONTACTS_FILENAME:
                contacts_text = parse_article(filepath)["content"]
            else:
                articles.append(parse_article(filepath))
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
        lines.append(article["content"])
        lines.append("")  # blank separator between articles
    if contacts_text:
        lines += ["", "--- [ESCALATION CONTACTS] ---", "", contacts_text, ""]
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
    lines.append(article["content"])
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

    while True:
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


# ---------------------------------------------------------------------------
# Web Application
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KB AI Search Agent</title>
  <style>
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
      padding: 8px 12px 4px 12px;
      text-align: center;
      flex-shrink: 0;
      border-bottom: 2px solid #555;
    }
    #page-header h1 {
      font-size: 16px;
      font-weight: bold;
      font-style: italic;
      color: #7aaee8;
      margin: 0 0 4px 0;
    }
    #page-header .header-link {
      float: right;
      margin-top: -22px;
    }
    #page-header .header-link a {
      color: #7aaee8;
      font-style: italic;
      font-size: 12px;
    }
    #page-header .header-link a:hover { text-decoration: underline; }
    hr.rule {
      border: none;
      border-top: 1px solid #555;
      margin: 0;
    }

    /* ── CHAT TRANSCRIPT ── */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .entry { width: 100%; }

    /* Section bar (like the gray record-details bars in the reference) */
    .entry-label {
      background: #3c5070;
      color: #ffffff;
      font-weight: bold;
      font-size: 12px;
      padding: 2px 6px;
      border: 1px solid #506080;
    }

    .entry-body {
      white-space: pre-wrap;
      word-wrap: break-word;
      line-height: 1.5;
      font-size: 13px;
      padding: 5px 8px;
      background: #2e2e2e;
      border: 1px solid #444;
      border-top: none;
      color: #cccccc;
    }
    .entry.user .entry-label { background: #3a4a3a; border-color: #4a6050; }
    .entry.user .entry-body  { background: #2a2e2a; border-color: #404840; color: #b8ccb8; }
    .entry.error .entry-label { background: #4a2a2a; border-color: #6a3030; }
    .entry.error .entry-body  { background: #2e2020; border-color: #503030; color: #d08080; }

    .entry.assistant .entry-body a { color: #7aaee8; text-decoration: underline; }
    .entry.assistant .entry-body a:hover { color: #aaccff; }

    .empty-state {
      color: #777;
      font-size: 13px;
      font-style: italic;
      margin: auto;
      text-align: center;
    }

    /* ── FOOTER / INPUT ── */
    #page-footer {
      background: #262626;
      border-top: 2px solid #555;
      padding: 6px 12px;
      flex-shrink: 0;
    }
    #page-footer .section-bar {
      background: #3c5070;
      color: #fff;
      font-weight: bold;
      font-size: 12px;
      padding: 2px 6px;
      border: 1px solid #506080;
      margin-bottom: 0;
    }
    #page-footer form {
      display: flex;
      gap: 6px;
      align-items: flex-end;
      background: #2e2e2e;
      border: 1px solid #444;
      border-top: none;
      padding: 6px;
    }
    textarea {
      flex: 1;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #1e1e1e;
      color: #cccccc;
      border: 2px inset #555;
      padding: 3px 5px;
      resize: none;
      height: 40px;
      line-height: 1.4;
    }
    textarea:focus { outline: none; border-color: #7aaee8; }
    button[type=submit] {
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      background: #3a3a3a;
      color: #cccccc;
      border: 2px outset #666;
      padding: 3px 14px;
      height: 40px;
      cursor: pointer;
      white-space: nowrap;
    }
    button[type=submit]:hover { background: #484848; }
    button[type=submit]:active { border-style: inset; }
    button[type=submit]:disabled { color: #666; cursor: default; }
  </style>
</head>
<body>
  <div id="page-header">
    <div class="header-link"><a href="/clear">[ New Session ]</a></div>
    <h1>KB AI Search Agent</h1>
  </div>
  <hr class="rule">

  <div id="chat">
    {% if not conversation %}
      <p class="empty-state">Enter a support question below to get started.</p>
    {% endif %}
    {% for msg in conversation %}
      <div class="entry {{ msg.role }}">
        <div class="entry-label">
          {% if msg.role == 'user' %}You:{% else %}KB Agent:{% endif %}
        </div>
        <div class="entry-body">{{ msg.content }}</div>
      </div>
    {% endfor %}
    {% if error %}
      <div class="entry error">
        <div class="entry-label">Error:</div>
        <div class="entry-body">{{ error }}</div>
      </div>
    {% endif %}
  </div>

  <div id="page-footer">
    <div class="section-bar">Enter Query:</div>
    <form method="post" action="/">
      <textarea name="issue" placeholder="Describe the support issue..."
                autofocus>{{ prefill }}</textarea>
      <button type="submit">Send</button>
    </form>
  </div>

  <script>
    // Auto-scroll to bottom on load
    const chat = document.getElementById('chat');
    chat.scrollTop = chat.scrollHeight;

    const ta = document.querySelector('textarea');
    const form = ta.closest('form');

    function escapeHtml(str) {
      return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function submitQuery() {
      const val = ta.value.trim();
      if (!val) return;

      // Remove empty state placeholder if present
      const emptyState = chat.querySelector('.empty-state');
      if (emptyState) emptyState.remove();

      // Optimistically append the user message
      const userEntry = document.createElement('div');
      userEntry.className = 'entry user';
      userEntry.innerHTML =
        '<div class="entry-label">You:</div>' +
        '<div class="entry-body">' + escapeHtml(val) + '</div>';
      chat.appendChild(userEntry);

      // Append a "processing" indicator
      const thinkingEntry = document.createElement('div');
      thinkingEntry.className = 'entry assistant';
      thinkingEntry.innerHTML =
        '<div class="entry-label">KB Agent:</div>' +
        '<div class="entry-body" style="color:#888">Processing...</div>';
      chat.appendChild(thinkingEntry);

      chat.scrollTop = chat.scrollHeight;

      // Submit the form before disabling the textarea so the value is
      // included in the POST body (disabled fields are excluded by browsers).
      form.submit();

      // Disable input while waiting for the server response.
      ta.disabled = true;
      document.querySelector('button[type=submit]').disabled = true;
    }

    // Enter submits; Shift+Enter inserts newline
    ta.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
      }
    });

    form.addEventListener('submit', function(e) {
      e.preventDefault();
      submitQuery();
    });
  </script>
</body>
</html>
"""


def create_app(
    client: anthropic.Anthropic,
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
    contacts_text: str = "",
) -> Flask:
    """Create and configure the Flask application.

    For each incoming query the TF-IDF index is used to select only the most
    relevant articles before building the system prompt, keeping context size
    small and reducing API cost.  Claude may then fetch additional articles
    mid-response via tool calls handled by the agentic loop.
    """
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if "conversation" not in session:
            session["conversation"] = []

        error = ""
        prefill = ""

        if request.method == "POST":
            user_input = request.form.get("issue", "").strip()
            if user_input:
                conversation = session["conversation"]
                conversation.append({"role": "user", "content": user_input})

                # Select only the most relevant articles for this query.
                relevant = select_relevant_articles(user_input, articles, vectorizer, doc_matrix)
                system_prompt = build_system_prompt(relevant, contacts_text)

                try:
                    assistant_message = run_agent(
                        client, system_prompt, conversation, articles, vectorizer, doc_matrix
                    )
                except anthropic.RateLimitError:
                    conversation.pop()
                    error = "Rate limit reached. The request was too large or too many requests were made in a short period. Please wait a moment and try again."
                    prefill = user_input
                    assistant_message = None
                except anthropic.APIError as exc:
                    conversation.pop()
                    error = f"API error: {exc}"
                    prefill = user_input
                    assistant_message = None

                if assistant_message:
                    conversation.append({"role": "assistant", "content": assistant_message})
                elif not error:
                    conversation.pop()  # remove the unanswered user turn
                    error = "No text response received from the assistant."
                    prefill = user_input

                session["conversation"] = conversation
                session.modified = True

        return render_template_string(
            HTML_TEMPLATE,
            conversation=session.get("conversation", []),
            error=error,
            prefill=prefill,
        )

    @app.route("/clear")
    def clear():
        session.pop("conversation", None)
        return redirect(url_for("index"))

    return app


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
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
    client = anthropic.Anthropic(api_key=api_key)
    app = create_app(client, articles, vectorizer, doc_matrix, contacts_text)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nStarting web server at http://{host}:{port}/")
    print("Press Ctrl+C to stop.\n")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
