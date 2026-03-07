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
    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --green:  #33ff33;
      --green2: #00cc00;
      --amber:  #ffaa00;
      --dim:    #1a5c1a;
      --bg:     #0d0d0d;
      --bg2:    #111111;
      --border: #225522;
    }

    body {
      font-family: "Courier New", Courier, monospace;
      background: var(--bg);
      color: var(--green);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }

    /* scanline flicker overlay */
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.07) 0px,
        rgba(0,0,0,0.07) 1px,
        transparent 1px,
        transparent 3px
      );
      pointer-events: none;
      z-index: 999;
    }

    /* ── HEADER ── */
    header {
      background: var(--bg2);
      border-bottom: 2px solid var(--green2);
      padding: 6px 12px;
      flex-shrink: 0;
    }
    .hdr-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
    }
    header h1 {
      font-size: 1rem;
      font-weight: bold;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--green);
      text-shadow: 0 0 8px var(--green2);
    }
    header a {
      color: var(--amber);
      font-size: 0.75rem;
      text-decoration: none;
      border: 1px solid var(--amber);
      padding: 2px 8px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    header a:hover { background: var(--amber); color: #000; }
    .hdr-rule {
      font-size: 0.65rem;
      color: var(--dim);
      letter-spacing: 1px;
      margin-top: 3px;
      user-select: none;
    }

    /* ── TRANSCRIPT ── */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 0;
    }
    /* custom scrollbar */
    #chat::-webkit-scrollbar { width: 8px; }
    #chat::-webkit-scrollbar-track { background: var(--bg); }
    #chat::-webkit-scrollbar-thumb { background: var(--green2); }

    .entry { margin-bottom: 10px; }

    .entry-label {
      font-size: 0.7rem;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 2px;
    }
    .entry.user   .entry-label { color: var(--amber); }
    .entry.assistant .entry-label { color: var(--green2); }
    .entry.error  .entry-label { color: #ff3333; }

    .entry-body {
      white-space: pre-wrap;
      word-wrap: break-word;
      line-height: 1.55;
      font-size: 0.9rem;
      padding: 6px 10px;
      border-left: 3px solid;
    }
    .entry.user      .entry-body { border-color: var(--amber);  color: #ffd080; }
    .entry.assistant .entry-body { border-color: var(--green2); color: var(--green); }
    .entry.error     .entry-body { border-color: #ff3333; color: #ff6666; }

    .entry.assistant .entry-body a {
      color: var(--amber);
      text-decoration: underline;
    }
    .entry.assistant .entry-body a:hover { color: #fff; }

    .empty-state {
      color: var(--dim);
      font-size: 0.85rem;
      letter-spacing: 1px;
      margin: auto;
      text-align: center;
    }
    .cursor { animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    @media (prefers-reduced-motion: reduce) {
      .cursor { animation: none; }
    }

    /* ── FOOTER / INPUT ── */
    footer {
      background: var(--bg2);
      border-top: 2px solid var(--green2);
      padding: 8px 12px;
      flex-shrink: 0;
    }
    .prompt-line {
      font-size: 0.7rem;
      color: var(--dim);
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    form { display: flex; gap: 8px; align-items: flex-end; }
    textarea {
      flex: 1;
      background: #000;
      color: var(--green);
      border: 1px solid var(--green2);
      padding: 6px 10px;
      font-family: inherit;
      font-size: 0.9rem;
      resize: none;
      height: 42px;
      line-height: 1.45;
      caret-color: var(--green);
      outline: none;
    }
    textarea::placeholder { color: var(--dim); }
    textarea:focus { border-color: var(--green); box-shadow: 0 0 6px var(--green2); }
    button[type=submit] {
      background: #000;
      color: var(--green);
      border: 1px solid var(--green2);
      font-family: inherit;
      font-size: 0.85rem;
      letter-spacing: 2px;
      text-transform: uppercase;
      padding: 0 14px;
      height: 42px;
      cursor: pointer;
      white-space: nowrap;
    }
    button[type=submit]:hover {
      background: var(--green2);
      color: #000;
      box-shadow: 0 0 8px var(--green2);
    }
  </style>
</head>
<body>
  <header>
    <div class="hdr-top">
      <h1><span aria-hidden="true">&#x25A0;</span> KB AI Search Agent v1.0</h1>
      <a href="/clear">[NEW SESSION]</a>
    </div>
    <div class="hdr-rule">
      &gt;&gt; HELP DESK KNOWLEDGE BASE TERMINAL &lt;&lt;
      &nbsp;&nbsp;///&nbsp;&nbsp;
      TYPE QUERY. PRESS ENTER.
    </div>
  </header>

  <div id="chat">
    {% if not conversation %}
      <p class="empty-state">
        C:\\HELPDESK&gt; _<span class="cursor" aria-hidden="true">&#x2588;</span><br><br>
        SYSTEM READY. ENTER SUPPORT ISSUE BELOW.
      </p>
    {% endif %}
    {% for msg in conversation %}
      <div class="entry {{ msg.role }}">
        <div class="entry-label">
          {% if msg.role == 'user' %}
            &gt;&gt; USER INPUT
          {% else %}
            &lt;&lt; KB AGENT
          {% endif %}
        </div>
        <div class="entry-body">{{ msg.content }}</div>
      </div>
    {% endfor %}
    {% if error %}
      <div class="entry error">
        <div class="entry-label">!! ERROR</div>
        <div class="entry-body">{{ error }}</div>
      </div>
    {% endif %}
  </div>

  <footer>
    <div class="prompt-line">C:\\HELPDESK&gt; enter query:</div>
    <form method="post" action="/">
      <textarea name="issue" placeholder="Describe the support issue..."
                autofocus>{{ prefill }}</textarea>
      <button type="submit">[SEND]</button>
    </form>
  </footer>

  <script>
    // Auto-scroll to bottom on load
    const chat = document.getElementById('chat');
    chat.scrollTop = chat.scrollHeight;
    // Enter submits the form; Shift+Enter inserts a newline
    const ta = document.querySelector('textarea');
    ta.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        ta.closest('form').submit();
      }
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
