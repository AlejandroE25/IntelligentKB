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
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

# Maximum number of articles sent to Claude per query.
# Raising this improves recall at the cost of larger context windows.
TOP_K_ARTICLES = 3

KB_BASE_URL = "https://answers.uillinois.edu/illinois/internal"

SYSTEM_PROMPT_HEADER = (
    "You are a help desk assistant. Below are the contents of our Knowledge Base articles.\n"
    "When given a support issue, provide concise troubleshooting steps using only the\n"
    "information in these articles. Cite the source article for each step using a markdown\n"
    f"clickable link in the format [Article XXXXX]({KB_BASE_URL}/XXXXX) where XXXXX is the\n"
    "article number. If the articles don't contain enough information to fully resolve the\n"
    "issue, say so.\n"
)


# ---------------------------------------------------------------------------
# HTML Parsing
# ---------------------------------------------------------------------------

def parse_article(path: Path) -> dict[str, str]:
    """Parse a UW-Madison-style KB HTML article and return a dict with:
       - filename:  base name of the file
       - title:     page title
       - keywords:  keywords string
       - content:   cleaned plain-text main content
       - internal:  internal-staff section text (may be empty)
    """
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # 1. Strip noise elements before any text extraction
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
    }


def load_articles(articles_dir: Path) -> list[dict[str, str]]:
    """Load and parse all .htm and .html files from the given directory."""
    articles = []
    for ext in ("*.htm", "*.html"):
        for filepath in sorted(articles_dir.glob(ext)):
            articles.append(parse_article(filepath))
    return articles

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


def build_system_prompt(articles: list[dict[str, str]]) -> str:
    """Construct the full system prompt with all KB article content."""
    lines = [SYSTEM_PROMPT_HEADER, "", "--- KNOWLEDGE BASE ---", ""]
    for article in articles:
        article_id = article["article_id"]
        article_url = f"{KB_BASE_URL}/{article_id}" if article_id else ""
        label = f"Article {article_id} ({article_url})" if article_id else article["filename"]
        lines.append(f"[ARTICLE: {label}]")
        if article["keywords"]:
            lines.append(f"Keywords: {article['keywords']}")
        lines.append("---")
        lines.append(article["content"])
        lines.append("")  # blank separator between articles
    return "\n".join(lines)


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
) -> Flask:
    """Create and configure the Flask application.

    For each incoming query the TF-IDF index is used to select only the most
    relevant articles before building the system prompt, keeping context size
    small and reducing API cost.
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
                system_prompt = build_system_prompt(relevant)

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    messages=conversation,
                )

                content_blocks = [b for b in response.content if hasattr(b, "text")]
                if content_blocks:
                    assistant_message = content_blocks[0].text
                    conversation.append({"role": "assistant", "content": assistant_message})
                else:
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

    articles = load_articles(ARTICLES_DIR)
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
    app = create_app(client, articles, vectorizer, doc_matrix)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nStarting web server at http://{host}:{port}/")
    print("Press Ctrl+C to stop.\n")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
