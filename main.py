"""
KB AI Search Agent
==================
A locally-run web app where a help desk consultant describes a support problem
in plain English and receives relevant troubleshooting steps drawn from local KB
articles. All article content is loaded directly into Claude's context window.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import anthropic
from flask import Flask, redirect, render_template_string, request, session, url_for

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTICLES_DIR = Path(__file__).parent / "articles"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

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
# System Prompt Builder
# ---------------------------------------------------------------------------

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
    body { font-family: sans-serif; background: #f4f4f4; display: flex;
           flex-direction: column; height: 100vh; }
    header { background: #1a73e8; color: white; padding: 12px 20px;
             display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.1rem; }
    header a { color: white; font-size: 0.85rem; text-decoration: none;
               border: 1px solid rgba(255,255,255,0.6); padding: 4px 10px;
               border-radius: 4px; }
    header a:hover { background: rgba(255,255,255,0.15); }
    #chat { flex: 1; overflow-y: auto; padding: 16px; display: flex;
            flex-direction: column; gap: 12px; }
    .msg { max-width: 80%; padding: 10px 14px; border-radius: 8px;
           line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
    .msg.user { background: #1a73e8; color: white; align-self: flex-end; }
    .msg.assistant { background: white; border: 1px solid #ddd;
                     align-self: flex-start; }
    .msg.assistant a { color: #1a73e8; }
    .msg.error { background: #fdecea; border: 1px solid #f5c6cb;
                 color: #721c24; align-self: flex-start; }
    footer { background: white; border-top: 1px solid #ddd; padding: 12px 16px; }
    form { display: flex; gap: 8px; }
    textarea { flex: 1; padding: 8px 12px; border: 1px solid #ccc;
               border-radius: 6px; resize: none; font-size: 0.95rem;
               font-family: inherit; height: 44px; line-height: 1.4; }
    textarea:focus { outline: none; border-color: #1a73e8; }
    button[type=submit] { background: #1a73e8; color: white; border: none;
                          border-radius: 6px; padding: 0 18px; font-size: 0.95rem;
                          cursor: pointer; white-space: nowrap; }
    button[type=submit]:hover { background: #1558b0; }
    .empty-state { color: #888; text-align: center; margin: auto;
                   font-size: 0.95rem; }
  </style>
</head>
<body>
  <header>
    <h1>KB AI Search Agent</h1>
    <a href="/clear">New Conversation</a>
  </header>
  <div id="chat">
    {% if not conversation %}
      <p class="empty-state">Describe a support issue and press Enter (or click Send).</p>
    {% endif %}
    {% for msg in conversation %}
      <div class="msg {{ msg.role }}">{{ msg.content }}</div>
    {% endfor %}
    {% if error %}
      <div class="msg error">{{ error }}</div>
    {% endif %}
  </div>
  <footer>
    <form method="post" action="/">
      <textarea name="issue" placeholder="Describe the support issue…"
                autofocus>{{ prefill }}</textarea>
      <button type="submit">Send</button>
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


def create_app(client: anthropic.Anthropic, system_prompt: str) -> Flask:
    """Create and configure the Flask application."""
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

    system_prompt = build_system_prompt(articles)
    client = anthropic.Anthropic(api_key=api_key)
    app = create_app(client, system_prompt)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nStarting web server at http://{host}:{port}/")
    print("Press Ctrl+C to stop.\n")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
