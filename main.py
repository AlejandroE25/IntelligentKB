"""
KB AI Search Agent
==================
A locally-run CLI tool where a help desk consultant describes a support problem
in plain English and receives relevant troubleshooting steps drawn from local KB
articles. All article content is loaded directly into Claude's context window.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTICLES_DIR = Path(__file__).parent / "articles"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

SYSTEM_PROMPT_HEADER = (
    "You are a help desk assistant. Below are the contents of our Knowledge Base articles.\n"
    "When given a support issue, provide concise troubleshooting steps using only the\n"
    "information in these articles. Cite the source article for each step. If the articles\n"
    "don't contain enough information to fully resolve the issue, say so.\n"
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

    # 3. Extract title
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
        lines.append(f"[ARTICLE: {article['filename']}]")
        if article["keywords"]:
            lines.append(f"Keywords: {article['keywords']}")
        lines.append("---")
        lines.append(article["content"])
        lines.append("")  # blank separator between articles
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Loop
# ---------------------------------------------------------------------------

def run_cli(client: anthropic.Anthropic, system_prompt: str) -> None:
    """Run the interactive support query loop."""
    conversation: list[dict[str, str]] = []

    print("KB AI Search Agent — type your support issue and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Enter your support issue: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=conversation,
        )

        content_blocks = [b for b in response.content if hasattr(b, "text")]
        if not content_blocks:
            print("(No text response received from the assistant.)\n")
            conversation.pop()  # remove the user turn so history stays consistent
            continue
        assistant_message = content_blocks[0].text
        conversation.append({"role": "assistant", "content": assistant_message})

        print(f"\n{assistant_message}\n")


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

    print(f"Loaded {len(articles)} KB article(s): {', '.join(a['filename'] for a in articles)}")

    system_prompt = build_system_prompt(articles)

    client = anthropic.Anthropic(api_key=api_key)
    run_cli(client, system_prompt)


if __name__ == "__main__":
    main()
