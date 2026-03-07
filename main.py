"""
KB AI Search Agent
==================
A locally-run CLI tool where a help desk consultant describes a support problem
in plain English and receives relevant troubleshooting steps drawn from local KB
articles. TF-IDF pre-filtering is used to select only the most relevant articles
for each query before sending them to Claude, reducing context size and cost.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

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
# CLI Loop
# ---------------------------------------------------------------------------

def run_cli(
    client: anthropic.Anthropic,
    articles: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
) -> None:
    """Run the interactive support query loop.

    For each user query, TF-IDF cosine similarity is used to select the
    most relevant articles before building the system prompt, so that only
    a small subset of the KB is sent to Claude rather than the full corpus.
    """
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

        # Select only the most relevant articles for this query.
        relevant = select_relevant_articles(user_input, articles, vectorizer, doc_matrix)
        system_prompt = build_system_prompt(relevant)

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

    print(f"Loaded {len(articles)} KB article(s): \n    {'\n    '.join(a['filename'] for a in articles)}")

    vectorizer, doc_matrix = build_article_index(articles)

    client = anthropic.Anthropic(api_key=api_key)
    run_cli(client, articles, vectorizer, doc_matrix)


if __name__ == "__main__":
    main()
