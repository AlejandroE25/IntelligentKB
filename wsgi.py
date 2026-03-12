import os
from dotenv import load_dotenv
from pathlib import Path
import anthropic
from main import (
    load_articles, build_article_index, create_app,
    ARTICLES_DIR, _SEARCH_ENHANCEMENT_AVAILABLE
)

load_dotenv()

articles, contacts_text = load_articles(ARTICLES_DIR)
vectorizer, doc_matrix = build_article_index(articles)

retriever = None
if _SEARCH_ENHANCEMENT_AVAILABLE:
    from search_enhancement import FeatureFlags, HybridRetriever
    flags = FeatureFlags()
    if flags.any_enabled():
        retriever = HybridRetriever(flags)
        retriever.build(articles, vectorizer, doc_matrix)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
app = create_app(client, articles, vectorizer, doc_matrix, contacts_text, retriever)