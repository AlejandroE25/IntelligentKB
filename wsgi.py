import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path
import anthropic
from main import (
    load_articles, build_article_index, create_app,
    ARTICLES_DIR, STALE_ARTICLE_YEARS, _SEARCH_ENHANCEMENT_AVAILABLE,
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
brave_api_key = os.environ.get("BRAVE_API_KEY", "")

from blob_store import get_blob_service_client, download_articles_from_blob
from feedback_store import FeedbackStore
from quality import build_quality_cache

blob_service = get_blob_service_client()
articles_container = os.environ.get("AZURE_STORAGE_ARTICLES_CONTAINER", "")
data_container = os.environ.get("AZURE_STORAGE_DATA_CONTAINER", "")

if blob_service and articles_container:
    tmp_dir = Path(tempfile.mkdtemp(prefix="kb_articles_"))
    try:
        count = download_articles_from_blob(articles_container, tmp_dir, blob_service)
        if count > 0:
            blob_articles, blob_contacts = load_articles(tmp_dir)
            if blob_articles:
                articles = blob_articles
                contacts_text = blob_contacts
                vectorizer, doc_matrix = build_article_index(articles)
                if retriever is not None:
                    retriever.build(articles, vectorizer, doc_matrix)
    except Exception:
        pass  # Fall back to local articles on any blob error

feedback_path = Path(__file__).parent / "feedback.json"
quality_cache_path = Path(__file__).parent / "quality_cache.json"

feedback_store = FeedbackStore(feedback_path, blob_service=blob_service, data_container_name=data_container)
if blob_service and data_container:
    feedback_store.load_from_blob(blob_service, data_container)

quality_assessments: dict = {}
if not os.environ.get("SKIP_QUALITY_ASSESSMENT"):
    quality_assessments = build_quality_cache(
        client, articles, quality_cache_path,
        stale_years=STALE_ARTICLE_YEARS,
        blob_service=blob_service,
        data_container_name=data_container,
    )

app = create_app(
    client, articles, vectorizer, doc_matrix, contacts_text, retriever, brave_api_key,
    feedback_store=feedback_store,
    quality_assessments=quality_assessments,
)
if blob_service:
    app.config["BLOB_SERVICE"] = blob_service
    app.config["ARTICLES_CONTAINER"] = articles_container
