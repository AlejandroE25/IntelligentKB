"""
Article quality assessment using Claude.

Assessments are cached to quality_cache.json so Claude is only called for
articles that haven't been assessed yet (or when the cache version changes).

Cache invalidation:
  - Bump QUALITY_CACHE_VERSION to force a full re-assessment of all articles.
  - Per-article: if the article's 'updated' date is newer than 'assessed_at',
    the cached entry is dropped and the article is re-assessed.

Cost note: ~30 articles × ~700 tokens each ≈ $0.06 per full run (Claude Sonnet 4).
The cache ensures this cost is only paid when articles actually change.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

QUALITY_CACHE_VERSION = "v1"
BLOB_QUALITY_CACHE_NAME = "quality_cache.json"

_ASSESSMENT_PROMPT = """\
You are a knowledge base quality assessor for a university IT help desk.
Evaluate the following KB article and return a JSON object with exactly these fields:
- "overall_score": integer 1-5 (5=excellent and reliable, 1=completely unreliable or empty)
- "issues": list of strings describing specific quality problems (empty list if none found)
- "conflict_ids": list of article IDs from "Related Articles" below that this article CONFLICTS with (not just overlaps)
- "summary": one sentence describing the article's quality

Scoring guidance:
  5 = Clear, complete, current steps; trustworthy
  4 = Mostly good; minor gaps or slightly dated phrasing
  3 = Usable but has gaps, vague steps, or uncertain currency
  2 = Significant problems: missing steps, wrong UI paths, or partially contradicts another article
  1 = Unreliable: empty content, completely broken steps, or fundamentally conflicts with another article

Be specific in "issues". Common problems: steps refer to UI elements that may no longer exist,
instructions are vague without exact tool names, steps are missing, article contradicts a related
article on the same topic, content is a stub.

Only flag genuine problems — not minor style issues or preference differences.

Related articles (check these for conflicts only):
{related_summary}

Article to assess (ID: {article_id}):
Title: {title}
Keywords: {keywords}
Last updated: {updated}

Content:
{content}

Respond with ONLY valid JSON, no commentary before or after.
"""


@dataclass
class ArticleQuality:
    article_id: str
    assessed_at: str          # ISO date string (YYYY-MM-DD)
    cache_version: str
    overall_score: int        # 1–5
    issues: list[str] = field(default_factory=list)
    conflict_ids: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArticleQuality":
        return cls(
            article_id=d.get("article_id", ""),
            assessed_at=d.get("assessed_at", ""),
            cache_version=d.get("cache_version", ""),
            overall_score=int(d.get("overall_score", 3)),
            issues=d.get("issues", []),
            conflict_ids=d.get("conflict_ids", []),
            summary=d.get("summary", ""),
        )

    @classmethod
    def placeholder(cls, article_id: str) -> "ArticleQuality":
        """Return a neutral placeholder used when assessment fails."""
        return cls(
            article_id=article_id,
            assessed_at=date.today().isoformat(),
            cache_version=QUALITY_CACHE_VERSION,
            overall_score=3,
            issues=["Quality could not be assessed automatically."],
            conflict_ids=[],
            summary="Assessment unavailable.",
        )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_quality_cache(cache_path: Path) -> dict[str, ArticleQuality]:
    """Load quality assessments from cache_path.

    Drops entries whose cache_version does not match QUALITY_CACHE_VERSION.
    Returns dict keyed by article_id.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw: list[dict] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read quality cache at %s: %s", cache_path, exc)
        return {}

    result: dict[str, ArticleQuality] = {}
    for entry in raw:
        if entry.get("cache_version") != QUALITY_CACHE_VERSION:
            continue
        aq = ArticleQuality.from_dict(entry)
        if aq.article_id:
            result[aq.article_id] = aq
    return result


def save_quality_cache(
    cache_path: Path,
    assessments: dict[str, ArticleQuality],
    blob_service=None,
    data_container_name: str = "",
) -> None:
    """Persist assessments to cache_path (atomic write), then sync to blob if configured."""
    payload = [aq.to_dict() for aq in assessments.values()]
    tmp_path = cache_path.with_suffix(".json.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, cache_path)
    except OSError as exc:
        logger.warning("Could not write quality cache to %s: %s", cache_path, exc)
        return

    if blob_service and data_container_name:
        try:
            import blob_store as bs
            bs.upload_blob_json(data_container_name, BLOB_QUALITY_CACHE_NAME, payload, blob_service)
        except Exception as exc:
            logger.warning("Blob sync of quality cache failed: %s", exc)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def _build_related_summary(related_articles: list[dict]) -> str:
    lines = []
    for a in related_articles[:5]:
        aid = a.get("article_id", "?")
        title = a.get("title", "(untitled)")
        lines.append(f"  ID {aid}: {title}")
    return "\n".join(lines) if lines else "  (none)"


def assess_article_quality(
    client: anthropic.Anthropic,
    article: dict,
    related_articles: list[dict],
    stale_years: int = 2,
) -> ArticleQuality:
    """Call Claude once to assess a single article's quality.

    Uses a direct messages.create() call (not the agentic loop) with a
    JSON-response prompt. Returns a placeholder on any error.
    """
    article_id = article.get("article_id", "")
    content = article.get("content", "")

    prompt = _ASSESSMENT_PROMPT.format(
        article_id=article_id or "(unknown)",
        title=article.get("title", "(untitled)"),
        keywords=article.get("keywords", ""),
        updated=article.get("updated", "unknown"),
        content=content,
        related_summary=_build_related_summary(related_articles),
    )

    try:
        response = client.messages.create(
            model=os.environ.get("QUALITY_ASSESSMENT_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        # Strip markdown code fences if the model wraps its JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        parsed = json.loads(raw_text)
    except Exception as exc:
        logger.warning("Quality assessment failed for article %s: %s", article_id, exc)
        return ArticleQuality.placeholder(article_id)

    return ArticleQuality(
        article_id=article_id,
        assessed_at=date.today().isoformat(),
        cache_version=QUALITY_CACHE_VERSION,
        overall_score=max(1, min(5, int(parsed.get("overall_score", 3)))),
        issues=[str(i) for i in parsed.get("issues", [])],
        conflict_ids=[str(c) for c in parsed.get("conflict_ids", [])],
        summary=str(parsed.get("summary", "")),
    )


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def _is_cache_entry_fresh(aq: ArticleQuality, article: dict) -> bool:
    """Return True if the cached assessment is still valid for this article."""
    updated = article.get("updated", "")
    if not updated or not aq.assessed_at:
        return True
    try:
        return aq.assessed_at >= updated
    except TypeError:
        return True


def build_quality_cache(
    client: anthropic.Anthropic,
    articles: list[dict],
    cache_path: Path,
    force_refresh: bool = False,
    stale_years: int = 2,
    blob_service=None,
    data_container_name: str = "",
) -> dict[str, ArticleQuality]:
    """Build or load the quality cache at startup.

    Only calls Claude for articles not already in the cache (or when
    force_refresh=True or the article has been updated since last assessment).
    Saves updated cache after each new assessment.
    """
    # Try to seed from blob storage first
    if blob_service and data_container_name and not cache_path.exists():
        try:
            import blob_store as bs
            blob_data = bs.download_blob_json(data_container_name, BLOB_QUALITY_CACHE_NAME, blob_service)
            if blob_data is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(blob_data, f, indent=2)
                logger.info("Seeded quality cache from blob storage")
        except Exception as exc:
            logger.warning("Could not seed quality cache from blob: %s", exc)

    existing = {} if force_refresh else load_quality_cache(cache_path)
    article_id_set = {a.get("article_id", "") for a in articles}

    # Drop cached entries for articles that no longer exist
    existing = {aid: aq for aid, aq in existing.items() if aid in article_id_set}

    # Identify which articles need assessment
    to_assess = []
    for article in articles:
        aid = article.get("article_id", "")
        if not aid:
            continue
        cached = existing.get(aid)
        if cached is None or not _is_cache_entry_fresh(cached, article):
            to_assess.append(article)

    if not to_assess:
        logger.info("Quality cache is up to date (%d articles)", len(existing))
        return existing

    logger.info("Running quality assessment for %d article(s)...", len(to_assess))

    # Build a quick lookup for finding related articles by TF-IDF similarity
    # (simplified: just pick the 3 articles closest by index for now — the
    # full TF-IDF vectorizer isn't available here without adding a dependency)
    all_ids = [a.get("article_id", "") for a in articles]

    for i, article in enumerate(to_assess):
        aid = article.get("article_id", "")
        # Pick a few nearby articles as "related" for conflict checking
        related = [a for a in articles if a.get("article_id", "") != aid][:4]

        aq = assess_article_quality(client, article, related, stale_years)
        existing[aid] = aq
        save_quality_cache(
            cache_path, existing,
            blob_service=blob_service,
            data_container_name=data_container_name,
        )
        logger.info(
            "  Assessed article %s: score=%d issues=%d",
            aid, aq.overall_score, len(aq.issues),
        )

    logger.info("Quality assessment complete: %d articles in cache", len(existing))
    return existing


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def format_quality_warning(quality: Optional[ArticleQuality]) -> str:
    """Return an inline warning string for the system prompt, or '' if no issues."""
    if quality is None or quality.overall_score >= 4:
        return ""
    parts = [f"score {quality.overall_score}/5"]
    if quality.issues:
        parts.append(quality.issues[0])
    if quality.conflict_ids:
        parts.append(f"conflicts with article(s) {', '.join(quality.conflict_ids)}")
    return f"[QUALITY ALERT: {'; '.join(parts)}]"


def get_quality_badge(quality: Optional[ArticleQuality]) -> tuple[str, str]:
    """Return (label, css_class) for the quality badge in the article card.

    Returns ('', '') when quality is None or score is good (4-5).
    """
    if quality is None:
        return ("", "")
    if quality.overall_score <= 1:
        return ("Poor Quality 1/5", "badge-quality-poor")
    if quality.overall_score <= 3:
        return (f"Quality Issues {quality.overall_score}/5", "badge-quality-warn")
    return ("", "")
