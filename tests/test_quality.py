"""Tests for quality.py — Claude API calls are fully mocked."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import quality
from quality import (
    ArticleQuality,
    QUALITY_CACHE_VERSION,
    assess_article_quality,
    build_quality_cache,
    format_quality_warning,
    get_quality_badge,
    load_quality_cache,
    save_quality_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_article():
    return {
        "article_id": "90275",
        "title": "WiFi Troubleshooting",
        "keywords": "wifi wireless network",
        "content": "Step 1: Open settings. Step 2: Connect.",
        "updated": "2024-01-15",
        "owner": "Networking",
    }


@pytest.fixture()
def related_articles():
    return [
        {"article_id": "90276", "title": "Eduroam Setup"},
        {"article_id": "90277", "title": "VPN Access"},
    ]


def _mock_client(score=4, issues=None, conflict_ids=None, summary="Good article"):
    """Return a mock Anthropic client that returns a quality assessment JSON."""
    payload = {
        "overall_score": score,
        "issues": issues or [],
        "conflict_ids": conflict_ids or [],
        "summary": summary,
    }
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(payload))]
    client = MagicMock()
    client.messages.create.return_value = mock_response
    return client


# ---------------------------------------------------------------------------
# ArticleQuality
# ---------------------------------------------------------------------------

class TestArticleQuality:
    def test_placeholder_has_score_3(self):
        aq = ArticleQuality.placeholder("12345")
        assert aq.overall_score == 3
        assert aq.cache_version == QUALITY_CACHE_VERSION

    def test_round_trip_dict(self, sample_article):
        aq = ArticleQuality(
            article_id="90275",
            assessed_at="2024-06-01",
            cache_version=QUALITY_CACHE_VERSION,
            overall_score=4,
            issues=["Minor gap"],
            conflict_ids=["90277"],
            summary="Mostly good",
        )
        restored = ArticleQuality.from_dict(aq.to_dict())
        assert restored.article_id == aq.article_id
        assert restored.overall_score == aq.overall_score
        assert restored.issues == aq.issues
        assert restored.conflict_ids == aq.conflict_ids


# ---------------------------------------------------------------------------
# load_quality_cache / save_quality_cache
# ---------------------------------------------------------------------------

class TestLoadQualityCache:
    def test_returns_empty_on_missing_file(self, tmp_path):
        result = load_quality_cache(tmp_path / "nonexistent.json")
        assert result == {}

    def test_loads_valid_entries(self, tmp_path):
        path = tmp_path / "cache.json"
        entries = [
            ArticleQuality(
                article_id="90275", assessed_at="2024-01-01",
                cache_version=QUALITY_CACHE_VERSION, overall_score=4,
                summary="OK",
            ).to_dict()
        ]
        path.write_text(json.dumps(entries), encoding="utf-8")
        result = load_quality_cache(path)
        assert "90275" in result
        assert result["90275"].overall_score == 4

    def test_drops_entries_with_old_cache_version(self, tmp_path):
        path = tmp_path / "cache.json"
        entries = [
            {
                "article_id": "90275",
                "assessed_at": "2024-01-01",
                "cache_version": "v0",
                "overall_score": 5,
                "issues": [],
                "conflict_ids": [],
                "summary": "old",
            }
        ]
        path.write_text(json.dumps(entries), encoding="utf-8")
        result = load_quality_cache(path)
        assert result == {}

    def test_handles_corrupt_json(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("not json", encoding="utf-8")
        result = load_quality_cache(path)
        assert result == {}


class TestSaveQualityCache:
    def test_saves_and_reloads(self, tmp_path):
        path = tmp_path / "cache.json"
        aq = ArticleQuality(
            article_id="90275", assessed_at="2024-01-01",
            cache_version=QUALITY_CACHE_VERSION, overall_score=3,
            summary="Poor",
        )
        save_quality_cache(path, {"90275": aq})
        loaded = load_quality_cache(path)
        assert loaded["90275"].overall_score == 3

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "subdir" / "cache.json"
        save_quality_cache(path, {})
        assert path.exists()


# ---------------------------------------------------------------------------
# assess_article_quality
# ---------------------------------------------------------------------------

class TestAssessArticleQuality:
    def test_calls_claude_once(self, sample_article, related_articles):
        client = _mock_client(score=4)
        assess_article_quality(client, sample_article, related_articles)
        assert client.messages.create.call_count == 1

    def test_parses_response_correctly(self, sample_article, related_articles):
        client = _mock_client(
            score=2,
            issues=["Steps are vague"],
            conflict_ids=["90277"],
            summary="Needs work",
        )
        aq = assess_article_quality(client, sample_article, related_articles)
        assert aq.overall_score == 2
        assert "Steps are vague" in aq.issues
        assert "90277" in aq.conflict_ids
        assert aq.summary == "Needs work"
        assert aq.article_id == "90275"
        assert aq.cache_version == QUALITY_CACHE_VERSION

    def test_returns_placeholder_on_api_error(self, sample_article, related_articles):
        client = MagicMock()
        client.messages.create.side_effect = Exception("API error")
        aq = assess_article_quality(client, sample_article, related_articles)
        assert aq.overall_score == 3
        assert aq.article_id == "90275"

    def test_returns_placeholder_on_malformed_json(self, sample_article, related_articles):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="this is not json at all")]
        client = MagicMock()
        client.messages.create.return_value = mock_response
        aq = assess_article_quality(client, sample_article, related_articles)
        assert aq.overall_score == 3

    def test_handles_markdown_wrapped_json(self, sample_article, related_articles):
        payload = {"overall_score": 5, "issues": [], "conflict_ids": [], "summary": "Great"}
        wrapped = f"```json\n{json.dumps(payload)}\n```"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=wrapped)]
        client = MagicMock()
        client.messages.create.return_value = mock_response
        aq = assess_article_quality(client, sample_article, related_articles)
        assert aq.overall_score == 5

    def test_clamps_score_to_1_5(self, sample_article, related_articles):
        client = _mock_client(score=99)
        aq = assess_article_quality(client, sample_article, related_articles)
        assert aq.overall_score == 5

        client2 = _mock_client(score=-5)
        aq2 = assess_article_quality(client2, sample_article, related_articles)
        assert aq2.overall_score == 1


# ---------------------------------------------------------------------------
# build_quality_cache
# ---------------------------------------------------------------------------

class TestBuildQualityCache:
    def test_skips_already_cached_articles(self, tmp_path, sample_article):
        path = tmp_path / "cache.json"
        existing = ArticleQuality(
            article_id="90275",
            assessed_at="2099-01-01",  # future date, always fresh
            cache_version=QUALITY_CACHE_VERSION,
            overall_score=4,
            summary="cached",
        )
        save_quality_cache(path, {"90275": existing})

        client = _mock_client()
        result = build_quality_cache(client, [sample_article], path)
        assert client.messages.create.call_count == 0
        assert result["90275"].summary == "cached"

    def test_adds_new_articles_to_cache(self, tmp_path, sample_article):
        path = tmp_path / "cache.json"
        client = _mock_client(score=3, summary="Assessed")
        result = build_quality_cache(client, [sample_article], path)
        assert client.messages.create.call_count == 1
        assert result["90275"].summary == "Assessed"

    def test_re_assesses_when_article_updated(self, tmp_path, sample_article):
        path = tmp_path / "cache.json"
        stale = ArticleQuality(
            article_id="90275",
            assessed_at="2020-01-01",  # older than article.updated
            cache_version=QUALITY_CACHE_VERSION,
            overall_score=5,
            summary="old",
        )
        save_quality_cache(path, {"90275": stale})

        client = _mock_client(score=2, summary="Re-assessed")
        result = build_quality_cache(client, [sample_article], path)
        assert client.messages.create.call_count == 1
        assert result["90275"].summary == "Re-assessed"

    def test_saves_updated_cache_to_file(self, tmp_path, sample_article):
        path = tmp_path / "cache.json"
        client = _mock_client(score=4)
        build_quality_cache(client, [sample_article], path)
        assert path.exists()
        loaded = load_quality_cache(path)
        assert "90275" in loaded

    def test_force_refresh_ignores_cache(self, tmp_path, sample_article):
        path = tmp_path / "cache.json"
        fresh = ArticleQuality(
            article_id="90275",
            assessed_at="2099-01-01",
            cache_version=QUALITY_CACHE_VERSION,
            overall_score=5,
            summary="cached",
        )
        save_quality_cache(path, {"90275": fresh})

        client = _mock_client(score=1, summary="Fresh")
        result = build_quality_cache(client, [sample_article], path, force_refresh=True)
        assert client.messages.create.call_count == 1
        assert result["90275"].summary == "Fresh"


# ---------------------------------------------------------------------------
# format_quality_warning / get_quality_badge
# ---------------------------------------------------------------------------

class TestFormatQualityWarning:
    def test_empty_for_none(self):
        assert format_quality_warning(None) == ""

    def test_empty_for_high_score(self):
        aq = ArticleQuality.placeholder("x")
        aq.overall_score = 4
        aq.issues = []
        assert format_quality_warning(aq) == ""

    def test_returns_warning_for_low_score(self):
        aq = ArticleQuality(
            article_id="x", assessed_at="2024-01-01",
            cache_version=QUALITY_CACHE_VERSION, overall_score=2,
            issues=["Steps are wrong"],
        )
        warning = format_quality_warning(aq)
        assert "[QUALITY ALERT:" in warning
        assert "score 2/5" in warning
        assert "Steps are wrong" in warning

    def test_includes_conflict_ids(self):
        aq = ArticleQuality(
            article_id="x", assessed_at="2024-01-01",
            cache_version=QUALITY_CACHE_VERSION, overall_score=1,
            conflict_ids=["90277", "90280"],
        )
        warning = format_quality_warning(aq)
        assert "90277" in warning
        assert "90280" in warning


class TestGetQualityBadge:
    def test_returns_empty_for_none(self):
        label, css = get_quality_badge(None)
        assert label == ""
        assert css == ""

    def test_returns_empty_for_good_score(self):
        aq = ArticleQuality.placeholder("x")
        aq.overall_score = 4
        label, css = get_quality_badge(aq)
        assert label == ""

    def test_poor_quality_badge_for_score_1(self):
        aq = ArticleQuality.placeholder("x")
        aq.overall_score = 1
        label, css = get_quality_badge(aq)
        assert css == "badge-quality-poor"
        assert "1/5" in label

    def test_warn_badge_for_score_2_or_3(self):
        for score in (2, 3):
            aq = ArticleQuality.placeholder("x")
            aq.overall_score = score
            label, css = get_quality_badge(aq)
            assert css == "badge-quality-warn"
            assert str(score) in label
