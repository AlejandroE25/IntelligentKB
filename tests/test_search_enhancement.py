"""Tests for search_enhancement.py"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from search_enhancement import (
    DOMAIN_SYNONYMS,
    FeatureFlags,
    HybridRetriever,
    QueryNormalizer,
    SearchCache,
    SearchTimings,
    SemanticIndex,
    _lexical_retrieve,
    adaptive_top_k,
    ndcg_at_k,
    recall_at_k,
    rrf_fusion,
)
from main import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ARTICLES = [
    {
        "article_id": "1001",
        "title": "Duo MFA two-factor authentication setup",
        "keywords": "duo mfa 2fa authentication",
        "content": "This article explains how to set up Duo multi-factor authentication.",
        "filename": "duo.html",
        "updated": "2024-01-01",
        "owner": "IT Security",
    },
    {
        "article_id": "1002",
        "title": "VPN connection guide",
        "keywords": "vpn cisco anyconnect remote",
        "content": "Install Cisco AnyConnect and connect to the university VPN.",
        "filename": "vpn.html",
        "updated": "2024-02-01",
        "owner": "Networking",
    },
    {
        "article_id": "1003",
        "title": "Eduroam WiFi setup",
        "keywords": "wifi wireless eduroam network",
        "content": "Connect your device to the eduroam wireless network on campus.",
        "filename": "wifi.html",
        "updated": "2024-03-01",
        "owner": "Networking",
    },
    {
        "article_id": "1004",
        "title": "NetID password reset",
        "keywords": "netid password reset login credentials",
        "content": "Reset your NetID password via the self-service portal.",
        "filename": "netid.html",
        "updated": "2024-04-01",
        "owner": "Identity",
    },
    {
        "article_id": "1005",
        "title": "Email and Outlook setup",
        "keywords": "email outlook office365 microsoft",
        "content": "Configure Outlook for your university email account.",
        "filename": "email.html",
        "updated": "2024-05-01",
        "owner": "Email Team",
    },
]


@pytest.fixture
def vectorizer_and_matrix():
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    corpus = [
        f"{a['title']} {a['keywords']} {a['content']}"
        for a in SAMPLE_ARTICLES
    ]
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


@pytest.fixture
def all_flags_off(monkeypatch):
    """Ensure all feature flags are disabled."""
    for key in [
        "FEATURE_HYBRID_RETRIEVAL",
        "FEATURE_QUERY_NORMALIZATION",
        "FEATURE_ADAPTIVE_TOPK",
        "FEATURE_SEARCH_CACHE",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def all_flags_on(monkeypatch):
    """Enable all feature flags."""
    for key in [
        "FEATURE_HYBRID_RETRIEVAL",
        "FEATURE_QUERY_NORMALIZATION",
        "FEATURE_ADAPTIVE_TOPK",
        "FEATURE_SEARCH_CACHE",
    ]:
        monkeypatch.setenv(key, "1")


# ---------------------------------------------------------------------------
# FeatureFlags
# ---------------------------------------------------------------------------

class TestFeatureFlags:
    def test_all_flags_off_by_default(self, all_flags_off):
        flags = FeatureFlags()
        assert not flags.hybrid_retrieval
        assert not flags.query_normalization
        assert not flags.adaptive_topk
        assert not flags.search_cache
        assert not flags.any_enabled()

    def test_flags_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("FEATURE_HYBRID_RETRIEVAL", "1")
        monkeypatch.setenv("FEATURE_QUERY_NORMALIZATION", "true")
        monkeypatch.setenv("FEATURE_ADAPTIVE_TOPK", "yes")
        monkeypatch.setenv("FEATURE_SEARCH_CACHE", "1")
        flags = FeatureFlags()
        assert flags.hybrid_retrieval
        assert flags.query_normalization
        assert flags.adaptive_topk
        assert flags.search_cache
        assert flags.any_enabled()

    def test_reload_picks_up_new_env(self, monkeypatch):
        monkeypatch.delenv("FEATURE_HYBRID_RETRIEVAL", raising=False)
        flags = FeatureFlags()
        assert not flags.hybrid_retrieval
        monkeypatch.setenv("FEATURE_HYBRID_RETRIEVAL", "1")
        flags.reload()
        assert flags.hybrid_retrieval


# ---------------------------------------------------------------------------
# QueryNormalizer
# ---------------------------------------------------------------------------

class TestQueryNormalizer:
    def setup_method(self):
        self.norm = QueryNormalizer()

    def test_basic_normalize_lowercases(self):
        assert self.norm.basic_normalize("VPN") == "vpn"

    def test_basic_normalize_strips_punctuation(self):
        result = self.norm.basic_normalize("duo, MFA! help?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_basic_normalize_collapses_whitespace(self):
        result = self.norm.basic_normalize("  too   many   spaces  ")
        assert result == "too many spaces"

    def test_basic_normalize_preserves_apostrophes(self):
        result = self.norm.basic_normalize("can't login")
        assert "'" in result

    def test_basic_normalize_preserves_hyphens(self):
        result = self.norm.basic_normalize("two-factor")
        assert "-" in result

    def test_expand_synonyms_duo(self):
        result = self.norm.expand_synonyms("duo not working")
        # Should contain at least one synonym for 'duo'
        assert "mfa" in result or "two-factor" in result or "2fa" in result

    def test_expand_synonyms_no_match_unchanged(self):
        result = self.norm.expand_synonyms("article about printing")
        assert result == "article about printing"

    def test_expand_synonyms_two_word_phrase(self):
        result = self.norm.expand_synonyms("net id login issue")
        # "net id" is in DOMAIN_SYNONYMS
        assert "netid" in result or "username" in result

    def test_normalize_with_synonyms(self):
        result = self.norm.normalize("VPN not connecting", use_synonyms=True)
        # should be lowercase and include vpn synonyms
        assert "vpn" in result
        assert result == result.lower()

    def test_normalize_without_synonyms(self):
        result = self.norm.normalize("VPN not connecting", use_synonyms=False)
        assert result == "vpn not connecting"
        assert "anyconnect" not in result

    def test_typo_correct_known_word_unchanged(self):
        result = self.norm.typo_correct("duo")
        assert result == "duo"

    def test_typo_correct_short_token_unchanged(self):
        result = self.norm.typo_correct("do")
        assert result == "do"

    def test_normalize_with_typo(self):
        # "vpn" is a known word; should not be changed
        result = self.norm.normalize("vpn", use_synonyms=False, use_typo=True)
        assert "vpn" in result


# ---------------------------------------------------------------------------
# SearchTimings
# ---------------------------------------------------------------------------

class TestSearchTimings:
    def test_to_dict_contains_all_keys(self):
        t = SearchTimings(total_ms=15.5, lexical_ms=10.0)
        d = t.to_dict()
        assert "total_ms" in d
        assert "lexical_ms" in d
        assert "semantic_ms" in d
        assert d["total_ms"] == 15.5

    def test_confidence_high(self):
        t = SearchTimings()
        results = [("1001", 0.35), ("1002", 0.10)]
        assert t.confidence(results) == "high"

    def test_confidence_medium(self):
        t = SearchTimings()
        results = [("1001", 0.15), ("1002", 0.05)]
        assert t.confidence(results) == "medium"

    def test_confidence_low(self):
        t = SearchTimings()
        results = [("1001", 0.03)]
        assert t.confidence(results) == "low"

    def test_confidence_empty(self):
        t = SearchTimings()
        assert t.confidence([]) == "none"


# ---------------------------------------------------------------------------
# rrf_fusion
# ---------------------------------------------------------------------------

class TestRrfFusion:
    def test_returns_all_unique_ids(self):
        lex = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        sem = [("b", 0.8), ("c", 0.6), ("d", 0.4)]
        result = rrf_fusion(lex, sem)
        ids = [x[0] for x in result]
        assert set(ids) == {"a", "b", "c", "d"}

    def test_item_in_both_lists_ranks_higher(self):
        lex = [("a", 0.9), ("b", 0.7)]
        sem = [("b", 0.8), ("c", 0.6)]
        result = rrf_fusion(lex, sem)
        ids = [x[0] for x in result]
        # 'b' appears in both lists, should be ranked above 'c' (only in semantic)
        assert ids.index("b") < ids.index("c")

    def test_sorted_descending(self):
        lex = [("a", 0.9), ("b", 0.5)]
        sem = [("a", 0.8), ("b", 0.3)]
        result = rrf_fusion(lex, sem)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_semantic_returns_lexical_order(self):
        lex = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        result = rrf_fusion(lex, [])
        ids = [x[0] for x in result]
        assert ids == ["a", "b", "c"]

    def test_empty_lexical_returns_semantic_order(self):
        sem = [("x", 0.9), ("y", 0.7)]
        result = rrf_fusion([], sem)
        ids = [x[0] for x in result]
        assert ids == ["x", "y"]

    def test_alpha_zero_uses_only_semantic(self):
        lex = [("a", 0.9), ("b", 0.5)]
        sem = [("b", 0.9), ("a", 0.1)]
        result = rrf_fusion(lex, sem, alpha=0.0)
        ids = [x[0] for x in result]
        assert ids[0] == "b"  # semantic-top item first


# ---------------------------------------------------------------------------
# adaptive_top_k
# ---------------------------------------------------------------------------

class TestAdaptiveTopK:
    def test_returns_min_k_when_insufficient_scores(self):
        assert adaptive_top_k([0.8], min_k=3) == 1  # fewer than min_k available

    def test_returns_min_k_exactly(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        # No gap large enough; should return max_k
        result = adaptive_top_k(scores, min_k=3, max_k=5, gap_threshold=0.5)
        assert result == 5

    def test_stops_at_significant_gap(self):
        # Gap between index 3 and 4: 0.7 - 0.1 = 0.6 >= threshold 0.5
        scores = [0.9, 0.85, 0.8, 0.7, 0.1, 0.05]
        result = adaptive_top_k(scores, min_k=2, max_k=6, gap_threshold=0.5)
        assert result == 4

    def test_never_below_min_k(self):
        scores = [0.9, 0.0, 0.0]
        result = adaptive_top_k(scores, min_k=3, max_k=5)
        assert result >= 3

    def test_never_above_max_k(self):
        scores = list(range(20, 0, -1))
        result = adaptive_top_k(scores, min_k=2, max_k=5)
        assert result <= 5

    def test_equal_scores_returns_max_k(self):
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = adaptive_top_k(scores, min_k=2, max_k=5, gap_threshold=0.05)
        assert result == 5


# ---------------------------------------------------------------------------
# SearchCache
# ---------------------------------------------------------------------------

class TestSearchCache:
    def test_miss_on_empty_cache(self):
        cache = SearchCache()
        assert cache.get("query", "v1") is None

    def test_set_and_get(self):
        cache = SearchCache()
        cache.set("query", ["a", "b"], "v1")
        assert cache.get("query", "v1") == ["a", "b"]

    def test_different_version_misses(self):
        cache = SearchCache()
        cache.set("query", ["a"], "v1")
        assert cache.get("query", "v2") is None

    def test_evicts_oldest_when_full(self):
        cache = SearchCache(maxsize=2)
        cache.set("q1", 1, "v1")
        cache.set("q2", 2, "v1")
        cache.set("q3", 3, "v1")  # should evict q1
        assert cache.get("q1", "v1") is None
        assert cache.get("q2", "v1") == 2
        assert cache.get("q3", "v1") == 3

    def test_lru_eviction_order(self):
        cache = SearchCache(maxsize=2)
        cache.set("q1", 1, "v1")
        cache.set("q2", 2, "v1")
        # Access q1 to make it recently used
        cache.get("q1", "v1")
        # Add q3 – should evict q2 (LRU)
        cache.set("q3", 3, "v1")
        assert cache.get("q1", "v1") == 1
        assert cache.get("q2", "v1") is None
        assert cache.get("q3", "v1") == 3

    def test_ttl_expiry(self):
        cache = SearchCache(ttl_seconds=0.01)
        cache.set("q", "val", "v1")
        time.sleep(0.05)
        assert cache.get("q", "v1") is None

    def test_hits_and_misses_tracked(self):
        cache = SearchCache()
        cache.set("q", "v", "v1")
        cache.get("q", "v1")  # hit
        cache.get("miss", "v1")  # miss
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    def test_clear_resets_cache(self):
        cache = SearchCache()
        cache.set("q", "v", "v1")
        cache.get("q", "v1")  # generates a hit
        cache.clear()
        # After clear, the cache should be empty and counters reset
        assert len(cache._cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_thread_safety(self):
        """Multiple threads can write/read without corruption."""
        import threading
        cache = SearchCache(maxsize=100)
        errors = []

        def worker(tid: int):
            try:
                for i in range(20):
                    cache.set(f"q{i}-{tid}", i, "v1")
                    cache.get(f"q{i}-{tid}", "v1")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# SemanticIndex (LSA path)
# ---------------------------------------------------------------------------

class TestSemanticIndex:
    def test_not_ready_before_build(self):
        idx = SemanticIndex()
        assert not idx.ready

    def test_ready_after_build_with_lsa(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        idx = SemanticIndex()
        # Force LSA by patching _try_build_st to return False
        with patch.object(idx, "_try_build_st", return_value=False):
            idx.build(SAMPLE_ARTICLES, vectorizer, matrix)
        assert idx.ready

    def test_query_returns_tuples(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        idx = SemanticIndex()
        with patch.object(idx, "_try_build_st", return_value=False):
            idx.build(SAMPLE_ARTICLES, vectorizer, matrix)
        results = idx.query("duo authentication", vectorizer=vectorizer, top_n=3)
        assert len(results) == 3
        for aid, score in results:
            assert isinstance(aid, str)
            assert isinstance(score, float)

    def test_query_sorted_descending(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        idx = SemanticIndex()
        with patch.object(idx, "_try_build_st", return_value=False):
            idx.build(SAMPLE_ARTICLES, vectorizer, matrix)
        results = idx.query("vpn network", vectorizer=vectorizer, top_n=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_returns_empty_when_not_ready(self):
        idx = SemanticIndex()
        results = idx.query("anything")
        assert results == []

    def test_build_without_vectorizer_leaves_unready(self):
        idx = SemanticIndex()
        with patch.object(idx, "_try_build_st", return_value=False):
            idx.build(SAMPLE_ARTICLES)  # no vectorizer, no matrix
        assert not idx.ready

    def test_sentence_transformers_fallback(self, vectorizer_and_matrix):
        """When sentence-transformers import fails, LSA fallback is used."""
        vectorizer, matrix = vectorizer_and_matrix
        idx = SemanticIndex()
        with patch.object(idx, "_try_build_st", return_value=False):
            idx.build(SAMPLE_ARTICLES, vectorizer, matrix)
        assert idx.ready
        assert not idx._use_st


# ---------------------------------------------------------------------------
# _lexical_retrieve helper
# ---------------------------------------------------------------------------

class TestLexicalRetrieve:
    def test_returns_article_ids(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        results = _lexical_retrieve("duo authentication", SAMPLE_ARTICLES, vectorizer, matrix)
        ids = [aid for aid, _ in results]
        assert "1001" in ids  # duo article should be near the top

    def test_zero_vector_returns_all_articles(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        # Query consisting only of stop-words produces zero vector
        results = _lexical_retrieve("the and or", SAMPLE_ARTICLES, vectorizer, matrix)
        assert len(results) == len(SAMPLE_ARTICLES)
        for _, score in results:
            assert score == 0.0

    def test_top_result_most_relevant(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        results = _lexical_retrieve("vpn cisco anyconnect", SAMPLE_ARTICLES, vectorizer, matrix)
        assert results[0][0] == "1002"  # VPN article should rank first


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_flags(self, **kwargs) -> FeatureFlags:
        flags = FeatureFlags.__new__(FeatureFlags)
        flags.hybrid_retrieval = kwargs.get("hybrid_retrieval", False)
        flags.query_normalization = kwargs.get("query_normalization", False)
        flags.adaptive_topk = kwargs.get("adaptive_topk", False)
        flags.search_cache = kwargs.get("search_cache", False)
        flags.any_enabled = lambda: any([
            flags.hybrid_retrieval,
            flags.query_normalization,
            flags.adaptive_topk,
            flags.search_cache,
        ])
        return flags

    def test_baseline_retrieve_returns_results(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags()
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)
        results, timings = retriever.retrieve(
            "duo mfa", SAMPLE_ARTICLES, vectorizer, matrix, top_k=5
        )
        assert len(results) > 0
        assert len(results) <= 5
        for aid, score in results:
            assert isinstance(aid, str)

    def test_normalization_flag_applied(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags(query_normalization=True)
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)
        results, timings = retriever.retrieve(
            "VPN CISCO", SAMPLE_ARTICLES, vectorizer, matrix, top_k=5
        )
        assert timings.normalization_ms >= 0
        # VPN article should appear in results
        ids = [aid for aid, _ in results]
        assert "1002" in ids

    def test_cache_flag_stores_and_retrieves(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags(search_cache=True)
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)

        # First call: cache miss
        results1, _ = retriever.retrieve(
            "wifi eduroam", SAMPLE_ARTICLES, vectorizer, matrix
        )
        stats_after_first = retriever.cache.stats.copy()

        # Second call: cache hit
        results2, _ = retriever.retrieve(
            "wifi eduroam", SAMPLE_ARTICLES, vectorizer, matrix
        )
        stats_after_second = retriever.cache.stats

        assert results1 == results2
        assert stats_after_second["hits"] > stats_after_first.get("hits", 0)

    def test_adaptive_topk_flag_limits_results(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags(adaptive_topk=True)
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)
        results, _ = retriever.retrieve(
            "duo vpn wifi netid email",
            SAMPLE_ARTICLES,
            vectorizer,
            matrix,
            top_k=5,
        )
        # Adaptive top-k must stay within bounds
        assert 1 <= len(results) <= 5

    def test_timings_populated(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags()
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)
        _, timings = retriever.retrieve(
            "password reset", SAMPLE_ARTICLES, vectorizer, matrix
        )
        assert isinstance(timings, SearchTimings)
        assert timings.total_ms > 0
        assert timings.lexical_ms > 0

    def test_semantic_unavailable_graceful(self, vectorizer_and_matrix):
        """Hybrid retrieval degrades gracefully when sentence-transformers is absent."""
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags(hybrid_retrieval=True)
        retriever = HybridRetriever(flags)

        # Patch _try_build_st to fail, forcing LSA
        if retriever._semantic is None:
            retriever._semantic = SemanticIndex()
        with patch.object(
            SemanticIndex, "_try_build_st", return_value=False
        ):
            retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)

        results, timings = retriever.retrieve(
            "duo authentication", SAMPLE_ARTICLES, vectorizer, matrix
        )
        assert len(results) > 0

    def test_no_cache_when_flag_off(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        flags = self._make_flags(search_cache=False)
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)
        assert retriever.cache is None


# ---------------------------------------------------------------------------
# Recall & nDCG helpers
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial_recall(self):
        result = recall_at_k(["a", "x", "y"], ["a", "b"], k=3)
        assert result == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0

    def test_k_limits_scope(self):
        # 'b' is relevant but outside top-1 window
        assert recall_at_k(["a", "b"], ["b"], k=1) == 0.0


class TestNdcgAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], [], k=2) == 0.0

    def test_higher_ranked_relevant_scores_better(self):
        # [a, b] vs [b, a] — 'a' is the relevant item
        score_a_first = ndcg_at_k(["a", "b"], ["a"], k=2)
        score_b_first = ndcg_at_k(["b", "a"], ["a"], k=2)
        assert score_a_first > score_b_first


# ---------------------------------------------------------------------------
# Integration: HybridRetriever + create_app (response shape)
# ---------------------------------------------------------------------------

class TestSearchEndpointWithRetriever:
    """Integration-style tests: verify /search returns expected shape
    when the HybridRetriever is wired into create_app."""

    def _make_retriever(self, vectorizer, matrix) -> HybridRetriever:
        flags = FeatureFlags.__new__(FeatureFlags)
        flags.hybrid_retrieval = False
        flags.query_normalization = True
        flags.adaptive_topk = False
        flags.search_cache = True
        flags.any_enabled = lambda: True
        retriever = HybridRetriever(flags)
        retriever.build(SAMPLE_ARTICLES, vectorizer, matrix)
        return retriever

    def test_search_returns_expected_fields(self, vectorizer_and_matrix):
        vectorizer, matrix = vectorizer_and_matrix
        retriever = self._make_retriever(vectorizer, matrix)
        client = MagicMock()
        app = create_app(client, SAMPLE_ARTICLES, vectorizer, matrix, retriever=retriever)

        with app.test_client() as tc:
            resp = tc.get("/search?q=duo+authentication")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "articles" in data
            assert "count" in data
            for article in data["articles"]:
                for field in ("article_id", "title", "badge_label", "badge_class", "in_ai"):
                    assert field in article, f"Missing field: {field}"

    def test_search_ordering_consistency(self, vectorizer_and_matrix):
        """The same query should return the same ordered results across calls."""
        vectorizer, matrix = vectorizer_and_matrix
        retriever = self._make_retriever(vectorizer, matrix)
        client = MagicMock()
        app = create_app(client, SAMPLE_ARTICLES, vectorizer, matrix, retriever=retriever)

        with app.test_client() as tc:
            resp1 = tc.get("/search?q=vpn+connection")
            resp2 = tc.get("/search?q=vpn+connection")
            ids1 = [a["article_id"] for a in resp1.get_json()["articles"]]
            ids2 = [a["article_id"] for a in resp2.get_json()["articles"]]
            assert ids1 == ids2
