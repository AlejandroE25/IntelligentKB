"""Tests for feedback_store.py."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from feedback_store import FeedbackStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback.json")


# ---------------------------------------------------------------------------
# Article flags
# ---------------------------------------------------------------------------

class TestAddArticleFlag:
    def test_creates_record_on_first_flag(self, tmp_path):
        store = make_store(tmp_path)
        record = store.add_article_flag("12345", "Steps are wrong")
        assert record["flag_count"] == 1
        assert len(record["flags"]) == 1
        assert record["flags"][0]["reason"] == "Steps are wrong"

    def test_multiple_flags_accumulate(self, tmp_path):
        store = make_store(tmp_path)
        store.add_article_flag("12345", "Wrong")
        store.add_article_flag("12345", "Outdated")
        record = store.add_article_flag("12345", "Confusing")
        assert record["flag_count"] == 3
        assert len(record["flags"]) == 3

    def test_flag_stores_session_id(self, tmp_path):
        store = make_store(tmp_path)
        record = store.add_article_flag("12345", "Bad info", session_id="sess_abc")
        assert record["flags"][0]["session_id"] == "sess_abc"

    def test_flag_stores_timestamp(self, tmp_path):
        store = make_store(tmp_path)
        record = store.add_article_flag("12345", "Bad info")
        assert record["flags"][0]["timestamp"]  # non-empty ISO string

    def test_independent_articles_tracked_separately(self, tmp_path):
        store = make_store(tmp_path)
        store.add_article_flag("111", "bad")
        store.add_article_flag("222", "also bad")
        counts = store.get_flag_counts()
        assert counts["111"] == 1
        assert counts["222"] == 1

    def test_persists_across_store_instances(self, tmp_path):
        path = tmp_path / "feedback.json"
        s1 = FeedbackStore(path)
        s1.add_article_flag("12345", "Wrong")

        s2 = FeedbackStore(path)
        counts = s2.get_flag_counts()
        assert counts["12345"] == 1


class TestGetFlagCounts:
    def test_empty_before_any_flags(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_flag_counts() == {}

    def test_returns_correct_counts(self, tmp_path):
        store = make_store(tmp_path)
        store.add_article_flag("aaa", "bad")
        store.add_article_flag("aaa", "also bad")
        store.add_article_flag("bbb", "bad")
        counts = store.get_flag_counts()
        assert counts == {"aaa": 2, "bbb": 1}


class TestGetArticleFlags:
    def test_returns_empty_for_unknown_article(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_article_flags("unknown") == {}

    def test_returns_full_record(self, tmp_path):
        store = make_store(tmp_path)
        store.add_article_flag("12345", "Wrong steps", session_id="s1")
        record = store.get_article_flags("12345")
        assert record["flag_count"] == 1
        assert record["flags"][0]["reason"] == "Wrong steps"


# ---------------------------------------------------------------------------
# Session feedback
# ---------------------------------------------------------------------------

class TestSessionFeedback:
    def test_yes_outcome_recorded(self, tmp_path):
        store = make_store(tmp_path)
        store.add_session_feedback("sess1", "wifi not connecting", "yes")
        summary = store.get_summary()
        assert summary["session_feedback_counts"]["yes"] == 1

    def test_no_outcome_recorded(self, tmp_path):
        store = make_store(tmp_path)
        store.add_session_feedback("sess1", "wifi not connecting", "no")
        summary = store.get_summary()
        assert summary["session_feedback_counts"]["no"] == 1

    def test_partial_outcome_recorded(self, tmp_path):
        store = make_store(tmp_path)
        store.add_session_feedback("sess1", "password reset", "partial")
        summary = store.get_summary()
        assert summary["session_feedback_counts"]["partial"] == 1

    def test_multiple_feedback_accumulated(self, tmp_path):
        store = make_store(tmp_path)
        store.add_session_feedback("s1", "q1", "yes")
        store.add_session_feedback("s2", "q2", "yes")
        store.add_session_feedback("s3", "q3", "no")
        summary = store.get_summary()
        assert summary["session_feedback_counts"] == {"yes": 2, "no": 1, "partial": 0}


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_empty_summary(self, tmp_path):
        store = make_store(tmp_path)
        s = store.get_summary()
        assert s["total_flags"] == 0
        assert s["flagged_articles"] == 0
        assert s["session_feedback_counts"] == {"yes": 0, "no": 0, "partial": 0}
        assert s["most_flagged"] == []

    def test_total_flags_counts_all(self, tmp_path):
        store = make_store(tmp_path)
        store.add_article_flag("aaa", "bad")
        store.add_article_flag("aaa", "bad again")
        store.add_article_flag("bbb", "bad")
        s = store.get_summary()
        assert s["total_flags"] == 3
        assert s["flagged_articles"] == 2

    def test_most_flagged_sorted_descending(self, tmp_path):
        store = make_store(tmp_path)
        store.add_article_flag("aaa", "1")
        store.add_article_flag("bbb", "1")
        store.add_article_flag("bbb", "2")
        store.add_article_flag("bbb", "3")
        s = store.get_summary()
        assert s["most_flagged"][0] == ("bbb", 3)
        assert s["most_flagged"][1] == ("aaa", 1)

    def test_most_flagged_capped_at_5(self, tmp_path):
        store = make_store(tmp_path)
        for i in range(10):
            store.add_article_flag(str(i), "bad")
        s = store.get_summary()
        assert len(s["most_flagged"]) == 5


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_handles_missing_file_gracefully(self, tmp_path):
        store = FeedbackStore(tmp_path / "nonexistent" / "feedback.json")
        # Should not raise; returns empty
        counts = store.get_flag_counts()
        assert counts == {}

    def test_handles_corrupt_json_gracefully(self, tmp_path):
        path = tmp_path / "feedback.json"
        path.write_text("this is not json", encoding="utf-8")
        store = FeedbackStore(path)
        counts = store.get_flag_counts()
        assert counts == {}

    def test_atomic_write_does_not_corrupt_on_concurrent_flags(self, tmp_path):
        store = make_store(tmp_path)
        errors = []

        def _flag(i):
            try:
                store.add_article_flag("shared", f"reason {i}", session_id=f"s{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_flag, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        record = store.get_article_flags("shared")
        assert record["flag_count"] == 20
        assert len(record["flags"]) == 20
