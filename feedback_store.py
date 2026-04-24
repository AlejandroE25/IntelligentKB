"""
Thread-safe JSON persistence for article flags and session feedback.

Uses file-level locking (fcntl.flock) for cross-process safety in multi-worker
gunicorn deployments, and threading.Lock for within-process concurrency.

Atomic writes use write-to-tmp + os.replace() to prevent corruption if the
process is killed mid-write.

Optionally syncs to Azure Blob Storage after each write when a blob_service
and data_container_name are provided at construction time.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FCNTL_AVAILABLE = False
try:
    import fcntl
    _FCNTL_AVAILABLE = True
except ImportError:
    pass  # Windows — threading.Lock is sufficient for single-worker dev

BLOB_FEEDBACK_NAME = "feedback.json"

_EMPTY_STORE: dict[str, Any] = {
    "article_flags": {},
    "session_feedback": [],
}


class FeedbackStore:
    """Persistent store for article flags and session outcome feedback."""

    def __init__(
        self,
        feedback_path: Path,
        blob_service=None,
        data_container_name: str = "",
    ) -> None:
        self._path = feedback_path
        self._lock = threading.Lock()
        self._blob_service = blob_service
        self._container = data_container_name

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return copy.deepcopy(_EMPTY_STORE)
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                if _FCNTL_AVAILABLE:
                    fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                if _FCNTL_AVAILABLE:
                    fcntl.flock(f, fcntl.LOCK_UN)
            if "article_flags" not in data:
                data["article_flags"] = {}
            if "session_feedback" not in data:
                data["session_feedback"] = []
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read feedback store at %s: %s", self._path, exc)
            return copy.deepcopy(_EMPTY_STORE)

    def _save(self, data: dict[str, Any]) -> None:
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                if _FCNTL_AVAILABLE:
                    fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(data, f, indent=2, default=str)
                if _FCNTL_AVAILABLE:
                    fcntl.flock(f, fcntl.LOCK_UN)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            logger.warning("Could not write feedback store to %s: %s", self._path, exc)
            return

        if self._blob_service and self._container:
            try:
                import blob_store as bs
                bs.upload_blob_json(self._container, BLOB_FEEDBACK_NAME, data, self._blob_service)
            except Exception as exc:
                logger.warning("Blob sync of feedback store failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_article_flag(
        self,
        article_id: str,
        reason: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Append a flag for an article. Returns the updated flag record."""
        with self._lock:
            data = self._load()
            flags_map: dict = data["article_flags"]
            record = flags_map.setdefault(article_id, {"flag_count": 0, "flags": []})
            record["flag_count"] += 1
            record["flags"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": reason.strip(),
                    "session_id": session_id,
                }
            )
            self._save(data)
            return dict(record)

    def get_flag_counts(self) -> dict[str, int]:
        """Return {article_id: flag_count} for all flagged articles."""
        data = self._load()
        return {aid: rec["flag_count"] for aid, rec in data["article_flags"].items()}

    def get_article_flags(self, article_id: str) -> dict[str, Any]:
        """Return the full flag record for an article, or empty dict if none."""
        data = self._load()
        return data["article_flags"].get(article_id, {})

    def add_session_feedback(
        self,
        session_id: str,
        query: str,
        outcome: str,
    ) -> None:
        """Record a session outcome. outcome must be 'yes', 'no', or 'partial'."""
        with self._lock:
            data = self._load()
            data["session_feedback"].append(
                {
                    "session_id": session_id,
                    "query": query.strip(),
                    "outcome": outcome,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._save(data)

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate stats for the admin dashboard."""
        data = self._load()
        flags_map = data["article_flags"]
        feedback_list = data["session_feedback"]

        outcome_counts: dict[str, int] = {"yes": 0, "no": 0, "partial": 0}
        for entry in feedback_list:
            outcome = entry.get("outcome", "")
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1

        sorted_flags = sorted(
            flags_map.items(),
            key=lambda kv: kv[1].get("flag_count", 0),
            reverse=True,
        )

        return {
            "total_flags": sum(rec.get("flag_count", 0) for rec in flags_map.values()),
            "flagged_articles": len(flags_map),
            "session_feedback_counts": outcome_counts,
            "most_flagged": [(aid, rec.get("flag_count", 0)) for aid, rec in sorted_flags[:5]],
        }

    def load_from_blob(self, blob_service, data_container_name: str) -> None:
        """Seed local file from blob storage at startup (call before first read)."""
        try:
            import blob_store as bs
            blob_data = bs.download_blob_json(data_container_name, BLOB_FEEDBACK_NAME, blob_service)
            if blob_data is not None:
                with self._lock:
                    tmp_path = self._path.with_suffix(".json.tmp")
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(blob_data, f, indent=2)
                    os.replace(tmp_path, self._path)
                    logger.info("Loaded feedback store from blob storage")
        except Exception as exc:
            logger.warning("Could not load feedback store from blob: %s", exc)
