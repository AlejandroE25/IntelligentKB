"""
search_enhancement.py – Phased search intelligence upgrades for IntelligentKB.

Feature flags (all default to False / current behaviour):
  FEATURE_HYBRID_RETRIEVAL      – TF-IDF + semantic RRF fusion
  FEATURE_QUERY_NORMALIZATION   – synonym expansion; typo fallback when lexical
                                   confidence is low
  FEATURE_ADAPTIVE_TOPK         – score-gap based top-k selection
  FEATURE_SEARCH_CACHE          – LRU cache for retrieval results

All features are independently toggle-able via environment variables.
Set FEATURE_<NAME>=1 (or true/yes) to enable.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------

def _flag(name: str, default: bool = False) -> bool:
    """Read a boolean feature flag from the environment."""
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.lower() in ("1", "true", "yes")


class FeatureFlags:
    """Reads feature flags once from the environment at construction time."""

    def __init__(self) -> None:
        self.hybrid_retrieval: bool = _flag("FEATURE_HYBRID_RETRIEVAL")
        self.query_normalization: bool = _flag("FEATURE_QUERY_NORMALIZATION")
        self.adaptive_topk: bool = _flag("FEATURE_ADAPTIVE_TOPK")
        self.search_cache: bool = _flag("FEATURE_SEARCH_CACHE")

    def reload(self) -> None:
        """Re-read all flags from the environment (useful in tests)."""
        self.__init__()

    def any_enabled(self) -> bool:
        return any([
            self.hybrid_retrieval,
            self.query_normalization,
            self.adaptive_topk,
            self.search_cache,
        ])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeatureFlags(hybrid={self.hybrid_retrieval}, "
            f"norm={self.query_normalization}, "
            f"adaptive_topk={self.adaptive_topk}, "
            f"cache={self.search_cache})"
        )


# Module-level default flags instance.  Import and use this in application
# code so that tests can monkeypatch it directly.
FLAGS = FeatureFlags()

# ---------------------------------------------------------------------------
# Domain Synonym Expansion
# ---------------------------------------------------------------------------

#: Mapping from a canonical help-desk term to a list of synonyms / related
#: terms that should be appended to the query to improve recall.
DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "duo": ["two-factor", "2fa", "mfa", "multi-factor", "authenticator"],
    "mfa": ["duo", "two-factor", "2fa", "multi-factor", "authenticator"],
    "2fa": ["duo", "mfa", "two-factor", "multi-factor"],
    "enroll": ["enrollment", "register", "setup", "activate"],
    "enrollment": ["enroll", "register", "setup", "activate"],
    "netid": ["username", "login", "account", "net id", "credentials"],
    "net id": ["netid", "username", "login", "account", "credentials"],
    "vpn": ["virtual private network", "anyconnect", "cisco", "remote access"],
    "wifi": ["wireless", "wi-fi", "network", "eduroam", "internet"],
    "wi-fi": ["wifi", "wireless", "network", "eduroam"],
    "eduroam": ["wifi", "wireless", "wi-fi", "network"],
    "password": ["passphrase", "credentials", "authentication", "login"],
    "reset": ["change", "recover", "forgot", "lost", "update"],
    "install": ["setup", "download", "configure", "set up"],
    "email": ["outlook", "mail", "microsoft 365", "o365", "exchange"],
    "office": ["microsoft 365", "o365", "word", "excel", "powerpoint"],
}

# ---------------------------------------------------------------------------
# Query Normalizer
# ---------------------------------------------------------------------------

class QueryNormalizer:
    """Normalises raw query text through configurable pipeline stages.

    Basic normalisation (always applied):
        - Lowercase
        - Punctuation / whitespace cleanup (preserves apostrophes & hyphens)

    Extended normalisation (controlled by caller):
        - Domain synonym expansion
        - Typo correction via ``difflib.get_close_matches`` when lexical
          confidence is low or the caller requests it explicitly
    """

    _PUNCT_RE = re.compile(r"[^\w\s'\-]")  # keep apostrophes and hyphens
    _WS_RE = re.compile(r"\s+")

    def __init__(self, synonyms: Dict[str, List[str]] = DOMAIN_SYNONYMS) -> None:
        self._synonyms = synonyms
        # Build a flat vocabulary for typo detection (all keys + all expanded terms)
        self._vocab: set = set(synonyms.keys())
        for terms in synonyms.values():
            for t in terms:
                self._vocab.update(t.split())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def basic_normalize(self, query: str) -> str:
        """Lowercase, strip punctuation, and collapse whitespace."""
        q = query.lower()
        q = self._PUNCT_RE.sub(" ", q)
        q = self._WS_RE.sub(" ", q).strip()
        return q

    def expand_synonyms(self, query: str) -> str:
        """Append synonym terms to *query* (which should already be lowercased)."""
        normalized = self.basic_normalize(query)
        extra: List[str] = []

        tokens = normalized.split()

        # Single-word matches
        for token in tokens:
            if token in self._synonyms:
                extra.extend(self._synonyms[token])

        # Two-word phrase matches
        for i in range(len(tokens) - 1):
            phrase = tokens[i] + " " + tokens[i + 1]
            if phrase in self._synonyms:
                extra.extend(self._synonyms[phrase])

        if extra:
            return normalized + " " + " ".join(extra)
        return normalized

    def typo_correct(self, query: str, cutoff: float = 0.82) -> str:
        """Best-effort per-token typo correction using stdlib ``difflib``."""
        import difflib

        tokens = self.basic_normalize(query).split()
        corrected: List[str] = []
        for token in tokens:
            if token in self._vocab or len(token) <= 2:
                corrected.append(token)
            else:
                matches = difflib.get_close_matches(
                    token, self._vocab, n=1, cutoff=cutoff
                )
                corrected.append(matches[0] if matches else token)
        return " ".join(corrected)

    def normalize(
        self,
        query: str,
        use_synonyms: bool = True,
        use_typo: bool = False,
    ) -> str:
        """Full normalisation pipeline.

        Args:
            query: raw query string
            use_synonyms: append synonym expansions
            use_typo: apply typo correction before synonym expansion
        """
        if use_typo:
            q = self.typo_correct(query)
        else:
            q = self.basic_normalize(query)

        if use_synonyms:
            q = self.expand_synonyms(q)

        return q


# ---------------------------------------------------------------------------
# Timing / Observability
# ---------------------------------------------------------------------------

@dataclass
class SearchTimings:
    """Per-phase latency measurements for a single retrieval call (in ms)."""

    normalization_ms: float = 0.0
    lexical_ms: float = 0.0
    semantic_ms: float = 0.0
    fusion_ms: float = 0.0
    cache_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "normalization_ms": round(self.normalization_ms, 2),
            "lexical_ms": round(self.lexical_ms, 2),
            "semantic_ms": round(self.semantic_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "cache_ms": round(self.cache_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }

    def confidence(self, results: List[Tuple[str, float]]) -> str:
        """Return a simple confidence indicator based on top score."""
        if not results:
            return "none"
        top_score = results[0][1]
        if top_score >= 0.20:
            return "high"
        if top_score >= 0.08:
            return "medium"
        return "low"


@contextmanager
def _timer(dest: list) -> Generator:
    """Context manager that stores elapsed milliseconds in ``dest[0]``."""
    start = time.perf_counter()
    try:
        yield dest
    finally:
        dest[0] = (time.perf_counter() - start) * 1_000.0


def log_retrieval(
    original: str,
    normalized: str,
    timings: SearchTimings,
    n_results: int,
    confidence: str = "",
) -> None:
    """Emit a single structured log line for each retrieval request."""
    logger.info(
        "retrieval query=%r normalized=%r n=%d confidence=%s total=%.1fms "
        "(norm=%.1f lex=%.1f sem=%.1f fuse=%.1f cache=%.1f)",
        original,
        normalized,
        n_results,
        confidence,
        timings.total_ms,
        timings.normalization_ms,
        timings.lexical_ms,
        timings.semantic_ms,
        timings.fusion_ms,
        timings.cache_ms,
    )


# ---------------------------------------------------------------------------
# Semantic Index (LSA primary, sentence-transformers upgrade)
# ---------------------------------------------------------------------------

class SemanticIndex:
    """In-memory semantic embedding index for KB articles.

    Build strategy (at startup):
        1. Try to load a ``sentence-transformers`` model (requires the optional
           ``sentence-transformers`` package to be installed).
        2. If unavailable, fall back to LSA via ``TruncatedSVD`` from
           scikit-learn, which is already a hard dependency.

    Query strategy:
        - Encode the query with the same model used for the corpus.
        - Compute cosine similarity (all articles fit in RAM).
        - Return (article_id, similarity) pairs sorted desc.
    """

    _ST_MODEL_NAME = "all-MiniLM-L6-v2"
    _LSA_COMPONENTS = 50
    _MAX_ARTICLE_CHARS = 1_024  # chars fed to sentence-transformers per article

    def __init__(self) -> None:
        self._st_model: Any = None          # SentenceTransformer instance or None
        self._use_st: bool = False
        self._svd: Optional[TruncatedSVD] = None
        self._embeddings: Optional[np.ndarray] = None
        self._article_ids: List[str] = []
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        articles: List[Dict[str, Any]],
        vectorizer: Any = None,
        tfidf_matrix: Any = None,
    ) -> None:
        """Build the embedding index.

        Pass *vectorizer* and *tfidf_matrix* so the LSA fallback can reuse the
        already-computed TF-IDF representation.
        """
        self._article_ids = [a["article_id"] for a in articles]

        # Try sentence-transformers first
        if self._try_build_st(articles):
            logger.info(
                "SemanticIndex: built with sentence-transformers (%s)", self._ST_MODEL_NAME
            )
            self._ready = True
            return

        # LSA fallback
        if vectorizer is not None and tfidf_matrix is not None:
            self._build_lsa(tfidf_matrix)
            logger.info(
                "SemanticIndex: built with LSA (n_components=%d)", self._LSA_COMPONENTS
            )
            self._ready = True
        else:
            logger.warning(
                "SemanticIndex: no vectorizer/matrix for LSA fallback; "
                "semantic branch disabled"
            )

    def _try_build_st(self, articles: List[Dict[str, Any]]) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(self._ST_MODEL_NAME)
            texts = [
                (a.get("title", "") + " " + a.get("content", ""))[
                    : self._MAX_ARTICLE_CHARS
                ]
                for a in articles
            ]
            self._embeddings = model.encode(texts, normalize_embeddings=True)
            self._st_model = model
            self._use_st = True
            return True
        except ImportError:
            logger.debug(
                "sentence-transformers not installed; falling back to LSA "
                "(install with: pip install sentence-transformers>=2.2.0)"
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed to load sentence-transformers model %r: %s; falling back to LSA",
                self._ST_MODEL_NAME,
                exc,
            )
            return False

    def _build_lsa(self, tfidf_matrix: Any) -> None:
        n_docs = tfidf_matrix.shape[0]
        n_terms = tfidf_matrix.shape[1]
        n_components = min(self._LSA_COMPONENTS, n_terms - 1, n_docs - 1)
        if n_components < 2:
            logger.warning(
                "SemanticIndex: too few features for LSA (%d); semantic branch disabled",
                n_components,
            )
            return
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        raw = self._svd.fit_transform(tfidf_matrix)
        self._embeddings = sk_normalize(raw, norm="l2")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        vectorizer: Any = None,
        top_n: int = 20,
    ) -> List[Tuple[str, float]]:
        """Return ``(article_id, similarity)`` pairs sorted descending.

        Returns an empty list if the index is not ready.
        """
        if not self._ready or self._embeddings is None:
            return []

        try:
            q_emb = self._encode_query(query_text, vectorizer)
            if q_emb is None:
                return []
            sims: np.ndarray = self._embeddings @ q_emb
            top_idx = np.argsort(sims)[::-1][:top_n]
            return [(self._article_ids[i], float(sims[i])) for i in top_idx]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticIndex.query failed for query %r: %s", query_text, exc
            )
            return []

    def _encode_query(
        self, query_text: str, vectorizer: Any
    ) -> Optional[np.ndarray]:
        if self._use_st and self._st_model is not None:
            emb = self._st_model.encode([query_text], normalize_embeddings=True)
            return emb[0]
        if self._svd is not None and vectorizer is not None:
            vec = vectorizer.transform([query_text])
            raw = self._svd.transform(vec)
            normed: np.ndarray = sk_normalize(raw, norm="l2")
            return normed[0]
        return None

    @property
    def ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fusion(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    k: int = 60,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    Both lists are ``(article_id, score)`` pairs sorted by score descending.
    Returns a new list sorted by fused RRF score descending.

    Args:
        k: RRF smoothing constant (standard default: 60).
        alpha: weight for the lexical branch; ``1 - alpha`` goes to semantic.
    """
    scores: Dict[str, float] = {}

    # Ignore duplicate IDs within each branch so repeated source articles do
    # not receive disproportionate weight.
    seen_lexical: set[str] = set()
    lexical_rank = 0
    for aid, _ in lexical_results:
        if aid in seen_lexical:
            continue
        seen_lexical.add(aid)
        scores[aid] = scores.get(aid, 0.0) + alpha / (k + lexical_rank + 1)
        lexical_rank += 1

    seen_semantic: set[str] = set()
    semantic_rank = 0
    for aid, _ in semantic_results:
        if aid in seen_semantic:
            continue
        seen_semantic.add(aid)
        scores[aid] = scores.get(aid, 0.0) + (1.0 - alpha) / (k + semantic_rank + 1)
        semantic_rank += 1

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Adaptive top-k
# ---------------------------------------------------------------------------

def adaptive_top_k(
    scores: List[float],
    min_k: int = 3,
    max_k: int = 10,
    gap_threshold: float = 0.05,
) -> int:
    """Choose top-k based on score gaps.

    Scans positions ``min_k … max_k`` and stops at the first significant
    consecutive score drop (≥ *gap_threshold*).  Returns *min_k* when
    there are not enough scores.

    Args:
        scores: descending similarity scores (same length as candidates).
        min_k: minimum articles to return (must be ≥ 1).
        max_k: maximum articles to return.
        gap_threshold: absolute score drop that triggers an early stop.

    Returns:
        Chosen k in ``[min(min_k, len(scores)), min(max_k, len(scores))]``.
    """
    n = len(scores)
    if n <= min_k:
        return n

    effective_max = min(max_k, n)

    for i in range(min_k, effective_max):
        if scores[i - 1] - scores[i] >= gap_threshold:
            return i

    return effective_max


# ---------------------------------------------------------------------------
# LRU Search Cache
# ---------------------------------------------------------------------------

class SearchCache:
    """Thread-safe LRU cache for retrieval results.

    Cache keys are ``(normalised_query, config_version)`` so that changing the
    retrieval configuration automatically invalidates all cached entries.
    An optional TTL (seconds) further bounds staleness.
    """

    def __init__(
        self,
        maxsize: int = 256,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------

    def _key(self, query: str, config_version: str) -> str:
        return f"{config_version}::{query}"

    def get(self, query: str, config_version: str = "v1") -> Optional[Any]:
        """Return cached value or ``None`` on miss / expiry."""
        key = self._key(query, config_version)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            entry_time, value = self._cache[key]
            if self._ttl is not None and (time.time() - entry_time) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, query: str, value: Any, config_version: str = "v1") -> None:
        """Store *value* under *query*; evicts oldest entry if at capacity."""
        key = self._key(query, config_version)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Orchestrates query normalisation, lexical retrieval, semantic retrieval,
    RRF fusion, adaptive top-k, and caching.

    Behaviour is controlled entirely by :class:`FeatureFlags`; when all flags
    are ``False`` the retrieval path reduces to plain TF-IDF cosine similarity
    (equivalent to the original ``select_display_articles`` implementation).

    Usage::

        retriever = HybridRetriever(flags)
        retriever.build(articles, vectorizer, doc_matrix)  # once at startup

        results, timings = retriever.retrieve(query, articles, vectorizer, doc_matrix)
    """

    #: Bump this when retrieval configuration changes to invalidate stale cache entries.
    CONFIG_VERSION: str = "hybrid_v1"

    def __init__(self, flags: FeatureFlags = FLAGS) -> None:
        self._flags = flags
        self._normalizer = QueryNormalizer()
        self._semantic: Optional[SemanticIndex] = None
        self._cache: Optional[SearchCache] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def build(
        self,
        articles: List[Dict[str, Any]],
        vectorizer: Any,
        tfidf_matrix: Any,
    ) -> None:
        """Build the semantic index and cache at application startup."""
        if self._flags.hybrid_retrieval:
            self._semantic = SemanticIndex()
            self._semantic.build(articles, vectorizer, tfidf_matrix)

        if self._flags.search_cache:
            self._cache = SearchCache(maxsize=256, ttl_seconds=300)

        logger.info("HybridRetriever ready (flags=%s)", self._flags)

    def _should_expand_synonyms(self, normalized_query: str) -> bool:
        """Return True when synonym expansion is likely to help recall.

        Expansion is intentionally conservative to avoid intent drift on longer,
        more specific natural-language queries.
        """
        tokens = normalized_query.split()
        if not tokens:
            return False
        return len(tokens) <= 4

    def _hybrid_display_score(
        self,
        article_id: str,
        lexical_scores: Dict[str, float],
        semantic_scores: Dict[str, float],
    ) -> float:
        """Map fused rankings to a TF-IDF-like score scale for UI confidence.

        RRF scores are rank-only and very small (around 0.01-0.03), which makes
        static TF-IDF thresholds look artificially "low" in the UI. Prefer the
        lexical score when available; otherwise map semantic cosine to a modest
        TF-IDF-like range.
        """
        lex = lexical_scores.get(article_id, 0.0)
        if lex > 0.0:
            return float(lex)

        sem = max(semantic_scores.get(article_id, 0.0), 0.0)
        return float(min(0.25, sem * 0.25))

    @staticmethod
    def _token_root(token: str) -> str:
        """Apply light stemming for overlap checks (e.g., enroll/enrollment)."""
        for suffix in ("ments", "ment", "ingly", "edly", "ing", "ed", "es", "s"):
            if token.endswith(suffix) and (len(token) - len(suffix)) >= 4:
                return token[: -len(suffix)]
        return token

    def _keyword_roots(self, text: str) -> set[str]:
        stop = {
            "how", "do", "i", "in", "to", "the", "a", "an", "for", "and",
            "or", "of", "on", "with", "my", "is", "are", "can", "you", "me",
            "help", "please", "issue", "problem", "not", "from", "at", "it",
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        roots: set[str] = set()
        for token in tokens:
            if len(token) < 3 or token in stop:
                continue
            roots.add(self._token_root(token))
        return roots

    def _has_keyword_overlap(self, query: str, article: Dict[str, Any]) -> bool:
        """Return True when query terms significantly overlap article metadata."""
        query_roots = self._keyword_roots(query)
        if not query_roots:
            return False

        article_text = (
            f"{article.get('title', '')} "
            f"{article.get('keywords', '')} "
            f"{article.get('content', '')[:1200]}"
        )
        article_roots = self._keyword_roots(article_text)
        overlap = query_roots & article_roots

        if len(overlap) >= 2:
            return True
        if len(overlap) == 1 and len(query_roots) <= 2:
            return True
        return False

    def _confidence_with_overlap(
        self,
        query: str,
        results: List[Tuple[str, float]],
        articles_by_id: Dict[str, Dict[str, Any]],
    ) -> str:
        """Promote near-exact top hits to high confidence only with term overlap."""
        base_confidence = SearchTimings().confidence(results)
        if base_confidence != "medium" or len(results) < 2:
            return base_confidence

        top_score = results[0][1]
        second_score = results[1][1]
        if top_score < 0.16 or (top_score - second_score) < 0.03:
            return base_confidence

        top_article = articles_by_id.get(results[0][0])
        if top_article and self._has_keyword_overlap(query, top_article):
            return "high"

        return base_confidence

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        articles: List[Dict[str, Any]],
        vectorizer: Any,
        tfidf_matrix: Any,
        top_k: int = 10,
    ) -> Tuple[List[Tuple[str, float]], SearchTimings]:
        """Retrieve *top_k* articles for *query*.

        Returns:
            A tuple ``(results, timings)`` where *results* is a list of
            ``(article_id, score)`` pairs sorted descending and *timings*
            contains per-phase latency measurements.
        """
        timings = SearchTimings()
        t_total_start = time.perf_counter()
        articles_by_id = {a.get("article_id", ""): a for a in articles if a.get("article_id")}

        # ── 1. Query normalisation ─────────────────────────────────────
        elapsed: list = [0.0]
        with _timer(elapsed):
            if self._flags.query_normalization:
                base_query = self._normalizer.basic_normalize(query)
                if self._should_expand_synonyms(base_query):
                    norm_query = self._normalizer.expand_synonyms(base_query)
                else:
                    norm_query = base_query
            else:
                norm_query = query
        timings.normalization_ms = elapsed[0]

        # ── 2. Cache lookup ────────────────────────────────────────────
        if self._cache is not None:
            with _timer(elapsed):
                cached = self._cache.get(norm_query, self.CONFIG_VERSION)
            timings.cache_ms = elapsed[0]
            if cached is not None:
                timings.total_ms = (time.perf_counter() - t_total_start) * 1_000.0
                logger.debug(
                    "cache hit for %r (%.1f ms total)", query, timings.total_ms
                )
                return cached, timings

        # ── 3. Lexical (TF-IDF) retrieval ─────────────────────────────
        with _timer(elapsed):
            lexical = _lexical_retrieve(
                norm_query, articles, vectorizer, tfidf_matrix, top_n=top_k * 2
            )
        timings.lexical_ms = elapsed[0]

        # ── 4. Semantic retrieval ──────────────────────────────────────
        semantic: List[Tuple[str, float]] = []
        if (
            self._flags.hybrid_retrieval
            and self._semantic is not None
            and self._semantic.ready
        ):
            with _timer(elapsed):
                semantic = self._semantic.query(
                    norm_query, vectorizer=vectorizer, top_n=top_k * 2
                )
            timings.semantic_ms = elapsed[0]

        # ── 5. Fusion ──────────────────────────────────────────────────
        with _timer(elapsed):
            if self._flags.hybrid_retrieval and semantic:
                fused_ranked = rrf_fusion(lexical, semantic)[:top_k]
                lexical_map = dict(lexical)
                semantic_map = dict(semantic)
                fused = [
                    (
                        aid,
                        self._hybrid_display_score(aid, lexical_map, semantic_map),
                    )
                    for aid, _ in fused_ranked
                ]
            else:
                fused = lexical[:top_k]
        timings.fusion_ms = elapsed[0]

        # ── 6. Adaptive top-k ──────────────────────────────────────────
        if self._flags.adaptive_topk and fused:
            score_list = [s for _, s in fused]
            k = adaptive_top_k(
                score_list, min_k=min(3, top_k), max_k=top_k
            )
            fused = fused[:k]

        # ── 7. Cache store ─────────────────────────────────────────────
        if self._cache is not None:
            self._cache.set(norm_query, fused, self.CONFIG_VERSION)

        timings.total_ms = (time.perf_counter() - t_total_start) * 1_000.0
        confidence = self._confidence_with_overlap(query, fused, articles_by_id)
        log_retrieval(query, norm_query, timings, len(fused), confidence)

        # ── 8. Typo-corrected retry when lexical confidence is low ─────
        if (
            self._flags.query_normalization
            and confidence == "low"
            and not semantic  # only in purely lexical mode
        ):
            typo_query = self._normalizer.normalize(
                query, use_synonyms=False, use_typo=True
            )
            if self._should_expand_synonyms(typo_query):
                typo_query = self._normalizer.expand_synonyms(typo_query)
            if typo_query != norm_query:
                logger.debug(
                    "Low confidence; retrying with typo correction: %r -> %r",
                    norm_query,
                    typo_query,
                )
                with _timer(elapsed):
                    lexical_retry = _lexical_retrieve(
                        typo_query, articles, vectorizer, tfidf_matrix, top_n=top_k * 2
                    )
                timings.lexical_ms += elapsed[0]
                if lexical_retry and lexical_retry[0][1] > (fused[0][1] if fused else 0):
                    fused = lexical_retry[:top_k]
                    timings.total_ms = (time.perf_counter() - t_total_start) * 1_000.0
                    log_retrieval(
                        query,
                        typo_query,
                        timings,
                        len(fused),
                        self._confidence_with_overlap(query, fused, articles_by_id),
                    )

        return fused, timings

    @property
    def cache(self) -> Optional[SearchCache]:
        return self._cache

    @property
    def semantic_index(self) -> Optional[SemanticIndex]:
        return self._semantic

    @property
    def flags(self) -> FeatureFlags:
        return self._flags


# ---------------------------------------------------------------------------
# Lexical retrieval helper (TF-IDF cosine similarity)
# ---------------------------------------------------------------------------

def _lexical_retrieve(
    query: str,
    articles: List[Dict[str, Any]],
    vectorizer: Any,
    tfidf_matrix: Any,
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """Run TF-IDF cosine similarity and return ``(article_id, score)`` pairs.

    Returns all articles with score 0.0 when the query produces an all-zero
    vector (all stop-words), mirroring the original ``select_relevant_articles``
    fallback.
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

    try:
        q_vec = vectorizer.transform([query])
    except Exception as exc:  # noqa: BLE001
        logger.warning("_lexical_retrieve: vectorizer.transform failed for query %r: %s", query, exc)
        return [(a["article_id"], 0.0) for a in articles]

    if q_vec.nnz == 0:
        # All stop-words: return all articles with score 0
        return [(a["article_id"], 0.0) for a in articles]

    sims = _cosine_sim(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(articles[i]["article_id"], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Evaluation helpers (used by eval_harness.py)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@k: fraction of relevant articles found in top-k retrieved."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & set(relevant)) / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """nDCG@k: normalised discounted cumulative gain."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)

    def _dcg(items: List[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(items[:k])
            if item in relevant_set
        )

    ideal_items = list(relevant_set)[:k]
    idcg = _dcg(ideal_items)
    if idcg == 0.0:
        return 0.0
    return _dcg(retrieved) / idcg
