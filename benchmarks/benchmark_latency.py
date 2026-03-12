"""
benchmark_latency.py – Latency benchmark for IntelligentKB search retrieval.

Usage (from repo root):
    python benchmarks/benchmark_latency.py [--runs N] [--enhanced]

Options:
    --runs N        Number of timed repetitions per query (default: 50)
    --enhanced      Enable all feature flags for the enhanced path

Output:
    Per-query p50 / p95 latency and an overall summary table.

Example::

    $ python benchmarks/benchmark_latency.py --runs 100
    Running latency benchmark (100 runs/query, baseline path)...
    query                                  p50 (ms)  p95 (ms)
    ------------------------------------------------------------
    duo two factor not working               0.12      0.19
    cannot connect to vpn                    0.10      0.17
    ...
    OVERALL  p50=0.11 ms  p95=0.18 ms
    All p95 values within target (120 ms server-side retrieval).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_article_index, load_articles, ARTICLES_DIR  # noqa: E402

SAMPLE_QUERIES: List[str] = [
    "duo two factor not working",
    "cannot connect to vpn",
    "forgot my netid password",
    "wifi eduroam keeps disconnecting",
    "mfa push notification not received",
    "outlook email not syncing",
    "office 365 installation error",
    "how do I reset my password",
    "vpn connection drops intermittently",
    "need to install duo mobile app",
]

P95_TARGET_MS = 120.0  # server-side retrieval latency target


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * pct / 100) - 1)
    return sorted_vals[idx]


def run_baseline(
    queries: List[str],
    runs: int,
) -> dict[str, List[float]]:
    """Benchmark the original TF-IDF retrieval path."""
    from main import select_display_articles

    articles, _ = load_articles(ARTICLES_DIR)
    vectorizer, doc_matrix = build_article_index(articles)

    results: dict[str, List[float]] = {q: [] for q in queries}
    for q in queries:
        # Warm-up
        select_display_articles(q, articles, vectorizer, doc_matrix)
        for _ in range(runs):
            t0 = time.perf_counter()
            select_display_articles(q, articles, vectorizer, doc_matrix)
            results[q].append((time.perf_counter() - t0) * 1_000.0)
    return results


def run_enhanced(
    queries: List[str],
    runs: int,
) -> dict[str, List[float]]:
    """Benchmark the hybrid retrieval path with all flags enabled."""
    os.environ.update({
        "FEATURE_HYBRID_RETRIEVAL": "1",
        "FEATURE_QUERY_NORMALIZATION": "1",
        "FEATURE_ADAPTIVE_TOPK": "1",
        "FEATURE_SEARCH_CACHE": "1",
    })

    from search_enhancement import FeatureFlags, HybridRetriever

    articles, _ = load_articles(ARTICLES_DIR)
    vectorizer, doc_matrix = build_article_index(articles)

    flags = FeatureFlags()
    retriever = HybridRetriever(flags)
    retriever.build(articles, vectorizer, doc_matrix)

    results: dict[str, List[float]] = {q: [] for q in queries}
    for q in queries:
        # Warm-up (first call also populates cache)
        retriever.retrieve(q, articles, vectorizer, doc_matrix)
        for _ in range(runs):
            t0 = time.perf_counter()
            retriever.retrieve(q, articles, vectorizer, doc_matrix)
            results[q].append((time.perf_counter() - t0) * 1_000.0)
    return results


def print_table(results: dict[str, List[float]], label: str) -> None:
    col_w = 42
    print(f"\n{'query'.ljust(col_w)}  p50 (ms)  p95 (ms)")
    print("-" * (col_w + 22))
    all_times: List[float] = []
    for q, times in results.items():
        p50 = _percentile(times, 50)
        p95 = _percentile(times, 95)
        all_times.extend(times)
        print(f"{q[:col_w].ljust(col_w)}  {p50:>8.2f}  {p95:>8.2f}")
    print("-" * (col_w + 22))
    overall_p50 = _percentile(all_times, 50)
    overall_p95 = _percentile(all_times, 95)
    print(f"OVERALL  p50={overall_p50:.2f} ms  p95={overall_p95:.2f} ms")
    if overall_p95 <= P95_TARGET_MS:
        print(f"✓ All p95 values within target ({P95_TARGET_MS} ms server-side retrieval).")
    else:
        print(
            f"⚠ p95 ({overall_p95:.2f} ms) exceeds target ({P95_TARGET_MS} ms)."
            " Consider profiling slow queries."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="IntelligentKB search latency benchmark")
    parser.add_argument("--runs", type=int, default=50, help="Repetitions per query")
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Benchmark enhanced hybrid retrieval path",
    )
    args = parser.parse_args()

    path_label = "enhanced" if args.enhanced else "baseline"
    print(f"Running latency benchmark ({args.runs} runs/query, {path_label} path)…")

    if not ARTICLES_DIR.is_dir():
        print(f"Error: articles directory not found at {ARTICLES_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.enhanced:
        results = run_enhanced(SAMPLE_QUERIES, args.runs)
    else:
        results = run_baseline(SAMPLE_QUERIES, args.runs)

    print_table(results, path_label)


if __name__ == "__main__":
    main()
