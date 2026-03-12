"""
eval_harness.py – Relevance evaluation harness for IntelligentKB retrieval.

Measures:
  • Recall@3, Recall@5, Recall@10
  • nDCG@5

Usage (from repo root):
    python benchmarks/eval_harness.py [--enhanced]

Options:
    --enhanced      Evaluate the hybrid retrieval path (all flags on)

Relevance ground-truth is defined in GROUND_TRUTH below.  Each entry maps
a natural-language query to a list of article IDs that are considered
relevant for that query.  IDs that do not exist in the current article set
are silently ignored during evaluation.

Example output::

    Evaluating retrieval (baseline path)…
    Loaded 12 articles.

    query                                  R@3   R@5   R@10  nDCG@5
    ----------------------------------------------------------------
    duo two factor not working            1.000 1.000 1.000  1.000
    …
    ----------------------------------------------------------------
    MACRO-AVERAGE                         0.833 0.867 0.900  0.851
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import build_article_index, load_articles, ARTICLES_DIR  # noqa: E402
from search_enhancement import recall_at_k, ndcg_at_k  # noqa: E402

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
# Map from query string to a list of article_ids considered relevant.
# These are EXAMPLE entries – update with real article IDs from your corpus.
# The harness will skip any query whose relevant set has no overlap with the
# loaded articles (emitting a warning) so partial ground truth is safe.

GROUND_TRUTH: Dict[str, List[str]] = {
    "duo two factor not working": [],          # fill with real IDs
    "cannot connect to vpn": [],
    "forgot my netid password": [],
    "wifi eduroam keeps disconnecting": [],
    "mfa push notification not received": [],
    "outlook email not syncing": [],
    "office 365 installation error": [],
    "how do I reset my password": [],
    "vpn connection drops intermittently": [],
    "need to install duo mobile app": [],
}


def _retrieve_baseline(
    query: str, articles: list, vectorizer, doc_matrix, top_k: int = 10
) -> List[str]:
    from main import select_display_articles

    pairs = select_display_articles(query, articles, vectorizer, doc_matrix, display_k=top_k)
    return [a["article_id"] for a, _ in pairs]


def _retrieve_enhanced(
    query: str,
    articles: list,
    vectorizer,
    doc_matrix,
    retriever,
    top_k: int = 10,
) -> List[str]:
    pairs, _ = retriever.retrieve(query, articles, vectorizer, doc_matrix, top_k=top_k)
    return [aid for aid, _ in pairs]


def evaluate(
    queries: Dict[str, List[str]],
    retrieve_fn,
    article_ids: set,
) -> None:
    """Run evaluation and print a results table."""
    col_w = 42
    header = f"{'query'.ljust(col_w)}  R@3   R@5   R@10  nDCG@5"
    print(header)
    print("-" * len(header))

    totals = {"r3": 0.0, "r5": 0.0, "r10": 0.0, "ndcg5": 0.0}
    evaluated = 0

    for query, relevant_ids in queries.items():
        # Filter ground truth to known article IDs
        relevant = [aid for aid in relevant_ids if aid in article_ids]
        if not relevant:
            print(
                f"  [SKIP] {query!r}: no ground-truth articles found in corpus "
                f"(set relevant IDs in GROUND_TRUTH)"
            )
            continue

        retrieved = retrieve_fn(query)
        r3 = recall_at_k(retrieved, relevant, k=3)
        r5 = recall_at_k(retrieved, relevant, k=5)
        r10 = recall_at_k(retrieved, relevant, k=10)
        ndcg5 = ndcg_at_k(retrieved, relevant, k=5)

        totals["r3"] += r3
        totals["r5"] += r5
        totals["r10"] += r10
        totals["ndcg5"] += ndcg5
        evaluated += 1

        label = query[:col_w].ljust(col_w)
        print(f"{label}  {r3:.3f} {r5:.3f} {r10:.3f}  {ndcg5:.3f}")

    print("-" * len(header))
    if evaluated:
        print(
            f"{'MACRO-AVERAGE'.ljust(col_w)}  "
            f"{totals['r3']/evaluated:.3f} "
            f"{totals['r5']/evaluated:.3f} "
            f"{totals['r10']/evaluated:.3f}  "
            f"{totals['ndcg5']/evaluated:.3f}"
        )
        print(f"\nQueries evaluated: {evaluated} / {len(queries)}")
    else:
        print(
            "\nNo queries could be evaluated.\n"
            "Edit GROUND_TRUTH in benchmarks/eval_harness.py with article IDs "
            "from your corpus (run `python main.py` and visit /articles to see them)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="IntelligentKB relevance evaluation")
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Evaluate the hybrid retrieval path (all flags enabled)",
    )
    args = parser.parse_args()

    if not ARTICLES_DIR.is_dir():
        print(f"Error: articles directory not found at {ARTICLES_DIR}", file=sys.stderr)
        sys.exit(1)

    articles, _ = load_articles(ARTICLES_DIR)
    vectorizer, doc_matrix = build_article_index(articles)
    article_ids = {a["article_id"] for a in articles}

    path_label = "enhanced" if args.enhanced else "baseline"
    print(f"Evaluating retrieval ({path_label} path)…")
    print(f"Loaded {len(articles)} articles.\n")

    if args.enhanced:
        os.environ.update({
            "FEATURE_HYBRID_RETRIEVAL": "1",
            "FEATURE_QUERY_NORMALIZATION": "1",
            "FEATURE_ADAPTIVE_TOPK": "1",
            "FEATURE_SEARCH_CACHE": "1",
        })
        from search_enhancement import FeatureFlags, HybridRetriever

        flags = FeatureFlags()
        retriever = HybridRetriever(flags)
        retriever.build(articles, vectorizer, doc_matrix)

        def retrieve_fn(q: str) -> List[str]:
            return _retrieve_enhanced(q, articles, vectorizer, doc_matrix, retriever)
    else:
        def retrieve_fn(q: str) -> List[str]:  # type: ignore[misc]
            return _retrieve_baseline(q, articles, vectorizer, doc_matrix)

    evaluate(GROUND_TRUTH, retrieve_fn, article_ids)


if __name__ == "__main__":
    main()
