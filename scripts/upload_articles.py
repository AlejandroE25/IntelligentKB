#!/usr/bin/env python3
"""
Parse KB article HTML files locally and upload them as a single articles.json
blob to Azure Blob Storage.

Storing pre-parsed articles (clean extracted text) rather than raw HTML means
the app never has to parse JavaScript-heavy HTML at startup or during quality
assessment.

Usage:
    python scripts/upload_articles.py --source ./articles/ --container articles

Requires either:
  - AZURE_STORAGE_CONNECTION_STRING env var (local dev)
  - AZURE_STORAGE_ACCOUNT_NAME env var + Managed Identity (production)
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import blob_store
from main import load_articles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse KB articles locally and upload as articles.json to Azure Blob Storage."
    )
    parser.add_argument(
        "--source", default="./articles/",
        help="Local directory containing .htm/.html files (default: ./articles/)",
    )
    parser.add_argument(
        "--container", default="articles",
        help="Azure Blob container name (default: articles)",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"Error: Source directory not found: {source}", file=sys.stderr)
        sys.exit(1)

    service = blob_store.get_blob_service_client()
    if service is None:
        print(
            "Error: Azure Blob Storage not configured.\n"
            "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Parsing articles from {source}...")
    articles, contacts_text = load_articles(source)
    if not articles:
        print(f"Error: No .htm/.html files found in {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(articles)} article(s). Uploading to container '{args.container}'...")
    blob_store.upload_parsed_articles_to_blob(articles, contacts_text, args.container, service)
    print(f"Done. {len(articles)} article(s) uploaded as articles.json.")


if __name__ == "__main__":
    main()
