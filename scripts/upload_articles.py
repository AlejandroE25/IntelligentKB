#!/usr/bin/env python3
"""
Upload KB article HTML files to Azure Blob Storage.

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload KB articles to Azure Blob Storage.")
    parser.add_argument("--source", default="./articles/", help="Local directory containing .htm/.html files")
    parser.add_argument("--container", default="articles", help="Azure Blob container name")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"Error: Source directory not found: {source}", file=sys.stderr)
        sys.exit(1)

    blob_service = blob_store.get_blob_service_client()
    if blob_service is None:
        print(
            "Error: Azure Blob Storage not configured.\n"
            "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME.",
            file=sys.stderr,
        )
        sys.exit(1)

    article_files = [p for p in source.iterdir() if p.suffix.lower() in (".htm", ".html")]
    if not article_files:
        print(f"No .htm/.html files found in {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading {len(article_files)} article(s) from {source} to container '{args.container}'...")
    count = blob_store.upload_articles_to_blob(source, args.container, blob_service)
    print(f"Done. {count} file(s) uploaded.")


if __name__ == "__main__":
    main()
