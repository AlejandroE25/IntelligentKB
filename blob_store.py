"""
Azure Blob Storage integration for article and data persistence.

Gracefully degrades to local filesystem if Azure is not configured.

Authentication priority:
  1. DefaultAzureCredential (Managed Identity on Azure Web App — no secrets needed)
  2. AZURE_STORAGE_CONNECTION_STRING env var (local dev)
  3. None — all blob operations are skipped, local filesystem is used
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_AZURE_AVAILABLE = False
try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import ResourceNotFoundError, AzureError
    _AZURE_AVAILABLE = True
except (ImportError, Exception):
    logger.info("azure-storage-blob not available; blob storage disabled")


def get_blob_service_client():
    """Return a BlobServiceClient or None if Azure is not configured/available."""
    if not _AZURE_AVAILABLE:
        return None

    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()

    if connection_string:
        try:
            return BlobServiceClient.from_connection_string(connection_string)
        except Exception as exc:
            logger.warning("Failed to init BlobServiceClient from connection string: %s", exc)
            return None

    if account_name:
        try:
            credential = DefaultAzureCredential()
            return BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=credential,
            )
        except Exception as exc:
            logger.warning("Failed to init BlobServiceClient via DefaultAzureCredential: %s", exc)
            return None

    return None


def download_articles_from_blob(
    container_name: str,
    dest_dir: Path,
    blob_service,
) -> int:
    """Download all .htm/.html blobs from container to dest_dir.

    Returns the count of files downloaded. Skips blobs that don't end in
    .htm or .html. Overwrites any existing files in dest_dir.
    """
    if not _AZURE_AVAILABLE:
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    container_client = blob_service.get_container_client(container_name)

    count = 0
    try:
        for blob in container_client.list_blobs():
            name: str = blob.name
            if not (name.endswith(".htm") or name.endswith(".html")):
                continue
            dest_path = dest_dir / Path(name).name
            blob_client = container_client.get_blob_client(name)
            with open(dest_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            count += 1
    except AzureError as exc:
        logger.error("Error downloading articles from blob container '%s': %s", container_name, exc)
        raise

    logger.info("Downloaded %d articles from blob container '%s'", count, container_name)
    return count


def upload_blob_json(
    container_name: str,
    blob_name: str,
    data: dict | list,
    blob_service,
) -> None:
    """Upload a Python dict/list as a JSON blob (overwrites if exists)."""
    if not _AZURE_AVAILABLE:
        return

    payload = json.dumps(data, indent=2, default=str).encode("utf-8")
    container_client = blob_service.get_container_client(container_name)
    try:
        container_client.upload_blob(blob_name, payload, overwrite=True)
    except AzureError as exc:
        logger.warning("Failed to upload blob '%s' to '%s': %s", blob_name, container_name, exc)


def download_blob_json(
    container_name: str,
    blob_name: str,
    blob_service,
) -> dict | None:
    """Download and parse a JSON blob. Returns None if blob doesn't exist."""
    if not _AZURE_AVAILABLE:
        return None

    container_client = blob_service.get_container_client(container_name)
    try:
        blob_client = container_client.get_blob_client(blob_name)
        raw = blob_client.download_blob().readall()
        return json.loads(raw)
    except ResourceNotFoundError:
        return None
    except (AzureError, json.JSONDecodeError) as exc:
        logger.warning("Failed to download/parse blob '%s' from '%s': %s", blob_name, container_name, exc)
        return None


def upload_articles_to_blob(
    source_dir: Path,
    container_name: str,
    blob_service,
) -> int:
    """Upload all .htm/.html files from source_dir to blob container.

    Returns count of files uploaded. Used by scripts/upload_articles.py.
    """
    if not _AZURE_AVAILABLE:
        return 0

    container_client = blob_service.get_container_client(container_name)
    count = 0
    for path in sorted(source_dir.iterdir()):
        if path.suffix.lower() not in (".htm", ".html"):
            continue
        with open(path, "rb") as f:
            container_client.upload_blob(path.name, f, overwrite=True)
        count += 1
        logger.info("Uploaded %s", path.name)

    logger.info("Uploaded %d articles to blob container '%s'", count, container_name)
    return count
