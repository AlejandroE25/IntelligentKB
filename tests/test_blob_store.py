"""Tests for blob_store.py — all Azure SDK calls are mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import blob_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(blobs=None):
    """Return a mock BlobServiceClient with a container that lists given blobs."""
    service = MagicMock()
    container = MagicMock()
    service.get_container_client.return_value = container

    blob_items = []
    for name, content in (blobs or {}).items():
        item = MagicMock()
        item.name = name
        blob_client = MagicMock()
        blob_client.download_blob.return_value.readall.return_value = content
        container.get_blob_client.side_effect = lambda n, _cache={name: blob_client}: _cache.get(n, MagicMock())
        blob_items.append(item)
    container.list_blobs.return_value = blob_items

    return service, container


# ---------------------------------------------------------------------------
# get_blob_service_client
# ---------------------------------------------------------------------------

class TestGetBlobServiceClient:
    def test_returns_none_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        result = blob_store.get_blob_service_client()
        assert result is None

    def test_uses_connection_string_when_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=abc123==;EndpointSuffix=core.windows.net")
        monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
        if blob_store._AZURE_AVAILABLE:
            with patch("blob_store.BlobServiceClient.from_connection_string") as mock_init:
                mock_init.return_value = MagicMock()
                result = blob_store.get_blob_service_client()
                assert result is not None
                mock_init.assert_called_once()

    def test_uses_default_credential_when_account_name_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "myaccount")
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        if blob_store._AZURE_AVAILABLE:
            with patch("blob_store.DefaultAzureCredential") as mock_cred, \
                 patch("blob_store.BlobServiceClient") as mock_client:
                mock_client.return_value = MagicMock()
                result = blob_store.get_blob_service_client()
                assert result is not None
                mock_client.assert_called_once()


# ---------------------------------------------------------------------------
# download_articles_from_blob
# ---------------------------------------------------------------------------

class TestDownloadArticlesFromBlob:
    def test_downloads_htm_and_html_blobs(self, tmp_path):
        if not blob_store._AZURE_AVAILABLE:
            pytest.skip("azure-storage-blob not installed")

        article1_content = b"<html>article 1</html>"
        article2_content = b"<html>article 2</html>"

        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container

        blobs = [
            ("article1.html", article1_content),
            ("article2.htm", article2_content),
        ]
        blob_items = []
        blob_clients = {}
        for name, content in blobs:
            item = MagicMock()
            item.name = name
            bc = MagicMock()
            bc.download_blob.return_value.readall.return_value = content
            blob_clients[name] = bc
            blob_items.append(item)
        container.list_blobs.return_value = blob_items
        container.get_blob_client.side_effect = lambda n: blob_clients[n]

        count = blob_store.download_articles_from_blob("articles", tmp_path, service)

        assert count == 2
        assert (tmp_path / "article1.html").read_bytes() == article1_content
        assert (tmp_path / "article2.htm").read_bytes() == article2_content

    def test_skips_non_article_blobs(self, tmp_path):
        if not blob_store._AZURE_AVAILABLE:
            pytest.skip("azure-storage-blob not installed")

        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container

        readme_item = MagicMock()
        readme_item.name = "README.md"
        json_item = MagicMock()
        json_item.name = "data.json"
        container.list_blobs.return_value = [readme_item, json_item]

        count = blob_store.download_articles_from_blob("articles", tmp_path, service)
        assert count == 0
        assert list(tmp_path.iterdir()) == []

    def test_returns_zero_when_azure_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blob_store, "_AZURE_AVAILABLE", False)
        service = MagicMock()
        count = blob_store.download_articles_from_blob("articles", tmp_path, service)
        assert count == 0


# ---------------------------------------------------------------------------
# upload_blob_json / download_blob_json
# ---------------------------------------------------------------------------

class TestBlobJson:
    def test_upload_encodes_as_json(self):
        if not blob_store._AZURE_AVAILABLE:
            pytest.skip("azure-storage-blob not installed")

        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container

        data = {"key": "value", "count": 42}
        blob_store.upload_blob_json("appdata", "test.json", data, service)

        container.upload_blob.assert_called_once()
        name_arg, payload_arg = container.upload_blob.call_args[0]
        assert name_arg == "test.json"
        assert json.loads(payload_arg) == data

    def test_download_returns_parsed_dict(self):
        if not blob_store._AZURE_AVAILABLE:
            pytest.skip("azure-storage-blob not installed")

        expected = {"article_flags": {}, "session_feedback": []}
        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container
        blob_client = MagicMock()
        blob_client.download_blob.return_value.readall.return_value = json.dumps(expected).encode()
        container.get_blob_client.return_value = blob_client

        result = blob_store.download_blob_json("appdata", "feedback.json", service)
        assert result == expected

    def test_download_returns_none_for_missing_blob(self):
        if not blob_store._AZURE_AVAILABLE:
            pytest.skip("azure-storage-blob not installed")

        from azure.core.exceptions import ResourceNotFoundError
        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container
        blob_client = MagicMock()
        blob_client.download_blob.side_effect = ResourceNotFoundError("not found")
        container.get_blob_client.return_value = blob_client

        result = blob_store.download_blob_json("appdata", "missing.json", service)
        assert result is None

    def test_upload_skipped_when_azure_unavailable(self, monkeypatch):
        monkeypatch.setattr(blob_store, "_AZURE_AVAILABLE", False)
        service = MagicMock()
        blob_store.upload_blob_json("appdata", "test.json", {}, service)
        service.get_container_client.assert_not_called()

    def test_download_returns_none_when_azure_unavailable(self, monkeypatch):
        monkeypatch.setattr(blob_store, "_AZURE_AVAILABLE", False)
        service = MagicMock()
        result = blob_store.download_blob_json("appdata", "test.json", service)
        assert result is None


# ---------------------------------------------------------------------------
# upload_articles_to_blob
# ---------------------------------------------------------------------------

class TestUploadArticlesToBlob:
    def test_uploads_htm_and_html_files(self, tmp_path):
        if not blob_store._AZURE_AVAILABLE:
            pytest.skip("azure-storage-blob not installed")

        (tmp_path / "article1.html").write_bytes(b"<html>1</html>")
        (tmp_path / "article2.htm").write_bytes(b"<html>2</html>")
        (tmp_path / "notes.txt").write_bytes(b"ignore me")

        service = MagicMock()
        container = MagicMock()
        service.get_container_client.return_value = container

        count = blob_store.upload_articles_to_blob(tmp_path, "articles", service)
        assert count == 2
        uploaded_names = {c[0][0] for c in container.upload_blob.call_args_list}
        assert "article1.html" in uploaded_names
        assert "article2.htm" in uploaded_names
        assert "notes.txt" not in uploaded_names

    def test_returns_zero_when_azure_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blob_store, "_AZURE_AVAILABLE", False)
        service = MagicMock()
        count = blob_store.upload_articles_to_blob(tmp_path, "articles", service)
        assert count == 0
