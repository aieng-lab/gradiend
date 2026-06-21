"""Tests for Hugging Face download environment helpers."""

import os

from gradiend.util.hf_env import configure_hf_download_env


def test_configure_hf_download_env_replaces_missing_cert_bundle(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/virtualenv/certifi/cacert.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/nonexistent/virtualenv/certifi/cacert.pem")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/tmp/old-transformers-cache")

    configure_hf_download_env()

    bundle = os.environ.get("SSL_CERT_FILE")
    assert bundle
    assert os.path.isfile(bundle)
    assert os.environ.get("REQUESTS_CA_BUNDLE") == bundle
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert "TRANSFORMERS_CACHE" not in os.environ
