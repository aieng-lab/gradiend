"""
Hugging Face download environment for minimal / HPC images.

Call ``configure_hf_download_env()`` before importing ``datasets`` or
``huggingface_hub`` so SSL and xet workarounds take effect.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _cert_bundle_is_usable(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        return Path(path).is_file()
    except OSError:
        return False


def configure_hf_download_env() -> None:
    """
    Disable hf-xet and point HTTP clients at certifi's CA bundle.

    Stale ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` values from another Python
    env (e.g. a login-node virtualenv) are cleared when the path does not exist
    in the current runtime (common with Apptainer + shared conda envs).
    """
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        if not _cert_bundle_is_usable(os.environ.get(key)):
            os.environ.pop(key, None)

    try:
        import certifi

        bundle = certifi.where()
        if _cert_bundle_is_usable(bundle):
            os.environ["SSL_CERT_FILE"] = bundle
            os.environ["REQUESTS_CA_BUNDLE"] = bundle
            os.environ["CURL_CA_BUNDLE"] = bundle
    except ImportError:
        pass

    # transformers>=4.x warns; HF_HOME is the supported cache root.
    os.environ.pop("TRANSFORMERS_CACHE", None)
