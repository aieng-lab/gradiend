"""
Spacy model loading with clearer errors and optional auto-download.
"""

from __future__ import annotations

from typing import Any

# Spacy model-not-found error code
_SPACY_E050 = "E050"


def load_spacy_model(
    model_name: str,
    download_if_missing: bool = False,
) -> Any:
    """Load a spacy model with improved error handling.

    Args:
        model_name: Spacy model name (e.g. "de_core_news_sm").
        download_if_missing: If True and the model is not found, attempt to
            download it via `python -m spacy download`. Default False.

    Returns:
        Loaded spacy Language model.

    Raises:
        ImportError: If spacy is not installed.
        OSError: If the model is not found, with a message including the
            install command: "python -m spacy download {model_name}".
    """
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spacy is required for morphological filtering. "
            "Install with: pip install spacy "
            "Or: pip install gradiend[data]"
        ) from e

    try:
        return spacy.load(model_name)
    except OSError as e:
        err_str = str(e)
        is_model_not_found = _SPACY_E050 in err_str or "Can't find model" in err_str

        if is_model_not_found and download_if_missing:
            try:
                from spacy.cli import download
                download(model_name)
                return spacy.load(model_name)
            except Exception as download_err:
                raise OSError(
                    f"Spacy model '{model_name}' not found and auto-download failed: {download_err}. "
                    f"Install manually with: python -m spacy download {model_name}"
                ) from e

        if is_model_not_found:
            raise OSError(
                f"Spacy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            ) from e
        raise
