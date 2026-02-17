"""
Preprocessing for text: split to sentences, filter by length/chars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional

from gradiend.data.core.spacy_util import load_spacy_model
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextPreprocessConfig:
    """Configuration for preprocessing text."""

    split_to_sentences: bool = False
    min_chars: Optional[int] = None
    max_chars: Optional[int] = None
    exclude_chars: Optional[str] = None
    custom_filter: Optional[Callable[[str], bool]] = field(default=None, repr=False)


def preprocess_texts(
    texts: List[str],
    config: Optional[TextPreprocessConfig] = None,
    spacy_model: Optional[str] = None,
    download_if_missing: bool = True,
) -> List[str]:
    """Preprocess texts: optionally split to sentences, filter.

    If config is None or split_to_sentences is False, returns texts as-is
    (with optional length/char filtering).

    When split_to_sentences is True:
    - If spacy_model is given: use spacy sentencizer.
    - Otherwise: use simple regex split on .!?

    Args:
        texts: Input text strings (paragraphs or documents).
        config: TextPreprocessConfig. If None, returns texts as-is.
        spacy_model: Spacy model name for sentencizer (e.g. "de_core_news_sm").
            Only used when split_to_sentences=True.
        download_if_missing: If True, download the spacy model if it is not found.

    Returns:
        List of (optionally filtered) text strings.
    """
    if config is None:
        return texts

    out: List[str] = []
    if config.split_to_sentences:
        sentences = _split_to_sentences(texts, spacy_model, download_if_missing)
    else:
        sentences = texts

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if config.min_chars is not None and len(s) < config.min_chars:
            continue
        if config.max_chars is not None and len(s) > config.max_chars:
            continue
        if config.exclude_chars and any(c in s for c in config.exclude_chars):
            continue
        if config.custom_filter is not None and not config.custom_filter(s):
            continue
        out.append(s)

    logger.debug(f"preprocess_texts: {len(texts)} -> {len(out)} items")
    return out


def _split_on_newlines(texts: List[str]) -> List[str]:
    """Split texts on newlines first so no segment contains \\n. Returns non-empty stripped lines."""
    out: List[str] = []
    for t in texts:
        for line in t.split("\n"):
            s = line.strip()
            if s:
                out.append(s)
    return out


def _split_to_sentences(
    texts: List[str],
    spacy_model: Optional[str],
    download_if_missing: bool = False,
) -> List[str]:
    """Split texts into sentences. Newlines are split first (no sentence contains \\n), then spacy or regex."""
    lines = _split_on_newlines(texts)
    if spacy_model:
        return _split_with_spacy(lines, spacy_model, download_if_missing)
    return _split_simple(lines)


def _split_simple(texts: List[str]) -> List[str]:
    """Simple regex split on sentence boundaries."""
    pattern = re.compile(r"(?<=[.!?])\s+")
    out: List[str] = []
    for t in texts:
        parts = pattern.split(t)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out


def _split_with_spacy(
    texts: List[str],
    model_name: str,
    download_if_missing: bool = False,
) -> List[str]:
    """Split using spacy sentencizer."""
    nlp = load_spacy_model(model_name, download_if_missing=download_if_missing)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    out: List[str] = []
    for t in texts:
        doc = nlp(t)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                out.append(s)
    return out


def _apply_sentence_filters(s: str, config: TextPreprocessConfig) -> bool:
    """Return True if sentence passes config filters."""
    s = s.strip()
    if not s:
        return False
    if config.min_chars is not None and len(s) < config.min_chars:
        return False
    if config.max_chars is not None and len(s) > config.max_chars:
        return False
    if config.exclude_chars and any(c in s for c in config.exclude_chars):
        return False
    if config.custom_filter is not None and not config.custom_filter(s):
        return False
    return True


def iter_sentences_from_texts(
    texts: Iterable[str],
    config: Optional[TextPreprocessConfig] = None,
    spacy_model: Optional[str] = None,
    download_if_missing: bool = True,
) -> Iterator[str]:
    """Yield sentences from texts with on-the-fly splitting (no full materialization).

    Splits each text into sentences only when needed. Use this to avoid building
    a huge list of sentences before filtering; stop when you have enough matches.

    If config is None: yields each non-empty stripped text as one item.
    If config.split_to_sentences is False: yields each text after filters.
    If config.split_to_sentences is True: for each text, splits (spacy or regex)
    and yields each sentence that passes config filters.

    Args:
        texts: Input text strings (documents/chunks).
        config: TextPreprocessConfig. If None, yields non-empty stripped texts.
        spacy_model: Spacy model for sentencizer when split_to_sentences=True.
        download_if_missing: Passed to load_spacy_model.

    Yields:
        Individual sentences (or whole texts when not splitting).
    """
    if config is None:
        for t in texts:
            for line in t.split("\n"):
                s = line.strip()
                if s:
                    yield s
        return

    use_spacy = config.split_to_sentences and spacy_model
    nlp = None
    if use_spacy:
        nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    simple_pattern = re.compile(r"(?<=[.!?])\s+") if config.split_to_sentences else None

    for t in texts:
        # Split on newlines first so no sentence contains \n
        lines = (s.strip() for s in t.split("\n") if s.strip())
        for line in lines:
            if not config.split_to_sentences:
                if _apply_sentence_filters(line, config):
                    yield line
                continue
            if nlp is not None:
                doc = nlp(line)
                for sent in doc.sents:
                    s = sent.text.strip()
                    if s and _apply_sentence_filters(s, config):
                        yield s
            else:
                for p in simple_pattern.split(line):
                    p = p.strip()
                    if p and _apply_sentence_filters(p, config):
                        yield p
