"""Neutral-data filtering for text-prediction data creation."""

from __future__ import annotations

import re
from itertools import islice
from typing import Dict, Iterable, List, Optional, Tuple, Union

from gradiend.util.tqdm_utils import gradiend_tqdm

from gradiend.data.core.spacy_util import load_spacy_model
from gradiend.data.text import SpacyTagSpec
from gradiend.data.text.prediction.filter_engine import token_matches_tags
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _filter_neutral(
    sentences: Union[Iterable[str], List[str]],
    excluded_words: List[str],
    excluded_spacy_tags: Optional[Union[SpacyTagSpec, List[SpacyTagSpec]]],
    spacy_model: Optional[str],
    download_if_missing: bool = False,
    max_size: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Filter out sentences containing excluded words or spacy tags.

    Consumes the sentence iterable (e.g. from iter_sentences_from_texts).
    If max_size is set, stops after collecting that many kept sentences.

    Returns:
        (kept_sentences, stats) with stats e.g. {"kept": N, "excluded": M, "total": T}.
    """
    if excluded_spacy_tags is not None and spacy_model is None:
        raise ValueError("spacy_model required when excluded_spacy_tags is set")

    word_pattern = None
    if excluded_words:
        word_pattern = re.compile(
            "|".join(rf"\b{re.escape(w)}\b" for w in excluded_words),
            re.IGNORECASE,
        )

    specs = excluded_spacy_tags
    if specs is not None and isinstance(specs, dict):
        specs = [specs]

    if not excluded_words and not specs:
        out = list(sentences) if max_size is None else list(islice(sentences, max_size))
        return out, {"kept": len(out), "excluded": 0, "total": len(out)}

    nlp = None
    if specs:
        nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)

    out: List[str] = []
    excluded_count = 0
    total_count = 0
    it = iter(sentences)
    pbar = gradiend_tqdm(
        it, desc="Neutral filter", unit=" sent", leave=True, position=0, mininterval=2.0,
    )
    try:
        for sent in pbar:
            total_count += 1
            if word_pattern and word_pattern.search(sent):
                excluded_count += 1
                pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=False)
                continue
            if nlp is not None and specs:
                doc = nlp(sent)
                for token in doc:
                    if any(token_matches_tags(token, s) for s in specs):
                        excluded_count += 1
                        pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=False)
                        break
                else:
                    out.append(sent)
                    pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=False)
                    if max_size is not None and len(out) >= max_size:
                        break
            else:
                out.append(sent)
                pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=False)
                if max_size is not None and len(out) >= max_size:
                    break
    except KeyboardInterrupt:
        logger.warning("Neutral filtering interrupted by user; keeping %s rows collected so far.", len(out))
    stats = {"kept": len(out), "excluded": excluded_count, "total": total_count}
    return out, stats

__all__ = ["_filter_neutral"]
