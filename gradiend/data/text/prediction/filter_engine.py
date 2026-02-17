"""
Filter engine: find matching spans and mask them.

Uses string equality when no spacy_tags/use_lemma; otherwise spacy (lazy).
Span-based replacement for automatic disambiguation (e.g. "das" DET vs PRON).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Tuple, Union

from tqdm import tqdm

from gradiend.data.core.spacy_util import load_spacy_model
from gradiend.data.text.filter_config import SpacyTagSpec, TextFilterConfig
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

# Match result: (sentence, [(start, end, matched_text), ...])
MatchResult = Tuple[str, List[Tuple[int, int, str]]]

# Span: (start, end, matched_text)
Span = Tuple[int, int, str]


def filter_sentences(
    sentences: Union[Iterable[str], List[str]],
    filter_config: TextFilterConfig,
    spacy_model: Optional[str] = None,
    download_if_missing: bool = False,
    desc: Optional[str] = None,
    max_matches: Optional[int] = None,
    total_target_overall: Optional[int] = None,
    total_so_far_initial: int = 0,
    stats: Optional[dict] = None,
) -> List[MatchResult]:
    """Find sentences matching TextFilterConfig and return match spans.

    When spacy_tags is None and use_lemma is False: uses regex only (no spacy).
    Otherwise: runs spacy only on sentences that pass string pre-filter.

    Args:
        sentences: List or iterable of sentences (e.g. from iter_sentences_from_texts).
        desc: Optional description for tqdm progress (e.g. class/group id).
        max_matches: If set, stop after this many matches (for early exit when streaming).
        total_target_overall: If set, show overall progress in bar (total matches across classes).
        total_so_far_initial: Starting count for overall progress (matches from previous classes).
        stats: If provided, updated with "sentences_processed" (number of sentences consumed).

    Returns:
        List of (sentence, [(start, end, matched_text), ...]) for each match.
    """
    flat = filter_config.flatten_targets_with_tags()
    use_spacy = filter_config.use_lemma or any(tags is not None for _, tags in flat)

    if not use_spacy:
        return _filter_string_only(
            sentences, flat, filter_config.mask,
            desc=desc, max_matches=max_matches,
            total_target_overall=total_target_overall, total_so_far_initial=total_so_far_initial,
            stats=stats,
        )
    return _filter_with_spacy(
        sentences, flat, filter_config, spacy_model, download_if_missing,
        desc=desc, max_matches=max_matches,
        total_target_overall=total_target_overall, total_so_far_initial=total_so_far_initial,
        stats=stats,
    )


def _format_filter_postfix(
    n_matches: int,
    max_matches: Optional[int],
    total_target_overall: Optional[int],
    total_so_far_initial: int,
) -> str:
    parts = [f"matches={n_matches}"]
    if max_matches is not None:
        parts[0] = f"matches={n_matches}/{max_matches}"
    if total_target_overall is not None:
        total_so_far = total_so_far_initial + n_matches
        pct = 100.0 * total_so_far / total_target_overall
        parts.append(f"total={total_so_far}/{total_target_overall} ({pct:.1f}%)")
    return " | ".join(parts)


def _filter_string_only(
    sentences: Union[Iterable[str], List[str]],
    flat_targets: List[Tuple[str, Optional[SpacyTagSpec]]],
    mask_str: str,
    desc: Optional[str] = None,
    max_matches: Optional[int] = None,
    total_target_overall: Optional[int] = None,
    total_so_far_initial: int = 0,
    stats: Optional[dict] = None,
) -> List[MatchResult]:
    """String equality only; no spacy."""
    targets = [t for t, _ in flat_targets]
    pattern = re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t in targets),
        re.IGNORECASE,
    )
    results: List[MatchResult] = []
    it = sentences if isinstance(sentences, list) else iter(sentences)
    total = len(sentences) if isinstance(sentences, list) else None
    pbar = tqdm(
        it, desc=desc or "Filtering", unit="sent",
        leave=True, total=total, position=0, dynamic_ncols=True,
    )
    for sent in pbar:
        if max_matches is not None and len(results) >= max_matches:
            break
        matches: List[Tuple[int, int, str]] = []
        for m in pattern.finditer(sent):
            matches.append((m.start(), m.end(), m.group()))
        if matches:
            results.append((sent, matches))
        pbar.set_postfix_str(
            _format_filter_postfix(
                len(results), max_matches, total_target_overall, total_so_far_initial
            ),
            refresh=True,
        )
    if stats is not None:
        stats["sentences_processed"] = pbar.n
    return results


def _filter_with_spacy(
    sentences: Union[Iterable[str], List[str]],
    flat_targets: List[Tuple[str, Optional[SpacyTagSpec]]],
    filter_config: TextFilterConfig,
    spacy_model: Optional[str],
    download_if_missing: bool = False,
    desc: Optional[str] = None,
    max_matches: Optional[int] = None,
    total_target_overall: Optional[int] = None,
    total_so_far_initial: int = 0,
    stats: Optional[dict] = None,
) -> List[MatchResult]:
    """Use spacy; pre-filter with regex to avoid nlp() on every sentence."""
    if spacy_model is None:
        raise ValueError("spacy_model required when TextFilterConfig has spacy_tags or use_lemma")
    nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)
    targets_lower = {t.lower() for t, _ in flat_targets}
    pattern = re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t, _ in flat_targets),
        re.IGNORECASE,
    )
    results: List[MatchResult] = []
    it = sentences if isinstance(sentences, list) else iter(sentences)
    total = len(sentences) if isinstance(sentences, list) else None
    pbar = tqdm(
        it, desc=desc or "Filtering", unit="sent",
        leave=True, total=total, position=0, dynamic_ncols=True,
    )
    for sent in pbar:
        if max_matches is not None and len(results) >= max_matches:
            break
        if not pattern.search(sent):
            pbar.set_postfix_str(
                _format_filter_postfix(
                    len(results), max_matches, total_target_overall, total_so_far_initial
                ),
                refresh=True,
            )
            continue
        doc = nlp(sent)
        matches: List[Tuple[int, int, str]] = []
        for token in doc:
            text = token.text
            text_lower = text.lower()
            lemma = token.lemma_.lower() if filter_config.use_lemma else text_lower
            match_text = lemma if filter_config.use_lemma else text
            if text_lower not in targets_lower and lemma not in targets_lower:
                continue
            for target_str, tags in flat_targets:
                if filter_config.use_lemma:
                    if lemma != target_str.lower():
                        continue
                else:
                    if text_lower != target_str.lower():
                        continue
                if tags is not None and not token_matches_tags(token, tags):
                    continue
                start = token.idx
                end = token.idx + len(token.text)
                matches.append((start, end, token.text))
                break
        if matches:
            results.append((sent, matches))
        pbar.set_postfix_str(
            _format_filter_postfix(
                len(results), max_matches, total_target_overall, total_so_far_initial
            ),
            refresh=True,
        )
    if stats is not None:
        stats["sentences_processed"] = pbar.n
    return results


def match_sentence_one_config(
    sentence: str,
    filter_config: TextFilterConfig,
    doc: Any = None,
    nlp: Any = None,
) -> Optional[List[Span]]:
    """Check one sentence against one config; return match spans or None if no match.

    Uses doc when config needs spacy (lemma/tags); otherwise regex only.
    If config needs spacy and doc is None, uses nlp(sentence) when nlp is provided.
    """
    flat = filter_config.flatten_targets_with_tags()
    use_spacy = filter_config.use_lemma or any(tags is not None for _, tags in flat)

    if not use_spacy:
        targets = [t for t, _ in flat]
        pattern = re.compile(
            "|".join(rf"\b{re.escape(t)}\b" for t in targets),
            re.IGNORECASE,
        )
        matches: List[Span] = []
        for m in pattern.finditer(sentence):
            matches.append((m.start(), m.end(), m.group()))
        return matches if matches else None

    # Spacy path: need doc
    if doc is None and nlp is not None:
        doc = nlp(sentence)
    if doc is None:
        return None
    targets_lower = {t.lower() for t, _ in flat}
    pattern = re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t, _ in flat),
        re.IGNORECASE,
    )
    if not pattern.search(sentence):
        return None
    matches = []
    for token in doc:
        text_lower = token.text.lower()
        lemma = token.lemma_.lower() if filter_config.use_lemma else text_lower
        if text_lower not in targets_lower and lemma not in targets_lower:
            continue
        for target_str, tags in flat:
            if filter_config.use_lemma:
                if lemma != target_str.lower():
                    continue
            else:
                if text_lower != target_str.lower():
                    continue
            if tags is not None and not token_matches_tags(token, tags):
                continue
            start = token.idx
            end = token.idx + len(token.text)
            matches.append((start, end, token.text))
            break
    return matches if matches else None


def filter_sentences_multi(
    sentences: Union[Iterable[str], List[str]],
    configs_with_ids: List[Tuple[str, TextFilterConfig]],
    spacy_model: Optional[str] = None,
    download_if_missing: bool = False,
    max_matches_per_class: Optional[int] = None,
    total_target_overall: Optional[int] = None,
    stats: Optional[dict] = None,
) -> Tuple[dict, dict]:
    """Single pass over sentences: for each sentence, run spacy once (if needed) and test all configs.

    A sentence can match multiple classes; each match is recorded for that class.
    Returns (class_id -> list of (sentence, spans), stats dict with sentences_processed and per-class counts).
    """
    if not configs_with_ids:
        return {}, stats or {}

    # Precompute which configs need spacy
    any_use_spacy = False
    for _, cfg in configs_with_ids:
        flat = cfg.flatten_targets_with_tags()
        if cfg.use_lemma or any(tags is not None for _, tags in flat):
            any_use_spacy = True
            break
    nlp = None
    if any_use_spacy:
        if spacy_model is None:
            raise ValueError("spacy_model required when any config has spacy_tags or use_lemma")
        nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)

    results: dict = {cid: [] for cid, _ in configs_with_ids}
    it = sentences if isinstance(sentences, list) else iter(sentences)
    pbar = tqdm(it, desc="Filtering", unit="sent", leave=True, position=0, dynamic_ncols=True)
    n_processed = 0

    for sent in pbar:
        if max_matches_per_class is not None:
            if all(len(results[cid]) >= max_matches_per_class for cid, _ in configs_with_ids):
                break
        doc = nlp(sent) if nlp is not None else None
        for class_id, cfg in configs_with_ids:
            if max_matches_per_class is not None and len(results[class_id]) >= max_matches_per_class:
                continue
            spans = match_sentence_one_config(sent, cfg, doc=doc, nlp=nlp)
            if spans:
                results[class_id].append((sent, spans))
        n_processed += 1
        total_so_far = sum(len(v) for v in results.values())
        postfix = " | ".join(f"{cid}:{len(v)}" for cid, v in results.items())
        if max_matches_per_class is not None:
            postfix += f" | total={total_so_far}/{total_target_overall or 0}"
            if total_target_overall and total_target_overall > 0:
                postfix += f" ({100.0 * total_so_far / total_target_overall:.1f}%)"
        pbar.set_postfix_str(postfix, refresh=True)

    if stats is not None:
        stats["sentences_processed"] = n_processed
    return results, stats or {}


def token_matches_tags(token: Any, tags: SpacyTagSpec) -> bool:
    """Check if token satisfies all morph/tag constraints."""
    for key, value in tags.items():
        key_lower = key.lower()
        if key_lower == "pos":
            if token.pos_ != value:
                return False
        elif key_lower in ("dep", "tag"):
            attr = getattr(token, f"{key_lower}_", None)
            if attr != value:
                return False
        else:
            morph_val = token.morph.get(key)
            if morph_val is None:
                return False
            expected = [value] if isinstance(value, str) else list(value)
            if morph_val != expected:
                return False
    return True


def mask_sentence(sentence: str, match_spans: List[Tuple[int, int, str]], mask_str: str) -> str:
    """Replace matched spans with mask_str (span-based, no placeholder trick)."""
    if not match_spans:
        return sentence
    # Sort by start descending to replace from end, preserving indices
    sorted_spans = sorted(match_spans, key=lambda x: x[0], reverse=True)
    result = sentence
    for start, end, _ in sorted_spans:
        result = result[:start] + mask_str + result[end:]
    return result
