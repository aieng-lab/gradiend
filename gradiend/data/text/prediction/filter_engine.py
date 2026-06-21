"""
Filter engine: find matching spans and mask them.

Uses string equality when no spacy_tags/use_lemma; otherwise spacy (lazy).
Span-based replacement for automatic disambiguation (e.g. "das" DET vs PRON).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple, Union

from tqdm import tqdm

from gradiend.data.core.spacy_util import load_spacy_model
from gradiend.data.text.filter_config import SpacyTagSpec, TextFilterConfig
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

AHOCORASICK_FEATURE_CLASS_THRESHOLD = 20
_AHOCORASICK_AVAILABLE: Optional[bool] = None


def _token_match_label(token: Any, *, use_lemma: bool) -> str:
    """Surface form or lemma stored as the masked label, depending on config."""
    if use_lemma:
        return str(token.lemma_).casefold()
    return token.text


def _has_ahocorasick() -> bool:
    global _AHOCORASICK_AVAILABLE
    if _AHOCORASICK_AVAILABLE is None:
        try:
            import ahocorasick  # noqa: F401
        except ImportError:
            _AHOCORASICK_AVAILABLE = False
        else:
            _AHOCORASICK_AVAILABLE = True
    return bool(_AHOCORASICK_AVAILABLE)

# Match result: (sentence, [(start, end, matched_text), ...])
MatchResult = Tuple[str, List[Tuple[int, int, str]]]

# Span: (start, end, matched_text)
Span = Tuple[int, int, str]


@dataclass(frozen=True)
class _CompiledTextFilter:
    class_id: str
    config: TextFilterConfig
    flat_targets: List[Tuple[str, Optional[SpacyTagSpec]]]
    use_spacy: bool
    targets_lower: set[str]
    pattern: Pattern[str]
    min_left_context_words: int


class _ActiveTargetMatcher:
    """Find candidate filters with one lexical pass over a sentence."""

    def __init__(self, filters: List[_CompiledTextFilter], *, prefer_ahocorasick: bool) -> None:
        self.filters = list(filters)
        self.target_to_filters: Dict[str, List[_CompiledTextFilter]] = {}
        targets: List[str] = []
        seen_targets = set()
        for compiled in self.filters:
            for target, _tags in compiled.flat_targets:
                key = target.lower()
                self.target_to_filters.setdefault(key, []).append(compiled)
                if key not in seen_targets:
                    seen_targets.add(key)
                    targets.append(target)

        self._automaton = None
        self._regex: Optional[Pattern[str]] = None
        if prefer_ahocorasick:
            self._automaton = self._build_ahocorasick(targets)
        if self._automaton is None:
            self._regex = re.compile(
                "|".join(rf"\b{re.escape(t)}\b" for t in targets),
                re.IGNORECASE,
            )

    @staticmethod
    def _build_ahocorasick(targets: List[str]) -> Any:
        try:
            import ahocorasick
        except ImportError:
            return None
        automaton = ahocorasick.Automaton()
        for target in targets:
            key = target.lower()
            automaton.add_word(key, key)
        automaton.make_automaton()
        return automaton

    @staticmethod
    def _is_word_char(value: str) -> bool:
        return value == "_" or value.isalnum()

    def _has_word_boundaries(self, text: str, start: int, end_exclusive: int) -> bool:
        before_ok = start == 0 or not self._is_word_char(text[start - 1])
        after_ok = end_exclusive >= len(text) or not self._is_word_char(text[end_exclusive])
        return before_ok and after_ok

    def candidates(self, sentence: str) -> List[_CompiledTextFilter]:
        if not self.filters:
            return []
        matched_targets = []
        if self._automaton is not None:
            lowered = sentence.lower()
            for end_idx, target in self._automaton.iter(lowered):
                start = end_idx - len(target) + 1
                if self._has_word_boundaries(lowered, start, end_idx + 1):
                    matched_targets.append(target)
        elif self._regex is not None:
            matched_targets = [m.group().lower() for m in self._regex.finditer(sentence)]

        if not matched_targets:
            return []

        seen_filter_ids = set()
        out: List[_CompiledTextFilter] = []
        for target in matched_targets:
            for compiled in self.target_to_filters.get(target, []):
                if compiled.class_id in seen_filter_ids:
                    continue
                seen_filter_ids.add(compiled.class_id)
                out.append(compiled)
        return out


def _compile_text_filter(
    class_id: str,
    config: TextFilterConfig,
    *,
    min_left_context_words_default: int = 0,
) -> _CompiledTextFilter:
    flat = config.flatten_targets_with_tags()
    use_spacy = config.use_lemma or any(tags is not None for _, tags in flat)
    pattern = re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t, _ in flat),
        re.IGNORECASE,
    )
    return _CompiledTextFilter(
        class_id=class_id,
        config=config,
        flat_targets=flat,
        use_spacy=use_spacy,
        targets_lower={t.lower() for t, _ in flat},
        pattern=pattern,
        min_left_context_words=config.effective_min_left_context_words(min_left_context_words_default),
    )


def _has_min_left_context_words(sentence: str, start: int, min_words: int) -> bool:
    if min_words <= 0:
        return True
    return len(re.findall(r"\w+", sentence[:start])) >= min_words


def _filter_spans_by_left_context_words(sentence: str, spans: List[Span], min_words: int) -> List[Span]:
    if min_words <= 0:
        return spans
    return [
        span
        for span in spans
        if _has_min_left_context_words(sentence, span[0], min_words)
    ]


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
    min_left_context_words_default: int = 0,
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
        min_left_context_words_default: Used when ``filter_config.min_left_context_words`` is ``None``.

    Returns:
        List of (sentence, [(start, end, matched_text), ...]) for each match.
    """
    flat = filter_config.flatten_targets_with_tags()
    use_spacy = filter_config.use_lemma or any(tags is not None for _, tags in flat)
    min_left_context_words = filter_config.effective_min_left_context_words(
        min_left_context_words_default
    )

    if not use_spacy:
        return _filter_string_only(
            sentences, flat, filter_config,
            min_left_context_words=min_left_context_words,
            desc=desc, max_matches=max_matches,
            total_target_overall=total_target_overall, total_so_far_initial=total_so_far_initial,
            stats=stats,
        )
    return _filter_with_spacy(
        sentences, flat, filter_config, spacy_model, download_if_missing,
        min_left_context_words=min_left_context_words,
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
    filter_config: TextFilterConfig,
    *,
    min_left_context_words: int,
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
        it, desc=desc or "Filtering", unit=" sentences",
        leave=True, total=total, position=0, dynamic_ncols=True,
        mininterval=2.0,
    )
    try:
        for sent in pbar:
            if max_matches is not None and len(results) >= max_matches:
                break
            matches: List[Tuple[int, int, str]] = []
            for m in pattern.finditer(sent):
                matches.append((m.start(), m.end(), m.group()))
            matches = _filter_spans_by_left_context_words(
                sent,
                matches,
                min_left_context_words,
            )
            if matches:
                results.append((sent, matches))
            pbar.set_postfix_str(
                _format_filter_postfix(
                    len(results), max_matches, total_target_overall, total_so_far_initial
                ),
                refresh=False,
            )
    except KeyboardInterrupt:
        if stats is not None:
            stats["interrupted"] = True
        logger.warning("Filtering interrupted by user; keeping %s matched sentences collected so far.", len(results))
    if stats is not None:
        stats["sentences_processed"] = pbar.n
    return results


def _filter_with_spacy(
    sentences: Union[Iterable[str], List[str]],
    flat_targets: List[Tuple[str, Optional[SpacyTagSpec]]],
    filter_config: TextFilterConfig,
    spacy_model: Optional[str],
    download_if_missing: bool = False,
    *,
    min_left_context_words: int,
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
        it, desc=desc or "Filtering", unit=" sent",
        leave=True, total=total, position=0, dynamic_ncols=True,
        mininterval=2.0,
    )
    try:
        for sent in pbar:
            if max_matches is not None and len(results) >= max_matches:
                break
            if not pattern.search(sent):
                pbar.set_postfix_str(
                    _format_filter_postfix(
                        len(results), max_matches, total_target_overall, total_so_far_initial
                    ),
                    refresh=False,
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
                    matches.append((start, end, _token_match_label(token, use_lemma=filter_config.use_lemma)))
                    break
            matches = _filter_spans_by_left_context_words(
                sent,
                matches,
                min_left_context_words,
            )
            if matches:
                results.append((sent, matches))
            pbar.set_postfix_str(
                _format_filter_postfix(
                    len(results), max_matches, total_target_overall, total_so_far_initial
                ),
                refresh=False,
            )
    except KeyboardInterrupt:
        if stats is not None:
            stats["interrupted"] = True
        logger.warning("Filtering interrupted by user; keeping %s matched sentences collected so far.", len(results))
    if stats is not None:
        stats["sentences_processed"] = pbar.n
    return results


def match_sentence_one_config(
    sentence: str,
    filter_config: TextFilterConfig,
    doc: Any = None,
    nlp: Any = None,
    min_left_context_words_default: int = 0,
) -> Optional[List[Span]]:
    """Check one sentence against one config; return match spans or None if no match.

    Uses doc when config needs spacy (lemma/tags); otherwise regex only.
    If config needs spacy and doc is None, uses nlp(sentence) when nlp is provided.
    """
    flat = filter_config.flatten_targets_with_tags()
    use_spacy = filter_config.use_lemma or any(tags is not None for _, tags in flat)
    min_left_context_words = filter_config.effective_min_left_context_words(
        min_left_context_words_default
    )

    if not use_spacy:
        targets = [t for t, _ in flat]
        pattern = re.compile(
            "|".join(rf"\b{re.escape(t)}\b" for t in targets),
            re.IGNORECASE,
        )
        matches: List[Span] = []
        for m in pattern.finditer(sentence):
            matches.append((m.start(), m.end(), m.group()))
        matches = _filter_spans_by_left_context_words(
            sentence,
            matches,
            min_left_context_words,
        )
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
            matches.append((start, end, _token_match_label(token, use_lemma=filter_config.use_lemma)))
            break
    matches = _filter_spans_by_left_context_words(
        sentence,
        matches,
        min_left_context_words,
    )
    return matches if matches else None


def _match_sentence_compiled(
    sentence: str,
    compiled: _CompiledTextFilter,
    doc: Any = None,
    nlp: Any = None,
) -> Optional[List[Span]]:
    """Check one sentence against a precompiled config; return match spans or None."""
    if not compiled.use_spacy:
        matches: List[Span] = []
        for m in compiled.pattern.finditer(sentence):
            matches.append((m.start(), m.end(), m.group()))
        matches = _filter_spans_by_left_context_words(
            sentence,
            matches,
            compiled.min_left_context_words,
        )
        return matches if matches else None

    if not compiled.pattern.search(sentence):
        return None
    if doc is None and nlp is not None:
        doc = nlp(sentence)
    if doc is None:
        return None

    matches = []
    for token in doc:
        text_lower = token.text.lower()
        lemma = token.lemma_.lower() if compiled.config.use_lemma else text_lower
        if text_lower not in compiled.targets_lower and lemma not in compiled.targets_lower:
            continue
        for target_str, tags in compiled.flat_targets:
            if compiled.config.use_lemma:
                if lemma != target_str.lower():
                    continue
            else:
                if text_lower != target_str.lower():
                    continue
            if tags is not None and not token_matches_tags(token, tags):
                continue
            start = token.idx
            end = token.idx + len(token.text)
            matches.append((start, end, _token_match_label(token, use_lemma=compiled.config.use_lemma)))
            break
    matches = _filter_spans_by_left_context_words(
        sentence,
        matches,
        compiled.min_left_context_words,
    )
    return matches if matches else None


def filter_sentences_multi(
    sentences: Union[Iterable[str], List[str]],
    configs_with_ids: List[Tuple[str, TextFilterConfig]],
    spacy_model: Optional[str] = None,
    download_if_missing: bool = False,
    max_matches_per_class: Optional[int] = None,
    total_target_overall: Optional[int] = None,
    stats: Optional[dict] = None,
    min_left_context_words_default: int = 0,
) -> Tuple[dict, dict]:
    """Single pass over sentences: prefilter once, then test only candidate configs.

    A sentence can match multiple classes; each match is recorded for that class.
    Returns (class_id -> list of (sentence, spans), stats dict with sentences_processed and, when capped,
    sentences_when_cap_reached: {class_id: sentence count when that class first reached its cap}).
    """
    if not configs_with_ids:
        return {}, stats or {}

    compiled_filters = [
        _compile_text_filter(class_id, cfg, min_left_context_words_default=min_left_context_words_default)
        for class_id, cfg in configs_with_ids
    ]
    any_use_spacy = any(compiled.use_spacy for compiled in compiled_filters)
    nlp = None
    if any_use_spacy:
        if spacy_model is None:
            raise ValueError("spacy_model required when any config has spacy_tags or use_lemma")
        nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)

    results: dict = {cid: [] for cid, _ in configs_with_ids}
    sentences_when_cap_reached: dict = {}  # class_id -> sentence count when that class first hit cap
    active_filters = list(compiled_filters)
    prefer_ahocorasick = False
    if len(compiled_filters) > AHOCORASICK_FEATURE_CLASS_THRESHOLD:
        if _has_ahocorasick():
            prefer_ahocorasick = True
            logger.info(
                "Using pyahocorasick global matcher for %s feature classes.",
                len(compiled_filters),
            )
        else:
            logger.info(
                "pyahocorasick is not installed; using regex global matcher for %s feature classes. "
                "Install pyahocorasick for faster many-class literal matching.",
                len(compiled_filters),
            )
    active_signature: Tuple[str, ...] = ()
    active_matcher: Optional[_ActiveTargetMatcher] = None
    it = sentences if isinstance(sentences, list) else iter(sentences)
    pbar = tqdm(
        it, desc="Filtering", unit=" sent", leave=True, position=0, dynamic_ncols=True,
        mininterval=2.0,
    )
    n_processed = 0

    try:
        for sent in pbar:
            if max_matches_per_class is not None and not active_filters:
                break
            signature = tuple(compiled.class_id for compiled in active_filters)
            if active_matcher is None or signature != active_signature:
                active_matcher = _ActiveTargetMatcher(
                    active_filters,
                    prefer_ahocorasick=prefer_ahocorasick,
                )
                active_signature = signature

            candidate_filters = active_matcher.candidates(sent)
            doc = None
            if nlp is not None and any(compiled.use_spacy for compiled in candidate_filters):
                doc = nlp(sent)

            for compiled in candidate_filters:
                class_id = compiled.class_id
                spans = _match_sentence_compiled(sent, compiled, doc=doc, nlp=nlp)
                if spans:
                    results[class_id].append((sent, spans))
                    if (
                        max_matches_per_class is not None
                        and len(results[class_id]) >= max_matches_per_class
                        and class_id not in sentences_when_cap_reached
                    ):
                        sentences_when_cap_reached[class_id] = n_processed + 1
            if max_matches_per_class is not None:
                active_filters = [
                    compiled for compiled in active_filters
                    if len(results[compiled.class_id]) < max_matches_per_class
                ]
            n_processed += 1
            total_so_far = sum(len(v) for v in results.values())
            postfix = f"matches={total_so_far} | active={len(active_filters)}/{len(compiled_filters)}"
            if max_matches_per_class is not None:
                completed = len(compiled_filters) - len(active_filters)
                postfix += f" | completed={completed} | total={total_so_far}/{total_target_overall or 0}"
                if total_target_overall and total_target_overall > 0:
                    postfix += f" ({100.0 * total_so_far / total_target_overall:.1f}%)"
            pbar.set_postfix_str(postfix, refresh=False)
    except KeyboardInterrupt:
        if stats is not None:
            stats["interrupted"] = True
        logger.warning(
            "Filtering interrupted by user; keeping partial matches: %s",
            {cid: len(v) for cid, v in results.items()},
        )

    if stats is not None:
        stats["sentences_processed"] = n_processed
        if sentences_when_cap_reached:
            stats["sentences_when_cap_reached"] = sentences_when_cap_reached
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
