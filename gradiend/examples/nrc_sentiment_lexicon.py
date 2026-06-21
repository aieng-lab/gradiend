"""
NRC Emotion Lexicon helpers for text-prediction data generation.

Loads the NRC Word-Emotion Association Lexicon (Mohammad & Turney, 2013) from the
Hugging Face mirror ``vladinc/nrc``. Positive/negative *sentiment* associations are
used as mask targets; base sentences still come from a separate corpus (e.g. tweet_eval).

Citation::

    Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association
    lexicon. Computational Intelligence, 29(3), 436–465.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gradiend.data.core.spacy_util import load_spacy_model

NRC_HF_DATASET = "vladinc/nrc"
NRC_CITATION = (
    "NRC Emotion Lexicon (Mohammad & Turney, 2013) via Hugging Face dataset "
    f"{NRC_HF_DATASET!r}"
)

_WORD_BOUNDARY = re.compile(r"[a-z0-9']+")


def dedupe_targets_case_insensitive(words: Sequence[str]) -> List[str]:
    """Keep one canonical target per case-insensitive key (casefolded form)."""
    unique: Dict[str, str] = {}
    for word in words:
        key = str(word).strip().casefold()
        if key and key not in unique:
            unique[key] = key
    return sorted(unique.values())


def canonicalize_target_word(word: str) -> str:
    """Normalize a mask target / label for case-insensitive identity."""
    return str(word).strip().casefold()


def load_nrc_sentiment_words(
    *,
    dataset_name: str = NRC_HF_DATASET,
    split: str = "train",
) -> Tuple[List[str], List[str]]:
    """Return sorted positive/negative sentiment word lists from the NRC lexicon."""
    from datasets import load_dataset

    rows = load_dataset(dataset_name, split=split)["text"]
    positive: set[str] = set()
    negative: set[str] = set()
    for row in rows:
        parts = str(row).split("\t")
        if len(parts) != 3:
            continue
        word, category, flag = parts
        if flag != "1":
            continue
        if category == "positive":
            positive.add(word)
        elif category == "negative":
            negative.add(word)
    return sorted(positive), sorted(negative)


def exclude_ambiguous_nrc_polarity_words(
    positive: Sequence[str],
    negative: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Drop NRC words that appear in both the positive and negative sentiment lists."""
    pos = list(positive)
    neg = list(negative)
    both_surface = set(pos) & set(neg)
    pos = [w for w in pos if w not in both_surface]
    neg = [w for w in neg if w not in both_surface]
    return pos, neg, sorted(both_surface)


def _nrc_match_keys(
    words: Sequence[str],
    nlp: Any,
) -> Tuple[set[str], Dict[str, str]]:
    """Normalized NRC keys (surface + lemma) and canonical lemma -> target string."""
    keys: set[str] = set()
    representative: Dict[str, str] = {}
    for word in words:
        word_str = str(word).strip()
        if not word_str:
            continue
        keys.add(word_str.casefold())
        doc = nlp(word_str)
        if not doc:
            continue
        token = doc[0]
        lemma = str(token.lemma_)
        lemma_cf = lemma.casefold()
        keys.add(lemma_cf)
        representative.setdefault(lemma_cf, str(lemma).casefold())
    return keys, representative


def filter_lexicon_to_corpus(
    words: Sequence[str],
    corpus_texts: Sequence[str],
    *,
    max_words: Optional[int] = None,
    min_count: int = 1,
) -> List[str]:
    """
    Keep lexicon words attested in *corpus_texts*, ranked by sentence frequency.

    Matching is case-insensitive on alphanumeric tokens extracted from the corpus.
    Each word is counted at most once per base text row (sentence), matching how
    text-prediction data generation emits one training row per matching sentence.
    Words below *min_count* are dropped so masking targets are reproducibly grounded
    in the base dataset.
    """
    if not words:
        return []
    word_set = {str(w).casefold() for w in words}
    counts: Counter[str] = Counter()
    for text in corpus_texts:
        seen: set[str] = set()
        for token in _WORD_BOUNDARY.findall(str(text).casefold()):
            if token in word_set and token not in seen:
                seen.add(token)
                counts[token] += 1
    ranked = sorted(
        (word for word, count in counts.items() if count >= min_count),
        key=lambda w: (-counts[w], w),
    )
    if max_words is not None and max_words > 0:
        ranked = ranked[:max_words]
    # Return original casing from lexicon where possible (first match by casefold).
    casefold_to_word = {str(w).casefold(): str(w) for w in words}
    return [casefold_to_word[w] for w in ranked]


def filter_lexicon_adjectives_in_corpus(
    words: Sequence[str],
    corpus_texts: Sequence[str],
    *,
    spacy_model: str = "en_core_web_sm",
    max_words: Optional[int] = None,
    min_count: int = 1,
    download_if_missing: bool = True,
    group_by_lemma: bool = True,
) -> List[str]:
    """
    Keep NRC lexicon words that appear as adjectives (spaCy ``ADJ``) in the corpus.

    Each target is counted at most once per base text row (sentence). When
    *group_by_lemma* is True (default), inflections share one bucket and the returned
    label is the lemma. When False, each attested surface form is a separate target
    (matching ``TextFilterConfig(use_lemma=False)``).
    """
    if not words:
        return []
    nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)
    match_keys, nrc_representative = _nrc_match_keys(words, nlp)
    pattern = re.compile(
        "|".join(
            rf"\b{re.escape(w)}\b"
            for w in sorted(match_keys, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )
    counts: Counter[str] = Counter()
    representative: Dict[str, str] = dict(nrc_representative)
    for text in corpus_texts:
        text_str = str(text)
        if not pattern.search(text_str):
            continue
        seen_in_sentence: set[str] = set()
        for token in nlp(text_str):
            if token.pos_ != "ADJ":
                continue
            text_cf = token.text.casefold()
            lemma_cf = token.lemma_.casefold()
            if text_cf not in match_keys and lemma_cf not in match_keys:
                continue
            canonical = lemma_cf if group_by_lemma else text_cf
            if canonical in seen_in_sentence:
                continue
            seen_in_sentence.add(canonical)
            counts[canonical] += 1
            if group_by_lemma:
                representative.setdefault(canonical, str(token.lemma_).casefold())
            else:
                representative.setdefault(canonical, str(token.text).casefold())
    ranked = sorted(
        (word for word, count in counts.items() if count >= min_count),
        key=lambda w: (-counts[w], w),
    )
    if max_words is not None and max_words > 0:
        ranked = ranked[:max_words]
    return [representative[k] for k in ranked]


def build_sentiment_lexicon_for_corpus(
    corpus_texts: Sequence[str],
    *,
    max_words_per_class: int = 50,
    min_count_per_word: int = 1,
    dataset_name: str = NRC_HF_DATASET,
    require_adjectives: bool = True,
    spacy_model: str = "en_core_web_sm",
    download_if_missing: bool = True,
    group_by_lemma: bool = True,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Build positive/negative mask-target lists for a corpus.

    When *require_adjectives* is True (default), only words attested as spaCy
    ``ADJ`` in the corpus are kept, ranked by canonical lemma when
    *group_by_lemma* is True (default). *min_count_per_word* drops weakly attested
    targets before per-target balancing can collapse the generated dataset to one
    row per target.

    Returns ``(positive_words, negative_words, stats)`` where *stats* reports raw NRC
    counts and how many words survived corpus filtering per class.
    """
    raw_pos, raw_neg = load_nrc_sentiment_words(dataset_name=dataset_name)
    nrc_positive_raw = len(raw_pos)
    nrc_negative_raw = len(raw_neg)
    raw_pos, raw_neg, excluded_polarity = exclude_ambiguous_nrc_polarity_words(
        raw_pos,
        raw_neg,
    )
    filter_fn = (
        filter_lexicon_adjectives_in_corpus
        if require_adjectives
        else filter_lexicon_to_corpus
    )
    filter_kw = {"max_words": max_words_per_class, "min_count": min_count_per_word}
    if require_adjectives:
        filter_kw.update(
            spacy_model=spacy_model,
            download_if_missing=download_if_missing,
            group_by_lemma=group_by_lemma,
        )
    positive = dedupe_targets_case_insensitive(filter_fn(raw_pos, corpus_texts, **filter_kw))
    negative = dedupe_targets_case_insensitive(filter_fn(raw_neg, corpus_texts, **filter_kw))
    stats = {
        "nrc_positive": nrc_positive_raw,
        "nrc_negative": nrc_negative_raw,
        "nrc_polarity_excluded": len(excluded_polarity),
        "corpus_positive": len(positive),
        "corpus_negative": len(negative),
        "min_count_per_word": min_count_per_word,
        "require_adjectives": require_adjectives,
    }
    return positive, negative, stats
