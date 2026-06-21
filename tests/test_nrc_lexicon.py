"""Tests for NRC lexicon loading and corpus filtering."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

from gradiend.examples.nrc_sentiment_lexicon import (
    build_sentiment_lexicon_for_corpus,
    dedupe_targets_case_insensitive,
    exclude_ambiguous_nrc_polarity_words,
    filter_lexicon_adjectives_in_corpus,
    filter_lexicon_to_corpus,
)


def _mock_token(text: str, pos: str, *, lemma: Optional[str] = None):
    token = MagicMock()
    token.text = text
    lemma_text = text if lemma is None else lemma
    token.lemma_ = lemma_text
    token.pos_ = pos
    return token


class TestNrcLexicon:
    def test_dedupe_targets_case_insensitive(self):
        assert dedupe_targets_case_insensitive(["less", "Less", "LESS", "bad"]) == ["bad", "less"]

    def test_filter_lexicon_to_corpus_ranks_by_frequency(self):
        words = ["love", "great", "zzzznotpresent"]
        texts = ["I love this", "love love love", "great day"]
        filtered = filter_lexicon_to_corpus(words, texts, max_words=2)
        assert filtered == ["love", "great"]

    def test_filter_lexicon_to_corpus_applies_min_count(self):
        words = ["love", "great", "rare"]
        texts = ["love great", "love rare", "love"]
        filtered = filter_lexicon_to_corpus(words, texts, max_words=5, min_count=2)
        assert filtered == ["love"]

    @patch("gradiend.examples.nrc_sentiment_lexicon.load_spacy_model")
    def test_filter_lexicon_adjectives_in_corpus(self, mock_load_spacy):
        def nlp(text: str):
            if text == "good":
                return [_mock_token("good", "ADJ")]
            if text == "love":
                return [_mock_token("love", "VERB")]
            if text == "bad":
                return [_mock_token("bad", "ADJ")]
            if text == "good day":
                return [_mock_token("good", "ADJ"), _mock_token("day", "NOUN")]
            if text == "I love this":
                return [_mock_token("love", "VERB")]
            if text == "bad weather":
                return [_mock_token("bad", "ADJ"), _mock_token("weather", "NOUN")]
            return []

        mock_load_spacy.return_value = nlp

        words = ["good", "love", "bad"]
        filtered = filter_lexicon_adjectives_in_corpus(
            words,
            ["good day", "I love this", "bad weather"],
            max_words=5,
        )
        assert filtered == ["bad", "good"]

    @patch("gradiend.examples.nrc_sentiment_lexicon.load_spacy_model")
    def test_filter_lexicon_adjectives_merges_case_variants_by_lemma(self, mock_load_spacy):
        def nlp(text: str):
            if text == "Less noise":
                return [
                    _mock_token("Less", "ADJ", lemma="less"),
                    _mock_token("noise", "NOUN"),
                ]
            if text == "even less noise":
                return [
                    _mock_token("even", "ADV"),
                    _mock_token("less", "ADJ", lemma="less"),
                    _mock_token("noise", "NOUN"),
                ]
            if text == "Less":
                return [_mock_token("Less", "ADJ", lemma="less")]
            if text == "less":
                return [_mock_token("less", "ADJ", lemma="less")]
            return []

        mock_load_spacy.return_value = nlp

        filtered = filter_lexicon_adjectives_in_corpus(
            ["Less", "less"],
            ["Less noise", "even less noise", "even less noise"],
            max_words=5,
            min_count=2,
        )
        assert filtered == ["less"]

    @patch("gradiend.examples.nrc_sentiment_lexicon.load_spacy_model")
    def test_filter_lexicon_adjectives_in_corpus_applies_min_count(self, mock_load_spacy):
        def nlp(text: str):
            if text == "good":
                return [_mock_token("good", "ADJ")]
            if text == "bad":
                return [_mock_token("bad", "ADJ")]
            if text == "good day":
                return [_mock_token("good", "ADJ"), _mock_token("day", "NOUN")]
            if text == "good weather":
                return [_mock_token("good", "ADJ"), _mock_token("weather", "NOUN")]
            if text == "bad weather":
                return [_mock_token("bad", "ADJ"), _mock_token("weather", "NOUN")]
            return []

        mock_load_spacy.return_value = nlp

        filtered = filter_lexicon_adjectives_in_corpus(
            ["good", "bad"],
            ["good day", "good weather", "bad weather"],
            max_words=5,
            min_count=2,
        )
        assert filtered == ["good"]

    @patch("gradiend.examples.nrc_sentiment_lexicon.load_spacy_model")
    def test_filter_lexicon_adjectives_in_corpus_counts_sentences_not_tokens(self, mock_load_spacy):
        def nlp(text: str):
            if text == "very good and good":
                return [
                    _mock_token("very", "ADV"),
                    _mock_token("good", "ADJ"),
                    _mock_token("and", "CCONJ"),
                    _mock_token("good", "ADJ"),
                ]
            return []

        mock_load_spacy.return_value = nlp

        filtered = filter_lexicon_adjectives_in_corpus(
            ["good"],
            ["very good and good"],
            max_words=5,
            min_count=2,
        )
        assert filtered == []

    def test_exclude_ambiguous_nrc_polarity_words_drops_shared_entries(self):
        pos = ["good", "highest", "nice"]
        neg = ["bad", "highest", "awful"]
        filtered_pos, filtered_neg, excluded = exclude_ambiguous_nrc_polarity_words(pos, neg)
        assert excluded == ["highest"]
        assert filtered_pos == ["good", "nice"]
        assert filtered_neg == ["bad", "awful"]

    @patch("gradiend.examples.nrc_sentiment_lexicon.load_spacy_model")
    def test_filter_lexicon_adjectives_in_corpus_surface_forms(self, mock_load_spacy):
        def nlp(text: str):
            if text == "The highest score":
                return [
                    _mock_token("The", "DET"),
                    _mock_token("highest", "ADJ"),
                    _mock_token("score", "NOUN"),
                ]
            if text == "A high score":
                return [
                    _mock_token("A", "DET"),
                    _mock_token("high", "ADJ"),
                    _mock_token("score", "NOUN"),
                ]
            return []

        mock_load_spacy.return_value = nlp

        filtered = filter_lexicon_adjectives_in_corpus(
            ["highest", "high"],
            ["The highest score", "A high score", "The highest score"],
            max_words=5,
            min_count=2,
            group_by_lemma=False,
        )
        assert filtered == ["highest"]

    @patch("gradiend.examples.nrc_sentiment_lexicon.load_nrc_sentiment_words")
    @patch("gradiend.examples.nrc_sentiment_lexicon.filter_lexicon_adjectives_in_corpus")
    def test_build_sentiment_lexicon_uses_adjective_filter(self, mock_adj_filter, mock_load_nrc):
        mock_load_nrc.return_value = (["good"], ["bad"])
        mock_adj_filter.side_effect = lambda words, texts, **kw: list(words)

        positive, negative, stats = build_sentiment_lexicon_for_corpus(
            ["good day", "bad weather"],
            max_words_per_class=5,
            min_count_per_word=3,
        )
        assert positive == ["good"]
        assert negative == ["bad"]
        assert stats["require_adjectives"] is True
        assert stats["min_count_per_word"] == 3
        assert mock_adj_filter.call_count == 2
        assert mock_adj_filter.call_args.kwargs["min_count"] == 3
