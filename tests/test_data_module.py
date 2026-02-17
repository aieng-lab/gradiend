"""Tests for gradiend.data module (TextPredictionDataCreator, TextFilterConfig, etc.)."""

import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# German articles often excluded for neutral data in examples
_NEUTRAL_EXCLUDE_DE_ARTICLES = ["die", "das", "ein", "eine"]

from gradiend.data.core.base_loader import resolve_base_data
from gradiend.data.text import TextPreprocessConfig, preprocess_texts
from gradiend.data.text.filter_config import TextFilterConfig
from gradiend.data.text.prediction.creator import TextPredictionDataCreator
from gradiend.data.text.prediction.filter_engine import filter_sentences, mask_sentence


class TestResolveBaseData:
    def test_list_of_strings(self):
        texts = resolve_base_data(["a", "b", "c"], max_size=2, seed=42)
        assert len(texts) == 2
        assert set(texts) <= {"a", "b", "c"}

    def test_dataframe(self):
        df = pd.DataFrame({"text": ["x", "y", "z"]})
        texts = resolve_base_data(df, text_column="text", max_size=2, seed=42)
        assert len(texts) == 2
        assert set(texts) <= {"x", "y", "z"}

    def test_shuffle_reproducible(self):
        texts = resolve_base_data(["a", "b", "c", "d"], max_size=4, seed=1)
        texts2 = resolve_base_data(["a", "b", "c", "d"], max_size=4, seed=1)
        assert texts == texts2

    def test_str_csv_path_loads_and_uses_text_column(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"text": ["first", "second", "third"]}).to_csv(path, index=False)
        texts = resolve_base_data(str(path), text_column="text", max_size=10, seed=42)
        assert len(texts) == 3
        assert "first" in texts and "second" in texts and "third" in texts

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError, match="source must be str, DataFrame, or List"):
            resolve_base_data(123)


class TestPreprocessTexts:
    def test_no_config_returns_as_is(self):
        inp = ["a", "b"]
        assert preprocess_texts(inp, None) == inp

    def test_min_max_chars(self):
        config = TextPreprocessConfig(min_chars=2, max_chars=5)
        inp = ["a", "ab", "abc", "abcd", "abcde", "abcdef"]
        out = preprocess_texts(inp, config)
        assert out == ["ab", "abc", "abcd", "abcde"]

    def test_exclude_chars(self):
        config = TextPreprocessConfig(exclude_chars="x")
        inp = ["a", "axb", "c"]
        out = preprocess_texts(inp, config)
        assert out == ["a", "c"]


class TestTextFilterConfig:
    def test_simple_targets(self):
        cfg = TextFilterConfig(targets=["der"])
        assert cfg.id == "der"
        flat = cfg.flatten_targets_with_tags()
        assert flat == [("der", None)]

    def test_single_string_targets(self):
        cfg = TextFilterConfig(targets="der")
        assert cfg.targets == ["der"]

    def test_target_singular(self):
        cfg = TextFilterConfig(target="das")
        assert cfg.targets == ["das"]

    def test_target_and_targets_raises(self):
        with pytest.raises(ValueError, match="provide either target or targets"):
            TextFilterConfig(target="das", targets=["der"])


class TestFilterEngine:
    def test_string_only_filter(self):
        cfg = TextFilterConfig(targets=["der"])
        matches = filter_sentences(["Der Mann ging.", "Die Frau kam."], cfg)
        assert len(matches) == 1
        sent, spans = matches[0]
        assert sent == "Der Mann ging."
        assert len(spans) == 1
        assert spans[0][2] == "Der"
        masked = mask_sentence(sent, spans, "[MASK]")
        assert "[MASK]" in masked
        assert "Mann" in masked

    def test_mask_sentence_multiple(self):
        sent = "der der Mann"
        spans = [(0, 3, "der"), (4, 7, "der")]
        masked = mask_sentence(sent, spans, "[MASK]")
        assert masked.count("[MASK]") == 2


class TestTextPredictionDataCreator:
    def test_generate_training_data_per_class(self):
        creator = TextPredictionDataCreator(
            base_data=["Der Mann ging.", "Die Frau kam.", "Das Kind spielte."],
            feature_targets=[
                TextFilterConfig(targets=["der"]),
                TextFilterConfig(targets=["die"]),
            ],
            seed=42,
        )
        training = creator.generate_training_data(format="per_class")
        assert "der" in training
        assert "die" in training
        assert len(training["der"]) == 1
        assert len(training["die"]) == 1
        assert "[MASK]" in training["der"]["masked"].iloc[0]
        assert training["der"]["label"].iloc[0] == "Der"

    def test_generate_training_data_minimal(self):
        creator = TextPredictionDataCreator(
            base_data=["Der Mann.", "Die Frau."],
            feature_targets=[
                TextFilterConfig(targets=["der"]),
                TextFilterConfig(targets=["die"]),
            ],
        )
        df = creator.generate_training_data(format="minimal")
        assert "masked" in df.columns
        assert "label" in df.columns
        assert "label_class" in df.columns
        assert "split" in df.columns

    def test_generate_neutral_data(self):
        creator = TextPredictionDataCreator(
            base_data=["Der Mann.", "Die Frau.", "Das Haus ist gross."],
            feature_targets=[
                TextFilterConfig(targets=["der"]),
                TextFilterConfig(targets=["die"]),
            ],
        )
        # Target words (der, die) are auto-excluded; no additional needed
        neutral = creator.generate_neutral_data(max_size=10)
        assert "text" in neutral.columns
        assert len(neutral) <= 3
        # "Das Haus ist gross." should be in neutral (no der/die)
        texts = neutral["text"].tolist()
        assert any("Das Haus" in t for t in texts)

    def test_generate_neutral_data_additional_excluded(self):
        creator = TextPredictionDataCreator(
            base_data=["Der Mann.", "Die Frau.", "Das Haus ist gross.", "Ein Baum steht."],
            feature_targets=[TextFilterConfig(targets=["der"])],
        )
        # der is auto-excluded; additional_excluded_words adds "Die", "Ein"
        neutral = creator.generate_neutral_data(
            additional_excluded_words=["die", "ein"],
            max_size=10,
        )
        texts = neutral["text"].tolist()
        assert any("Das Haus" in t for t in texts)
        assert not any("Der " in t or t.startswith("Der") for t in texts)
        assert not any("Die " in t or t.startswith("Die") for t in texts)


class TestFilterEngineWithMockedSpacy:
    """Tests for spacy-based filtering using mocked spacy (no real spacy needed)."""

    def _make_mock_token(self, text: str, pos: str = "NOUN", lemma: str = None, idx: int = 0, morph: dict = None):
        t = MagicMock()
        t.text = text
        t.lemma_ = lemma if lemma is not None else text.lower()
        t.pos_ = pos
        t.idx = idx
        t.dep_ = "ROOT"
        t.tag_ = "NN"
        m = morph or {}

        def morph_get(k):
            v = m.get(k)
            return [v] if isinstance(v, str) else (v if v is not None else [])

        t.morph = MagicMock()
        t.morph.get = morph_get
        return t

    def _make_mock_doc(self, tokens):
        doc = MagicMock()
        doc.__iter__ = lambda self: iter(tokens)
        return doc

    @patch("gradiend.data.text.prediction.filter_engine.load_spacy_model")
    def test_filter_with_spacy_tags(self, mock_spacy_load):
        """Test filter_sentences with spacy_tags using mocked nlp."""
        sent = "Der Mann ging in die Stadt."
        # Mock tokens: Der (DET, idx=0), Mann (NOUN), ging, in, die (DET, idx=15), Stadt
        tokens = [
            self._make_mock_token("Der", pos="DET", idx=0, morph={"Case": "Nom", "Gender": "Masc"}),
            self._make_mock_token("Mann", pos="NOUN", idx=4),
            self._make_mock_token("ging", pos="VERB", idx=9),
            self._make_mock_token("in", pos="ADP", idx=14),
            self._make_mock_token("die", pos="DET", idx=17, morph={"Case": "Acc", "Gender": "Fem"}),
            self._make_mock_token("Stadt", pos="NOUN", idx=21),
        ]
        # load_spacy_model() returns nlp; nlp(sent) must return a doc (iterable of tokens)
        doc = self._make_mock_doc(tokens)
        mock_spacy_load.return_value = lambda sent: doc

        cfg = TextFilterConfig(
            targets=["der"],
            spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Masc"},
        )
        matches = filter_sentences([sent], cfg, spacy_model="de_core_news_sm")
        assert len(matches) == 1
        _, spans = matches[0]
        assert len(spans) == 1
        assert spans[0][2] == "Der"

    @patch("gradiend.data.text.prediction.filter_engine.load_spacy_model")
    def test_filter_with_use_lemma(self, mock_spacy_load):
        """Test filter_sentences with use_lemma (lemma matching)."""
        sent = "Der Mann ging."
        tokens = [
            self._make_mock_token("Der", pos="DET", lemma="der", idx=0),
            self._make_mock_token("Mann", pos="NOUN", lemma="mann", idx=4),
            self._make_mock_token("ging", pos="VERB", lemma="gehen", idx=9),
        ]
        doc = self._make_mock_doc(tokens)
        mock_spacy_load.return_value = lambda s: doc

        cfg = TextFilterConfig(targets=["mann"], use_lemma=True)
        matches = filter_sentences([sent], cfg, spacy_model="de_core_news_sm")
        assert len(matches) == 1
        _, spans = matches[0]
        assert spans[0][2] == "Mann"

    @patch("gradiend.data.text.prediction.creator.load_spacy_model")
    def test_neutral_filter_with_mocked_spacy_tags(self, mock_load_spacy_model):
        """Test generate_neutral_data with excluded_spacy_tags using mocked spacy."""
        sentences = [
            "Der Mann ging.",
            "Die Frau kam.",
            "Hunde bellen laut.",
        ]

        def make_doc(text):
            tokens = []
            for m in re.finditer(r"\w+", text):
                word = m.group()
                pos = "DET" if word.lower() in ("der", "die", "das") else "NOUN"
                t = MagicMock()
                t.text = word
                t.lemma_ = word.lower()
                t.pos_ = pos
                t.morph = MagicMock(get=lambda k: [])
                tokens.append(t)
            doc = MagicMock()
            doc.__iter__ = lambda self: iter(tokens)
            return doc

        mock_nlp = MagicMock(side_effect=make_doc)
        mock_load_spacy_model.return_value = mock_nlp

        creator = TextPredictionDataCreator(
            base_data=sentences,
            feature_targets=[TextFilterConfig(targets=["der"])],
            spacy_model="de_core_news_sm",
        )
        neutral = creator.generate_neutral_data(excluded_spacy_tags={"pos": "DET"})
        # Should exclude sentences containing DET; "Hunde bellen laut." has no DET
        texts = neutral["text"].tolist()
        assert "text" in neutral.columns
        assert len(texts) >= 1
        assert any("Hunde" in t for t in texts)

    def test_neutral_additional_excluded_articles(self):
        """Test generate_neutral_data with additional_excluded_words (German articles)."""
        creator = TextPredictionDataCreator(
            base_data=["Der Mann ging.", "Hunde bellen.", "Die Katze schläft."],
            feature_targets=[TextFilterConfig(targets=["der"])],
        )
        neutral = creator.generate_neutral_data(
            additional_excluded_words=_NEUTRAL_EXCLUDE_DE_ARTICLES,
        )
        texts = neutral["text"].tolist()
        assert any("Hunde" in t for t in texts)
        assert not any("Der " in t or t.startswith("Der") for t in texts)
        assert not any("Die " in t or t.startswith("Die") for t in texts)

    def test_neutral_deduplicates_excluded_words(self):
        """Target words and additional_excluded_words are deduplicated."""
        creator = TextPredictionDataCreator(
            base_data=["Der Mann.", "Hunde bellen."],
            feature_targets=[TextFilterConfig(targets=["der"])],
        )
        neutral = creator.generate_neutral_data(
            additional_excluded_words=["der", "die"],
        )
        assert any("Hunde" in t for t in neutral["text"].tolist())
