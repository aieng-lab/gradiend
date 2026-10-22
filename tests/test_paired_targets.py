"""Tests for PairedTextFilterConfig: 1:1 counterpart targets in TextPredictionDataCreator."""

import pandas as pd
import pytest

from gradiend import TrainingArguments
from gradiend.data import PairedTextFilterConfig, TextFilterConfig, TextPredictionDataCreator
from gradiend.data.core.unified_splitting import _explicit_pair_groups
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer

PAIRS = [("colour", "color"), ("centre", "center"), ("favour", "favor")]
SENTENCES = [
    "We love the colour of the sky today.",
    "Colour is a nice thing about this painting.",
    "The centre was closed because of the weekend.",
    "They liked the color of the sea a lot.",
    "The center of the town is lovely.",
    "Please do me a favour and call me back soon.",
    "Please do me a favor and call me back soon.",
    "Both colour and favour matter here.",  # two different pair words: no unique counterpart
    "The colour and the colour again are fine.",  # the same word twice: unique counterpart
    "Nothing special happens in this neutral sentence.",
]


def _creator(feature_targets, **kwargs):
    return TextPredictionDataCreator(
        base_data=SENTENCES * 12, feature_targets=feature_targets, min_left_context_words=0, seed=0, **kwargs
    )


def _generate(creator, **kwargs):
    return creator.generate_training_data(max_size_per_class=50, min_rows_per_class_for_split=0, **kwargs)


class TestPairedTextFilterConfig:
    def test_tuple_and_dict_forms_are_equivalent(self):
        as_tuples = PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))
        as_dict = PairedTextFilterConfig(pairs={"BE": [p[0] for p in PAIRS], "AE": [p[1] for p in PAIRS]})
        assert as_tuples.ids == as_dict.ids == ("BE", "AE")
        assert as_tuples.pairs == as_dict.pairs
        assert [(c.id, c.targets) for c in as_tuples.to_configs()] == [
            ("BE", ["colour", "centre", "favour"]),
            ("AE", ["color", "center", "favor"]),
        ]

    def test_filter_options_are_forwarded_to_every_class(self):
        group = PairedTextFilterConfig(
            pairs=PAIRS, ids=("BE", "AE"), spacy_tags={"pos": "NOUN"}, min_left_context_words=3, weight=2.0
        )
        configs = group.to_configs()
        assert all(c.spacy_tags == {"pos": "NOUN"} for c in configs)
        assert all(c.min_left_context_words == 3 and c.weight == 2.0 for c in configs)
        assert configs[0].spacy_tags is not configs[1].spacy_tags

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(pairs=PAIRS), ValueError),  # tuples need ids
            (dict(pairs={"BE": ["colour"], "AE": ["color"]}, ids=("BE", "AE")), ValueError),  # dict has its own ids
            (dict(pairs={"BE": ["colour", "centre"], "AE": ["color"]}), ValueError),  # unaligned lists
            (dict(pairs=[("colour", "color", "x")], ids=("BE", "AE")), ValueError),  # wrong tuple length
            (dict(pairs=[("colour", "Colour")], ids=("BE", "AE")), ValueError),  # same word on both sides
            (dict(pairs=PAIRS, ids=("BE",)), ValueError),  # a single class
            (dict(pairs=PAIRS, ids=("BE", "BE")), ValueError),  # duplicate ids
            (dict(pairs=[], ids=("BE", "AE")), ValueError),
            (dict(pairs=PAIRS, ids=("BE", "AE"), targets=["x"]), TypeError),  # defined by pairs
            (dict(pairs=PAIRS, ids=("BE", "AE"), min_left_context_words=-1), ValueError),  # checked by TextFilterConfig
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, error):
        with pytest.raises(error):
            PairedTextFilterConfig(**kwargs)


class TestPairedCreator:
    def test_rows_carry_the_counterpart_of_their_word(self):
        data = _generate(_creator([PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))]))
        assert set(data) == {"BE", "AE"}
        be, ae = data["BE"], data["AE"]
        assert {"BE", "AE", "masked", "split"} <= set(be.columns) & set(ae.columns)
        to_ae, to_be = {a: b for a, b in PAIRS}, {b: a for a, b in PAIRS}
        assert all(to_ae[w.lower()] == c for w, c in zip(be["BE"], be["AE"]))
        assert all(to_be[w.lower()] == c for w, c in zip(ae["AE"], ae["BE"]))
        assert "Colour" in set(be["BE"])  # the factual word keeps its casing

    def test_sentences_with_several_different_pair_words_are_skipped(self):
        data = _generate(_creator([PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))]))
        assert not data["BE"]["masked"].str.contains("Both").any()
        assert data["BE"]["masked"].str.contains("The [MASK] and the [MASK] again", regex=False).any()

    def test_groups_with_the_same_ids_extend_the_same_classes(self):
        creator = _creator([
            PairedTextFilterConfig(pairs=PAIRS[:2], ids=("BE", "AE")),
            PairedTextFilterConfig(pairs=PAIRS[2:], ids=("BE", "AE")),
        ])
        assert len(creator.feature_targets) == 4
        data = _generate(creator)
        assert {"favour", "centre", "colour"} <= {w.lower() for w in data["BE"]["BE"]}

    def test_more_than_two_classes(self):
        creator = TextPredictionDataCreator(
            base_data=["I have one apple here.", "Tengo uno aqui hoy.", "Ich habe ein Buch."] * 12,
            feature_targets=[PairedTextFilterConfig(pairs=[("one", "uno", "ein")], ids=("en", "es", "de"))],
            min_left_context_words=0,
        )
        data = _generate(creator)
        assert set(data) == {"en", "es", "de"}
        assert data["es"].iloc[0]["en"] == "one" and data["es"].iloc[0]["de"] == "ein"

    def test_unified_and_minimal_keep_the_counterparts(self):
        creator = _creator([PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))])
        assert {"BE", "AE"} <= set(_generate(creator, format="unified").columns)
        assert {"BE", "AE"} <= set(_generate(creator, format="minimal").columns)

    def test_cached_data_keeps_the_counterparts(self, tmp_path):
        targets = [PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))]
        _generate(_creator(targets, output_dir=str(tmp_path), use_cache=True))
        cached = _generate(_creator(targets, output_dir=str(tmp_path), use_cache=True))
        assert cached["BE"]["AE"].notna().all() and cached["AE"]["BE"].notna().all()

    def test_split_by_target_keeps_counterparts_in_one_split(self):
        data = _generate(
            _creator([PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))]),
            split_by_target=True, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33,
        )
        splits_of_pair = {}
        for df in data.values():  # the BE column holds the British word of the pair in both classes
            for word, split in zip(df["BE"], df["split"]):
                splits_of_pair.setdefault(word.lower(), set()).add(split)
        assert len(splits_of_pair) == len(PAIRS) and all(len(s) == 1 for s in splits_of_pair.values())
        assert len({next(iter(s)) for s in splits_of_pair.values()}) == 3  # each split holds one pair

    def test_neutral_data_excludes_all_pair_words(self):
        neutral = _creator([PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE"))]).generate_neutral_data()
        assert set(neutral["text"]) == {"Nothing special happens in this neutral sentence."}


class TestFeatureTargetsValidation:
    def test_conflicting_counterparts_raise(self):
        with pytest.raises(ValueError, match="two different counterparts"):
            _creator([
                PairedTextFilterConfig(pairs=[("colour", "color")], ids=("BE", "AE")),
                PairedTextFilterConfig(pairs=[("colour", "colr")], ids=("BE", "AE")),
            ])

    def test_paired_and_plain_configs_cannot_be_mixed(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            _creator([PairedTextFilterConfig(pairs=PAIRS, ids=("BE", "AE")), TextFilterConfig(targets=["x"], id="X")])

    def test_groups_need_the_same_ids(self):
        with pytest.raises(ValueError, match="same ids"):
            _creator([
                PairedTextFilterConfig(pairs=PAIRS[:1], ids=("BE", "AE")),
                PairedTextFilterConfig(pairs=PAIRS[1:], ids=("UK", "US")),
            ])

    def test_plain_configs_are_unchanged(self):
        creator = _creator([TextFilterConfig(targets=["colour"], id="BE"), TextFilterConfig(targets=["color"], id="AE")])
        data = _generate(creator)
        assert "AE" not in data["BE"].columns  # no counterpart column without pairs


TEN_PAIRS = [
    ("colour", "color"), ("centre", "center"), ("favour", "favor"), ("honour", "honor"), ("neighbour", "neighbor"),
    ("flavour", "flavor"), ("defence", "defense"), ("organise", "organize"), ("realise", "realize"), ("analyse", "analyze"),
]


def _ten_pair_data(**generate_kwargs):
    sentences = [f"Sentence number {i} says that the {w} was really quite nice today." for i in range(40) for p in TEN_PAIRS for w in p]
    creator = TextPredictionDataCreator(
        base_data=sentences, feature_targets=[PairedTextFilterConfig(pairs=TEN_PAIRS, ids=("BE", "AE"))],
        min_left_context_words=0, seed=0,
    )
    return creator.generate_training_data(max_size_per_class=400, min_rows_per_class_for_split=0, **generate_kwargs)


class TestHeldoutSplitWithPairs:
    """Vocabulary-held-out splits must keep both counterparts together and must not redraw the alternatives."""

    def _assert_pairs_intact(self, unified):
        counterpart = {**dict(TEN_PAIRS), **{ae: be for be, ae in TEN_PAIRS}}
        assert (unified["alternative"].str.lower() == unified["factual"].str.lower().map(counterpart)).all()
        splits_of_word = unified.assign(word=unified["factual"].str.lower()).groupby("word")["split"].agg(set)
        assert all(len(splits) == 1 for splits in splits_of_word)
        assert all(splits_of_word[be] == splits_of_word[ae] for be, ae in TEN_PAIRS)
        assert set(unified["split"]) == {"train", "validation", "test"}

    def test_trainer_heldout_mode(self):
        trainer = TextPredictionTrainer(
            model="distilbert-base-cased", data=_ten_pair_data(), target_classes=["BE", "AE"], split_col="heldout",
            split_ratios=(0.6, 0.2, 0.2), args=TrainingArguments(seed=1, experiment_dir=None),
        )
        trainer._ensure_data()
        self._assert_pairs_intact(trainer.combined_data)

    @pytest.mark.parametrize("fmt", ["unified", "minimal"])
    @pytest.mark.parametrize("split_col", ["heldout", "split"])
    def test_single_table_input_keeps_explicit_alternatives(self, fmt, split_col):
        trainer = TextPredictionTrainer(
            model="distilbert-base-cased", data=_ten_pair_data(format=fmt), target_classes=["BE", "AE"],
            split_col=split_col, split_ratios=(0.6, 0.2, 0.2), args=TrainingArguments(seed=1, experiment_dir=None),
        )
        trainer._ensure_data()
        counterpart = {**dict(TEN_PAIRS), **{ae: be for be, ae in TEN_PAIRS}}
        unified = trainer.combined_data
        assert (unified["alternative"].str.lower() == unified["factual"].str.lower().map(counterpart)).all()

    def test_creator_split_is_kept_by_the_trainer(self):
        data = _ten_pair_data(split_by_target=True, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        trainer = TextPredictionTrainer(
            model="distilbert-base-cased", data=data, target_classes=["BE", "AE"], args=TrainingArguments(seed=1, experiment_dir=None),
        )
        trainer._ensure_data()
        self._assert_pairs_intact(trainer.combined_data)


class TestExplicitPairDetection:
    """The held-out resplit treats data as paired only if every row is an explicit 1:1 pair (unified columns only)."""

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(
            [{"factual_class": fc, "factual": f, "alternative_class": ac, "alternative": a} for fc, f, ac, a in rows]
        )

    def test_one_to_one_pairs_are_grouped_across_classes(self):
        rows = [("BE", be, "AE", ae) for be, ae in TEN_PAIRS] + [("AE", ae, "BE", be) for be, ae in TEN_PAIRS]
        groups = _explicit_pair_groups(self._frame(rows), [str.casefold], "factual_class")
        assert groups.notna().all()
        assert groups.nunique() == len(TEN_PAIRS)
        assert (groups.iloc[: len(TEN_PAIRS)].to_numpy() == groups.iloc[len(TEN_PAIRS):].to_numpy()).all()

    def test_pairs_without_rows_in_the_opposite_direction_are_recognised(self):
        groups = _explicit_pair_groups(self._frame([("BE", be, "AE", ae) for be, ae in TEN_PAIRS]), [str.casefold], "factual_class")
        assert groups.notna().all()

    def test_sampled_alternatives_are_not_pairs(self):
        rows = [("pos", f"p{i}", "neg", f"n{i % 3}") for i in range(9)] + [("neg", f"n{i}", "pos", f"p{i * 2}") for i in range(3)]
        assert not _explicit_pair_groups(self._frame(rows), [str.casefold], "factual_class").notna().all()

    def test_one_alternative_word_for_many_factual_words_is_not_a_pair(self):
        rows = [("white", f"w{i}", "black", "black") for i in range(5)] + [("black", f"b{i}", "white", "white") for i in range(5)]
        assert not _explicit_pair_groups(self._frame(rows), None, "factual_class").notna().all()
