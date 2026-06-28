import pandas as pd
import pytest

import gradiend.comparison.feature_cross_encoding as feature_cross_encoding_module
from gradiend.comparison.feature_cross_encoding import (
    collect_unified_test_rows_by_feature_class,
    compute_gradiend_feature_cross_encoding_matrix,
    _mean_encoded_by_transition_from_summary,
    _mean_encoded_for_feature_class,
)
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_TRANSITION,
    transition_id,
)


def _unified_row(
    *,
    masked: str,
    factual: str,
    alternative: str,
    factual_class: str,
    alternative_class: str,
    split: str = "test",
) -> dict:
    return {
        UNIFIED_MASKED: masked,
        UNIFIED_FACTUAL: factual,
        UNIFIED_ALTERNATIVE: alternative,
        UNIFIED_FACTUAL_CLASS: factual_class,
        UNIFIED_ALTERNATIVE_CLASS: alternative_class,
        UNIFIED_TRANSITION: transition_id(factual_class, alternative_class),
        UNIFIED_SPLIT: split,
    }


class _Trainer:
    def __init__(self, trainer_id: str, target_classes, rows, experiment_dir=None):
        self.run_id = trainer_id
        self.target_classes = list(target_classes)
        self.combined_data = pd.DataFrame(rows)
        self.experiment_dir = experiment_dir

    @property
    def model_path(self):
        return str(self.experiment_dir or "/mock/model")

    def _ensure_data(self, materialize_all_class_transitions=False):
        return None

    def _encoder_cache_path(self, model_path, **encoder_kwargs):
        from gradiend.util.paths import resolve_encoder_analysis_path

        if not self.experiment_dir:
            return None
        key_kwargs = {}
        split = encoder_kwargs.get("split")
        if split is not None:
            key_kwargs["split"] = split
        max_size = encoder_kwargs.get("max_size")
        if max_size is not None:
            key_kwargs["max_size"] = max_size
        return resolve_encoder_analysis_path(self.experiment_dir, None, **key_kwargs)

    def create_gradient_training_dataset(self, pairs, model):
        return pairs

    def evaluate_encoder(self, **kwargs):
        raise AssertionError("should not be called")


class _TinyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 103
    pad_token_id = 0
    model_max_length = 32

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=True,
        max_length=None,
        padding=None,
    ):
        if return_tensors == "pt":
            import torch

            ids = [101, self.mask_token_id if self.mask_token in str(text) else 200, 102]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": [201]}

    def convert_tokens_to_ids(self, token):
        return self.mask_token_id if token == self.mask_token else 201


class _TextProbeTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_dataset_calls = []

    def create_gradient_training_dataset(self, pairs, model):
        self.gradient_dataset_calls.append((pairs, model))
        return [{"source": "gradient", "label": pairs[0]["label"], "factual_id": pairs[0]["factual_id"]}]


class _TextProbeModel:
    tokenizer = _TinyTokenizer()
    is_decoder_only_model = False
    is_seq2seq_model = False


def test_collect_unified_test_rows_by_feature_class_merges_across_trainers():
    trainers = {
        "a": _Trainer(
            "a",
            ["F", "M"],
            [
                _unified_row(
                    masked="[MASK] she",
                    factual="she",
                    alternative="he",
                    factual_class="F",
                    alternative_class="M",
                ),
            ],
        ),
        "b": _Trainer(
            "b",
            ["positive", "negative"],
            [
                _unified_row(
                    masked="[MASK] good",
                    factual="good",
                    alternative="bad",
                    factual_class="positive",
                    alternative_class="negative",
                ),
                _unified_row(
                    masked="[MASK] she",
                    factual="she",
                    alternative="he",
                    factual_class="F",
                    alternative_class="M",
                ),
            ],
        ),
    }
    by_class = collect_unified_test_rows_by_feature_class(trainers, split="test")
    assert set(by_class) == {"F", "positive"}
    assert len(by_class["F"]) == 1
    assert len(by_class["positive"]) == 1


def test_cross_encoding_probe_pairs_are_delegated_to_trainer_before_encoding(monkeypatch):
    trainer = _TextProbeTrainer(
        "gender",
        ["F", "M"],
        [],
    )
    class_df = pd.DataFrame([
        _unified_row(
            masked="[MASK] runs",
            factual="she",
            alternative="he",
            factual_class="F",
            alternative_class="M",
        )
    ])

    def _capture_encoded_rows(model, dataset):
        item = dataset[0]
        assert item["source"] == "gradient"
        assert item["factual_id"] == "F"
        return [{"encoded": 0.75}]

    monkeypatch.setattr(
        feature_cross_encoding_module,
        "encode_dataset_to_rows",
        _capture_encoded_rows,
    )

    model = _TextProbeModel()
    assert _mean_encoded_for_feature_class(
        trainer,
        model,
        class_df,
    ) == pytest.approx(0.75)
    pairs, captured_model = trainer.gradient_dataset_calls[0]
    assert captured_model is model


def test_compute_gradiend_feature_cross_encoding_matrix_is_dense(monkeypatch):
    trainers = {
        "gender": _Trainer(
            "gender",
            ["F", "M"],
            [
                _unified_row(
                    masked="[MASK] she",
                    factual="she",
                    alternative="he",
                    factual_class="F",
                    alternative_class="M",
                ),
                _unified_row(
                    masked="[MASK] he",
                    factual="he",
                    alternative="she",
                    factual_class="M",
                    alternative_class="F",
                ),
            ],
        ),
        "sentiment": _Trainer(
            "sentiment",
            ["positive", "negative"],
            [
                _unified_row(
                    masked="[MASK] good",
                    factual="good",
                    alternative="bad",
                    factual_class="positive",
                    alternative_class="negative",
                ),
                _unified_row(
                    masked="[MASK] bad",
                    factual="bad",
                    alternative="good",
                    factual_class="negative",
                    alternative_class="positive",
                ),
            ],
        ),
    }
    encoded_by_trainer_class = {
        ("gender", "F"): 0.8,
        ("gender", "M"): -0.8,
        ("gender", "positive"): 0.1,
        ("gender", "negative"): -0.1,
        ("sentiment", "F"): 0.2,
        ("sentiment", "M"): -0.2,
        ("sentiment", "positive"): 0.9,
        ("sentiment", "negative"): -0.9,
    }

    def _fake_mean(trainer, model, class_df, *, max_size=None):
        factual_class = str(class_df.iloc[0][UNIFIED_FACTUAL_CLASS])
        trainer_id = trainer.run_id
        return encoded_by_trainer_class[(trainer_id, factual_class)]

    monkeypatch.setattr(
        feature_cross_encoding_module,
        "_load_eval_model_for_trainer",
        lambda trainer, load_directory=None: object(),
    )
    monkeypatch.setattr(
        feature_cross_encoding_module,
        "_mean_encoded_for_feature_class",
        _fake_mean,
    )

    result = compute_gradiend_feature_cross_encoding_matrix(
        trainers,
        ["F", "M", "positive", "negative"],
        trainer_order=["gender", "sentiment"],
    )

    assert result["model_ids"] == ["gender", "sentiment"]
    assert result["column_ids"] == ["F", "M", "positive", "negative"]
    assert result["matrix"][0] == pytest.approx([0.8, -0.8, 0.1, -0.1])
    assert result["matrix"][1] == pytest.approx([0.2, -0.2, 0.9, -0.9])
    assert all(count > 0 for row in result["n_matrix"] for count in row)


def test_compute_gradiend_transition_cross_encoding_matrix_from_encoder_summary():
    from gradiend.comparison.feature_cross_encoding import (
        compute_gradiend_transition_cross_encoding_matrix,
    )

    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "encoded": 1.0,
                        "type": "training",
                    },
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "encoded": 3.0,
                        "type": "training",
                    },
                    {
                        "factual_id": "C",
                        "counterfactual_id": "D",
                        "transition_id": "C->D",
                        "encoded": 9.0,
                        "type": "training",
                    },
                ]
            )
        }
    }
    trainers = {"ab": object()}
    result = compute_gradiend_transition_cross_encoding_matrix(
        trainers,
        trainer_order=["ab"],
        transition_order=[transition_id("A", "B"), transition_id("C", "D")],
        encoder_summary=encoder_summary,
    )
    assert result["measure"] == "gradiend_transition_cross_encoding_mean"
    assert result["matrix"][0] == pytest.approx([2.0, 9.0])
    assert result["n_matrix"][0] == [2, 1]


def test_compute_gradiend_transition_cross_encoding_matrix_reports_seed_std():
    from gradiend.comparison.feature_cross_encoding import (
        compute_gradiend_transition_cross_encoding_matrix,
    )

    row = {
        "factual_id": "A",
        "counterfactual_id": "B",
        "transition_id": "A->B",
        "encoded": 1.0,
        "type": "training",
    }
    encoder_summary = {
        "ab": {
            "encoder_dfs": [
                pd.DataFrame([{**row, "encoded": 1.0}]),
                pd.DataFrame([{**row, "encoded": 3.0}]),
            ]
        }
    }
    result = compute_gradiend_transition_cross_encoding_matrix(
        {"ab": object()},
        trainer_order=["ab"],
        transition_order=[transition_id("A", "B")],
        encoder_summary=encoder_summary,
    )
    assert result["multi_seed"] is True
    assert result["matrix"][0] == pytest.approx([2.0])
    assert result["cell_stats"][0][0]["std"] == pytest.approx(1.0)


def test_gradiend_transition_matrix_matches_unicode_column_order_with_ascii_encoder_rows():
    from gradiend.comparison.feature_cross_encoding import (
        compute_gradiend_transition_cross_encoding_matrix,
    )

    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "encoded": 4.0,
                        "type": "training",
                    }
                ]
            )
        }
    }
    result = compute_gradiend_transition_cross_encoding_matrix(
        {"ab": object()},
        trainer_order=["ab"],
        transition_order=[transition_id("A", "B")],
        encoder_summary=encoder_summary,
    )
    assert result["matrix"][0] == pytest.approx([4.0])
    assert result["n_matrix"][0] == [1]


def test_mean_encoded_by_transition_from_summary_averages_seed_dfs():
    payload = {
        "encoder_dfs": [
            pd.DataFrame(
                [
                    {
                        "masked": "[MASK] x",
                        "source_id": "A",
                        "target_id": "B",
                        "encoded": 1.0,
                        "type": "training",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "masked": "[MASK] x",
                        "source_id": "A",
                        "target_id": "B",
                        "encoded": 0.2,
                        "type": "training",
                    }
                ]
            ),
        ]
    }
    result = _mean_encoded_by_transition_from_summary(payload)
    transition = transition_id("A", "B")
    assert transition in result
    assert result[transition][0] == pytest.approx(0.6)
    assert result[transition][1] == 1


def test_build_cross_task_encoder_summary_uses_disk_cache(tmp_path, monkeypatch):
    rows = [
        _unified_row(
            masked="[MASK] der",
            factual="der",
            alternative="dem",
            factual_class="masc_nom",
            alternative_class="masc_dat",
        ),
        _unified_row(
            masked="[MASK] dem",
            factual="dem",
            alternative="der",
            factual_class="masc_dat",
            alternative_class="masc_nom",
        ),
    ]
    trainer = _Trainer("gender_de_masc_nom_masc_dat", ("masc_nom", "masc_dat"), rows)
    trainer.training_args = type("Args", (), {"use_cache": True})()
    monkeypatch.setattr(trainer, "experiment_dir", str(tmp_path / "gender_de_masc_nom_masc_dat"))

    encode_calls = {"count": 0}

    def _fake_encode(model, grad_ds):
        encode_calls["count"] += 1
        return [
            {
                "encoded": 0.5,
                "label": 1.0,
                "factual_id": "masc_nom",
                "alternative_id": "masc_dat",
                "transition_id": "masc_nom->masc_dat",
                "feature_class_id": "masc_nom->masc_dat",
            },
            {
                "encoded": -0.5,
                "label": -1.0,
                "factual_id": "masc_dat",
                "alternative_id": "masc_nom",
                "transition_id": "masc_dat->masc_nom",
                "feature_class_id": "masc_dat->masc_nom",
            },
        ]

    monkeypatch.setattr(feature_cross_encoding_module, "_load_eval_model_for_trainer", lambda trainer, **kwargs: object())
    monkeypatch.setattr(feature_cross_encoding_module, "_gradient_dataset_for_unified_df", lambda *args, **kwargs: [])
    monkeypatch.setattr(feature_cross_encoding_module, "encode_dataset_to_rows", _fake_encode)

    trainers = {"gender_de_masc_nom_masc_dat": trainer}
    first = feature_cross_encoding_module.build_cross_task_encoder_summary(
        trainers,
        ["masc_nom", "masc_dat"],
        split="test",
        max_size=None,
    )
    second = feature_cross_encoding_module.build_cross_task_encoder_summary(
        trainers,
        ["masc_nom", "masc_dat"],
        split="test",
        max_size=None,
    )

    assert encode_calls["count"] == 1
    assert len(first["gender_de_masc_nom_masc_dat"]["encoder_df"]) == 2
    assert len(second["gender_de_masc_nom_masc_dat"]["encoder_df"]) == 2
    cache_files = list(tmp_path.rglob("encoded_values*.csv"))
    assert len(cache_files) == 1
    assert "cross_task" not in cache_files[0].name


def test_supplement_unified_test_rows_adds_missing_factual_classes():
    base_rows = [
        _unified_row(
            masked="[MASK] he",
            factual="he",
            alternative="she",
            factual_class="3SG",
            alternative_class="3PL",
        ),
    ]
    trainers = {
        "pronoun_3SG_3PL": _Trainer("pronoun_3SG_3PL", ("3SG", "3PL"), base_rows),
    }
    probe_rows = [
        _unified_row(
            masked="[MASK] we",
            factual="we",
            alternative="I",
            factual_class="1PL",
            alternative_class="1SG",
        ),
    ]
    probe = _Trainer("pronoun_probe_pool", ("1SG", "1PL"), probe_rows)

    merged = feature_cross_encoding_module.collect_unified_test_rows(
        trainers,
        split="test",
        probe_trainers={"pronoun_probe_pool": probe},
        required_factual_classes=["1PL", "3SG"],
    )
    assert set(merged["factual_class"].astype(str)) == {"1PL", "3SG"}


def test_identity_encoder_cache_is_rejected_for_cross_task():
    cached = pd.DataFrame(
        [
            {
                "encoded": 0.5,
                "label": 1.0,
                "source_id": "1SG",
                "target_id": "1SG",
                "type": "training",
            }
        ]
    )
    assert not feature_cross_encoding_module._encoder_df_has_directed_transitions(cached)
    assert not feature_cross_encoding_module._cached_cross_task_encoder_df_matches(
        cached,
        object(),
        {"1SG->1PL"},
    )


def test_build_cross_task_encoder_summary_reuses_cached_probes_incrementally(
    tmp_path, monkeypatch
):
    rows = [
        _unified_row(
            masked="[MASK] der",
            factual="der",
            alternative="dem",
            factual_class="masc_nom",
            alternative_class="masc_dat",
        ),
        _unified_row(
            masked="[MASK] she",
            factual="she",
            alternative="he",
            factual_class="F",
            alternative_class="M",
        ),
    ]
    trainer = _Trainer("gender_de_masc_nom_masc_dat", ("masc_nom", "masc_dat"), rows)
    trainer.training_args = type("Args", (), {"use_cache": True})()
    cache_dir = tmp_path / "gender_de_masc_nom_masc_dat"
    cache_dir.mkdir(parents=True)
    trainer.experiment_dir = str(cache_dir)
    cache_path = trainer._encoder_cache_path("", split="test", max_size=None)
    pd.DataFrame(
        [
            {
                "encoded": 0.25,
                "label": 1.0,
                "masked": "[MASK] der",
                "factual_token": "der",
                "alternative_token": "dem",
                "factual_id": "masc_nom",
                "alternative_id": "masc_dat",
                "type": "training",
            }
        ]
    ).to_csv(cache_path, index=False)

    encode_calls = {"count": 0}

    def _fake_encode(model, grad_ds):
        encode_calls["count"] += 1
        return [
            {
                "encoded": -0.25,
                "label": 0.0,
                "masked": "[MASK] she",
                "factual_token": "she",
                "alternative_token": "he",
                "factual_id": "F",
                "alternative_id": "M",
                "type": "training",
            }
        ]

    monkeypatch.setattr(feature_cross_encoding_module, "_load_eval_model_for_trainer", lambda trainer, **kwargs: object())
    monkeypatch.setattr(feature_cross_encoding_module, "_gradient_dataset_for_unified_df", lambda *args, **kwargs: [])
    monkeypatch.setattr(feature_cross_encoding_module, "encode_dataset_to_rows", _fake_encode)

    result = feature_cross_encoding_module.build_cross_task_encoder_summary(
        {"gender_de_masc_nom_masc_dat": trainer},
        ["masc_nom", "masc_dat", "F", "M"],
        split="test",
        max_size=None,
    )

    assert encode_calls["count"] == 1
    encoder_df = result["gender_de_masc_nom_masc_dat"]["encoder_df"]
    assert len(encoder_df) == 2
    keys = feature_cross_encoding_module._encoder_probe_keys(encoder_df)
    assert ("[MASK] der", "der", "dem") in keys
    assert ("[MASK] she", "she", "he") in keys
