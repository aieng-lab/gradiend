import os
import shutil
from typing import Any, Dict

import pandas as pd
import pytest
import torch
from transformers import GPT2Config, PretrainedConfig

from gradiend.model.core.backbone import split_backbone_vs_head_params
from gradiend.trainer.core.pruning import _streaming_topk_from_accumulators
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.decoder_only_mlm import DataFrameMLMDataset, DecoderModelWithMLMHead
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
from gradiend.util.paths import has_saved_decoder_mlm_head


def test_decoder_only_mlm_wrapper_excludes_decoder_lm_head_from_default_params():
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=20,
        tie_word_embeddings=False,
    )
    model = DecoderModelWithMLMHead(config, target_token_ids=[1, 2])

    decoder_names = dict(model.decoder.named_parameters()).keys()
    wrapper_names = dict(model.named_parameters()).keys()

    assert "lm_head.weight" in decoder_names
    assert "lm_head.weight" not in wrapper_names
    assert "transformer.wte.weight" in wrapper_names
    assert all(not name.startswith("classifier") for name in wrapper_names)

    core, excluded = split_backbone_vs_head_params(model)
    assert "lm_head.weight" not in core
    assert "transformer.wte.weight" in core
    assert all(not name.startswith("classifier") for name in core)
    assert [item["name"] for item in excluded] == []


def test_decoder_only_mlm_classifier_aligns_to_hidden_states():
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=20,
        tie_word_embeddings=False,
    )
    model = DecoderModelWithMLMHead(config, target_token_ids=[1, 2])
    hidden_states = torch.zeros(1, 1, config.n_embd, dtype=torch.float16)

    model._align_classifier_to_hidden_states(hidden_states)

    assert model.classifier.weight.device == hidden_states.device
    assert model.classifier.weight.dtype == hidden_states.dtype


def test_decoder_only_mlm_uses_embedding_width_when_config_has_no_hidden_size():
    class ConfigWithoutHiddenSize(PretrainedConfig):
        model_type = "hiddenless-test"

        def __init__(self, vocab_size=11, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size

    class DecoderWithoutHiddenSize(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = ConfigWithoutHiddenSize()
            self.embed_tokens = torch.nn.Embedding(11, 7)
            self.lm_head = torch.nn.Linear(7, 11, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    model = DecoderModelWithMLMHead(
        ConfigWithoutHiddenSize(),
        target_token_ids=[1, 2, 3],
        decoder=DecoderWithoutHiddenSize(),
    )

    assert model.classifier.in_features == 7
    assert model.classifier.out_features == 3


def test_decoder_only_mlm_dataset_replaces_literal_mask_with_tokenizer_mask():
    class Tokenizer:
        mask_token = "<mask>"
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            ids = [5 if token == self.mask_token else 1 for token in str(text).split()]
            if kwargs.get("return_tensors") == "pt":
                return type(
                    "Encoded",
                    (),
                    {
                        "input_ids": torch.tensor([ids], dtype=torch.long),
                        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                    },
                )()
            return {"input_ids": ids}

    df = pd.DataFrame([{"masked": "Der [MASK] ist da", "label": "Mann"}])
    dataset = DataFrameMLMDataset(Tokenizer(), df, label_class_map={"Mann": 0})

    input_ids, _attention_mask, _label_ids = dataset[0]

    assert 5 in input_ids.tolist()


def test_decoder_only_mlm_head_uses_label_class_indices():
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=20,
        tie_word_embeddings=False,
    )
    model = DecoderModelWithMLMHead(config, target_labels=["AFRICAN", "EUROPEAN"])

    assert model.classifier.out_features == 2
    assert model.target_labels == ["AFRICAN", "EUROPEAN"]


def test_decoder_only_mlm_dataset_maps_label_string_to_class_index():
    class Tokenizer:
        mask_token = "<mask>"
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            ids = [5 if token == self.mask_token else 1 for token in str(text).split()]
            if kwargs.get("return_tensors") == "pt":
                return type(
                    "Encoded",
                    (),
                    {
                        "input_ids": torch.tensor([ids], dtype=torch.long),
                        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                    },
                )()
            return {"input_ids": ids}

    df = pd.DataFrame([{"masked": "people are [MASK]", "label": "AFRICAN"}])
    dataset = DataFrameMLMDataset(Tokenizer(), df, label_class_map={"AFRICAN": 0, "EUROPEAN": 1})

    _input_ids, _attention_mask, label_ids = dataset[0]

    assert label_ids.tolist() == [0]


def test_decoder_only_mlm_from_pretrained_filters_wrapper_kwargs(monkeypatch):
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=20,
        tie_word_embeddings=False,
    )
    captured: Dict[str, Any] = {}

    class DummyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.base_model = self

        def get_input_embeddings(self):
            return torch.nn.Embedding(20, 8)

    def spy_from_pretrained(name_or_path, *args, **kwargs):
        captured["name_or_path"] = name_or_path
        captured["kwargs"] = kwargs
        return DummyDecoder()

    import gradiend.trainer.text.prediction.decoder_only_mlm as module

    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", spy_from_pretrained)
    monkeypatch.setattr(module.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: config)

    DecoderModelWithMLMHead.from_pretrained(
        "dummy-llama",
        target_labels=["AFRICAN", "EUROPEAN"],
        trust_remote_code=True,
    )

    assert captured["kwargs"] == {"trust_remote_code": True}
    assert "target_labels" not in captured["kwargs"]


def test_decoder_only_mlm_forward_raises_when_labels_given_without_mask():
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=20,
        tie_word_embeddings=False,
    )
    config.mask_token_id = 19
    model = DecoderModelWithMLMHead(config, target_token_ids=[1, 2])

    with pytest.raises(ValueError, match="No mask token id"):
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            labels=torch.tensor([[1]]),
        )


def test_streaming_topk_reports_unused_mapped_parameter_clearly():
    class Selector:
        name = "lm_head.weight"
        num_selected = 3

    with pytest.raises(RuntimeError, match="included in the param_map but received no gradient"):
        _streaming_topk_from_accumulators({}, [Selector()], k=1, count=1)


def test_decoder_only_mlm_custom_checkpoint_device_map_loads_base_model(monkeypatch):
    base_dir = os.path.abspath("test_artifacts/decoder_only_mlm_base")
    head_dir = os.path.abspath("test_artifacts/decoder_only_mlm_head")
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.rmtree(head_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)

    try:
        config = GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            vocab_size=20,
            tie_word_embeddings=False,
        )
        base_model = DecoderModelWithMLMHead(config, target_token_ids=[1, 2]).decoder
        base_model.save_pretrained(base_dir)

        head_model = DecoderModelWithMLMHead(config, target_token_ids=[1, 2])
        head_model.save_pretrained(head_dir, base_model=base_dir)

        import gradiend.trainer.text.prediction.decoder_only_mlm as module

        calls = {}

        def spy_from_pretrained(name_or_path, *args, **kwargs):
            calls["name_or_path"] = name_or_path
            calls["kwargs"] = kwargs
            return base_model

        class Tokenizer:
            def __len__(self):
                return 20

        monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", spy_from_pretrained)
        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())

        loaded = DecoderModelWithMLMHead.from_pretrained(
            head_dir,
            device_map="auto",
            target_token_ids=[1, 2],
        )

        assert isinstance(loaded, DecoderModelWithMLMHead)
        assert calls["name_or_path"] == base_dir
        assert calls["kwargs"]["device_map"] == "auto"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
        shutil.rmtree(head_dir, ignore_errors=True)


def test_saved_decoder_mlm_head_cache_does_not_require_gradiend_training_args(tmp_path):
    head_dir = tmp_path / "decoder_mlm_head"
    head_dir.mkdir()
    (head_dir / "config.json").write_text('{"model_type": "gpt2-with-mlm-head"}')
    (head_dir / "config_mlm_head.json").write_text(
        '{"target_labels": ["A", "B"], "target_token_ids": [1, 2], "base_model": "gpt2"}'
    )
    (head_dir / "model.safetensors").write_bytes(b"placeholder")
    (head_dir / "tokenizer_config.json").write_text("{}")
    (head_dir / "tokenizer.json").write_text("{}")

    assert has_saved_decoder_mlm_head(str(head_dir))
    assert not (head_dir / "training_args.json").exists()


def test_train_decoder_only_mlm_head_uses_mlm_head_cache(monkeypatch, tmp_path):
    experiment_dir = tmp_path / "experiment"
    head_dir = experiment_dir / "decoder_mlm_head"
    head_dir.mkdir(parents=True)
    (head_dir / "config.json").write_text('{"model_type": "gpt2-with-mlm-head"}')
    (head_dir / "config_mlm_head.json").write_text(
        '{"target_labels": ["A", "B"], "target_token_ids": [1, 2], "base_model": "gpt2"}'
    )
    (head_dir / "pytorch_model.bin").write_bytes(b"placeholder")
    (head_dir / "tokenizer_config.json").write_text("{}")
    (head_dir / "tokenizer.json").write_text("{}")

    trainer = object.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(experiment_dir=str(experiment_dir), use_cache=True)
    trainer.run_id = None

    def fail_get_data(*_args, **_kwargs):
        raise AssertionError("MLM head training data should not be loaded when cache is valid.")

    monkeypatch.setattr(trainer, "get_decoder_mlm_training_data", fail_get_data)

    assert trainer.train_decoder_only_mlm_head("gpt2") == str(head_dir)


@pytest.mark.skipif(os.name == "nt", reason="symlink tests require POSIX")
def test_ensure_writable_dir_resolves_symlink_to_target(tmp_path):
    from gradiend.util.paths import ensure_writable_dir

    target = tmp_path / "shared_head"
    target.mkdir()
    link = tmp_path / "decoder_mlm_head"
    os.symlink(target, link, target_is_directory=True)

    resolved = ensure_writable_dir(str(link))

    assert resolved == str(target.resolve())
    assert os.path.isdir(resolved)


@pytest.mark.skipif(os.name == "nt", reason="symlink tests require POSIX")
def test_ensure_writable_dir_replaces_broken_symlink(tmp_path):
    from gradiend.util.paths import ensure_writable_dir

    link = tmp_path / "decoder_mlm_head"
    os.symlink(str(tmp_path / "missing"), link, target_is_directory=True)

    resolved = ensure_writable_dir(str(link))

    assert resolved == str(link.resolve())
    assert os.path.isdir(resolved)
    assert not os.path.islink(resolved)
