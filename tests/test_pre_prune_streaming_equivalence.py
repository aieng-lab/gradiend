"""Compare streaming vs classic pre-prune selection on identical inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradiend.model import ParamMappedGradiendModel
from gradiend.model.core.backbone import build_gradiend_from_base_model, split_backbone_vs_head_params
from gradiend.trainer.core.pruning import (
    PrePruneConfig,
    _gradient_to_vector,
    _pre_prune_streaming_topk,
    _resolve_pre_prune_use_streaming,
    _stratified_indices,
)
from gradiend.trainer.core.config import (
    alternative_computation_required_keywords,
    factual_computation_required_keywords,
)


class _TinyLm(nn.Module):
    """Minimal LM-like module: cross-entropy loss from input_ids + labels."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        logits = self.head(self.embed(input_ids))
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


class _TinyModelWithGradiend:
  """Minimal ModelWithGradiend surface for pre-prune tests."""

  def __init__(self, base: _TinyLm, gradiend: ParamMappedGradiendModel) -> None:
      self.base_model = base
      self.gradiend = gradiend
      self._gradient_creator = self

  def _get_base_forward_model(self):
      return self.base_model

  def _place_inputs_for_base_forward(self, batch):
      if isinstance(batch, dict):
          return {k: v.unsqueeze(0) if torch.is_tensor(v) and v.ndim == 1 else v for k, v in batch.items()}
      return batch

  def _zero_base_grad(self, *, set_to_none: bool = True) -> None:
      self.base_model.zero_grad(set_to_none=set_to_none)

  def _backward_through_base_model(self, loss: torch.Tensor) -> None:
      loss.backward()

  def forward(self, inputs, target_device=None, **kwargs):
      inputs = self._place_inputs_for_base_forward(inputs)
      if isinstance(inputs, dict):
          inputs = {
              k: (
                  v.unsqueeze(0)
                  if torch.is_tensor(v) and v.ndim == 1
                  else (v.squeeze(dim=1) if torch.is_tensor(v) and v.ndim == 3 and v.shape[1] == 1 else v)
              )
              for k, v in inputs.items()
          }
      outputs = self.base_model(**inputs)
      self._zero_base_grad(set_to_none=True)
      self._backward_through_base_model(outputs.loss)
      device = target_device or torch.device("cpu")
      grads = self.gradiend.extract_gradients(self.base_model, target_device=device)
      self._zero_base_grad(set_to_none=True)
      return grads

  def prune_gradiend(self, **kwargs):
      inplace = kwargs.pop("inplace", False)
      return_mask = kwargs.pop("return_mask", False)
      pruned_g, mask = self.gradiend.prune(**kwargs, inplace=inplace, return_mask=True)
      other = _TinyModelWithGradiend(self.base_model, pruned_g)
      if return_mask:
          return other, mask
      return other


def _build_tiny_model_with_gradiend() -> _TinyModelWithGradiend:
    base = _TinyLm()
    core, _ = split_backbone_vs_head_params(base)
    param_map = {
        name: {"shape": tuple(p.shape), "repr": "all"}
        for name, p in core.items()
    }
    input_dim = sum(int(torch.tensor(spec["shape"]).prod().item()) for spec in param_map.values())
    gradiend = ParamMappedGradiendModel(
        input_dim=input_dim,
        latent_dim=1,
        param_map=param_map,
        lazy_init=False,
    )
    return _TinyModelWithGradiend(base, gradiend)


def _make_item(factual_token: int, alternative_token: int) -> dict:
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    factual_labels = input_ids.clone()
    factual_labels[-1] = factual_token
    alternative_labels = input_ids.clone()
    alternative_labels[-1] = alternative_token
    return {
        "factual": {"input_ids": input_ids, "labels": factual_labels},
        "alternative": {"input_ids": input_ids, "labels": alternative_labels},
        "feature_class_id": factual_token % 2,
    }


class _DictDataset:
    def __init__(self, n: int = 16) -> None:
        self._items = [_make_item(i % 8, (i + 3) % 8) for i in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def _classic_importance_and_keep_idx(
    model,
    data,
    indices: list[int],
    *,
    source: str,
    topk: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the non-streaming pre_prune loop (gradient_creator path)."""
    requires_factual = source in factual_computation_required_keywords
    requires_alternative = source in alternative_computation_required_keywords
    running_sum = None
    count = 0
    for idx in indices:
        item = data[idx]
        fact_g = model.forward(item["factual"]) if requires_factual else None
        alt_g = model.forward(item["alternative"]) if requires_alternative else None
        if source == "factual":
            g = fact_g
        elif source == "alternative":
            g = alt_g
        else:
            g = fact_g - alt_g
        vec = _gradient_to_vector(g, model.gradiend)
        if running_sum is None:
            running_sum = torch.zeros_like(vec)
        running_sum.add_(vec)
        count += 1
    importance = (running_sum / count).abs()
    k = max(1, int(torch.ceil(torch.tensor(float(topk) * importance.numel())).item()))
    _, keep_idx = torch.topk(importance, k=k, largest=True, sorted=True)
    return importance, keep_idx.cpu().long()


class _MaskTokenizerStub:
    name_or_path = "tiny-local-tokenizer"
    mask_token = "[MASK]"
    mask_token_id = 4
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = None


def _make_mlm_item(factual_token: int, alternative_token: int, *, feature_class_id: int) -> dict:
    input_ids = torch.tensor([2, 5, 4, 6, 3], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    factual_labels = torch.full_like(input_ids, -100)
    factual_labels[2] = factual_token
    alternative_labels = torch.full_like(input_ids, -100)
    alternative_labels[2] = alternative_token
    return {
        "factual": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": factual_labels,
        },
        "alternative": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": alternative_labels,
        },
        "feature_class_id": feature_class_id,
    }


class _TinyBertDataset:
    def __init__(self) -> None:
        self._items = [
            _make_mlm_item(7, 11, feature_class_id=0),
            _make_mlm_item(8, 12, feature_class_id=1),
            _make_mlm_item(9, 13, feature_class_id=0),
            _make_mlm_item(10, 14, feature_class_id=1),
            _make_mlm_item(11, 7, feature_class_id=0),
            _make_mlm_item(12, 8, feature_class_id=1),
            _make_mlm_item(13, 9, feature_class_id=0),
            _make_mlm_item(14, 10, feature_class_id=1),
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def _build_tiny_text_prediction_model():
    transformers = pytest.importorskip("transformers")
    from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend

    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=24,
        max_position_embeddings=16,
        type_vocab_size=2,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    base = transformers.BertForMaskedLM(config)
    base.name_or_path = "tiny-local-bert"
    gradiend = build_gradiend_from_base_model(
        base,
        "tiny-local-bert",
        latent_dim=1,
        lazy_init=False,
    )
    return TextPredictionModelWithGradiend(
        base,
        gradiend,
        _MaskTokenizerStub(),
        source="alternative",
        target="diff",
    )


@pytest.mark.parametrize("source", ["factual", "alternative", "diff"])
def test_streaming_pre_prune_matches_classic_importance_topk(source: str, set_seed) -> None:
    """Streaming hook pre-prune should pick the same dims as the classic mean-|grad| path."""
    set_seed(0)
    model = _build_tiny_model_with_gradiend()
    data = _DictDataset()
    # Keep k below the number of nonzero diff dimensions in this tiny fixture.
    # Above that boundary, classic and streaming can only disagree on zero-score
    # tie winners, which is not meaningful pre-prune behavior.
    config = PrePruneConfig(n_samples=8, topk=0.05, source=source, seed=0)
    indices = _stratified_indices(
        data,
        config.n_samples,
        config.feature_class_key,
        target_feature_class_ids=[0, 1],
        seed=config.seed,
    )

    classic_importance, classic_keep = _classic_importance_and_keep_idx(
        model,
        data,
        indices,
        source=source,
        topk=config.topk,
    )

    streaming_model = _build_tiny_model_with_gradiend()
    streaming_model.base_model.load_state_dict(model.base_model.state_dict())

    _, stream_keep = _pre_prune_streaming_topk(
        streaming_model,
        data,
        config,
        indices,
        inplace=True,
        return_mask=False,
        return_keep_idx=True,
    )

    assert classic_keep.numel() == stream_keep.numel()
    assert set(classic_keep.tolist()) == set(stream_keep.tolist()), (
        f"keep_idx mismatch for source={source!r}: "
        f"classic-only={sorted(set(classic_keep.tolist()) - set(stream_keep.tolist()))[:10]}, "
        f"streaming-only={sorted(set(stream_keep.tolist()) - set(classic_keep.tolist()))[:10]}"
    )

    # Importance ranking tie-breaking: same selected set implies same top-k mass ordering up to ties.
    classic_scores = classic_importance[classic_keep]
    stream_scores = classic_importance[stream_keep]
    torch.testing.assert_close(
        torch.sort(classic_scores, descending=True).values,
        torch.sort(stream_scores, descending=True).values,
    )


def test_resolve_pre_prune_use_streaming_flag_and_env(monkeypatch) -> None:
    from dataclasses import replace

    cfg = PrePruneConfig(n_samples=4, topk=0.5, source="diff")
    eligible = dict(can_stream_inputs=True, can_stream_model=True)

    assert _resolve_pre_prune_use_streaming(cfg, **eligible) is True

    classic_cfg = replace(cfg, use_streaming=False)
    assert _resolve_pre_prune_use_streaming(classic_cfg, **eligible) is False

    streaming_cfg = replace(cfg, use_streaming=True)
    assert _resolve_pre_prune_use_streaming(streaming_cfg, **eligible) is True

    monkeypatch.setenv("GRADIEND_PREPRUNE_STREAMING", "classic")
    assert _resolve_pre_prune_use_streaming(streaming_cfg, **eligible) is False

    monkeypatch.setenv("GRADIEND_PREPRUNE_STREAMING", "streaming")
    assert _resolve_pre_prune_use_streaming(classic_cfg, **eligible) is True


@pytest.mark.slow(reason="Exercises the real TextPredictionModelWithGradiend + HF BERT wrapper path.")
@pytest.mark.parametrize("source", ["factual", "alternative", "diff"])
def test_streaming_pre_prune_matches_classic_for_text_prediction_bert_wrapper(source: str, set_seed) -> None:
    """
    Diagnostic regression test for the multilingual demo path.

    This uses an actual HF BertForMaskedLM plus TextPredictionModelWithGradiend,
    so it exercises backbone/head parameter selection, wrapper parameter names,
    requires_grad syncing, and TextPredictionModelWithGradiend.forward(). The tiny
    model is built from config only; no network or pretrained weights are needed.
    """
    set_seed(0)
    model = _build_tiny_text_prediction_model()
    data = _TinyBertDataset()
    config = PrePruneConfig(n_samples=6, topk=0.20, source=source, seed=0)
    indices = _stratified_indices(
        data,
        config.n_samples,
        config.feature_class_key,
        target_feature_class_ids=[0, 1],
        seed=config.seed,
    )

    classic_importance, classic_keep = _classic_importance_and_keep_idx(
        model,
        data,
        indices,
        source=source,
        topk=config.topk,
    )

    streaming_model = _build_tiny_text_prediction_model()
    streaming_model.base_model.load_state_dict(model.base_model.state_dict())

    _, stream_keep = _pre_prune_streaming_topk(
        streaming_model,
        data,
        config,
        indices,
        inplace=True,
        return_mask=False,
        return_keep_idx=True,
    )

    assert classic_keep.numel() == stream_keep.numel()
    assert set(classic_keep.tolist()) == set(stream_keep.tolist()), (
        f"keep_idx mismatch for source={source!r}: "
        f"classic-only={sorted(set(classic_keep.tolist()) - set(stream_keep.tolist()))[:20]}, "
        f"streaming-only={sorted(set(stream_keep.tolist()) - set(classic_keep.tolist()))[:20]}"
    )

    classic_scores = classic_importance[classic_keep]
    stream_scores = classic_importance[stream_keep]
    torch.testing.assert_close(
        torch.sort(classic_scores, descending=True).values,
        torch.sort(stream_scores, descending=True).values,
    )
