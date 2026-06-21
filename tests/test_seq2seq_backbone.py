"""Tests for seq2seq GRADIEND backbone selection."""

import pytest
import torch.nn as nn

from gradiend.model.core.backbone import split_backbone_vs_head_params
from gradiend.model.core.seq2seq_backbone import (
    Seq2SeqEncoderMLMBackbone,
    configure_seq2seq_gradiend_backbone,
)


class _FakeSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(10, 4)
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 10)
        self.config = type("Cfg", (), {"is_encoder_decoder": True})()


def test_encoder_mlm_backbone_excludes_decoder_weights():
    model = _FakeSeq2Seq()
    configure_seq2seq_gradiend_backbone(model, mode="seq2seq_encoder_mlm")
    core, excluded = split_backbone_vs_head_params(model)
    assert any(name.startswith("encoder.") or name == "encoder.weight" for name in core)
    assert any(name.startswith("lm_head.") or name == "lm_head.weight" for name in core)
    assert not any(name.startswith("decoder.") for name in core)
    assert any(e["name"].startswith("decoder.") for e in excluded)


def test_seq2seq_encoder_mlm_backbone_view_covers_expected_modules():
    model = _FakeSeq2Seq()
    view = Seq2SeqEncoderMLMBackbone(model)
    names = {n for n, _ in view.named_parameters()}
    assert any(n.startswith("encoder") for n in names)
    assert any(n.startswith("lm_head") for n in names)
    assert not any(n.startswith("decoder") for n in names)
