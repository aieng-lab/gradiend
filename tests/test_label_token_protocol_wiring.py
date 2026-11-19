"""Every text dataset built by the library must state its label-token rule explicitly.

``TextTrainingDataset`` defaults to ``"no_leading_space"``; a call site that forgets the argument silently
labels under the old rule even when the trainer runs under ``"first_differing_token"``.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import gradiend
import gradiend.comparison.cross_encoding as cross_encoding_module
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
    UNIFIED_TRANSITION,
    transition_id,
)


def test_every_text_training_dataset_construction_passes_label_token_protocol():
    root = Path(gradiend.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "TextTrainingDataset":
                continue
            if not any(kw.arg == "label_token_protocol" for kw in node.keywords):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"TextTrainingDataset built without label_token_protocol: {offenders}"


def test_cross_encoding_dataset_uses_the_trainers_label_token_protocol(monkeypatch):
    captured = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cross_encoding_module, "TextTrainingDataset", _Spy)
    monkeypatch.setattr(cross_encoding_module, "is_seq2seq_model", lambda tokenizer: False)
    monkeypatch.setattr(cross_encoding_module, "_is_decoder_only", lambda tokenizer: True)

    trainer = SimpleNamespace(
        _label_token_protocol=lambda: "first_differing_token",
        create_gradient_training_dataset=lambda ds, model: ds,
    )
    df = pd.DataFrame([{
        UNIFIED_MASKED: "[MASK] x",
        UNIFIED_FACTUAL: "a",
        UNIFIED_ALTERNATIVE: "b",
        UNIFIED_FACTUAL_CLASS: "A",
        UNIFIED_ALTERNATIVE_CLASS: "B",
        UNIFIED_TRANSITION: transition_id("A", "B"),
        "label": 1.0,
        "feature_class_id": "A->B",
    }])
    cross_encoding_module._gradient_dataset_for_unified_df(trainer, SimpleNamespace(tokenizer=object()), df)
    assert captured["label_token_protocol"] == "first_differing_token"
