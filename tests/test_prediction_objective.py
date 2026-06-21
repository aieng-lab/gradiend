import os

import pandas as pd
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_probability_shift_score_clm_sequence
from gradiend.trainer.text.prediction.prediction_objective import (
    DecoderMLMHeadObjective,
    resolve_prediction_objective,
)


class TinyTokenizer:
    def __init__(self):
        self.vocab = {"start": 0, "a": 1, "b": 2, "ok": 3, "bad": 4}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.mask_token = None
        self.mask_token_id = None

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        if isinstance(text, list):
            ids = [self(t, add_special_tokens=add_special_tokens)["input_ids"] for t in text]
            max_len = max(len(x) for x in ids)
            padded = [x + [0] * (max_len - len(x)) for x in ids]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in ids]),
            }
        return {"input_ids": [self.vocab[tok] for tok in str(text).split() if tok]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv[int(i)] for i in ids)


class TinyCausalModel(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
        logits[..., 1] = 5.0  # P(a | start)
        logits[..., 2] = 5.0  # P(b | start)
        for b in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1]):
                current = int(input_ids[b, pos])
                if current == 1:
                    logits[b, pos, 3] = 10.0  # a -> ok
                elif current == 2:
                    logits[b, pos, 4] = 10.0  # b -> bad
        return type("Output", (), {"logits": logits})()


def test_clm_sequence_cloze_uses_rhs_context():
    df = pd.DataFrame([
        {
            "masked": "start [MASK] ok",
            "factual": "a",
            "alternative": "b",
            "factual_id": "A",
            "alternative_id": "B",
        }
    ])

    result, per_row = compute_probability_shift_score_clm_sequence(
        TinyCausalModel(),
        TinyTokenizer(),
        df,
        targets={},
        use_row_wise=True,
        return_per_row_df=True,
    )

    assert result["A"]["A"] > 0.98
    assert result["A"]["B"] < 0.02
    assert per_row.loc[0, "P_factual"] > per_row.loc[0, "P_alternative"]


def test_decoder_mlm_head_objective_scores_original_clm_head():
    original = object()

    class Wrapper:
        def to_original_model(self):
            return original

    assert DecoderMLMHeadObjective().resolve_scoring_model(Wrapper(), None, None) is original


def test_explicit_clm_mlm_head_ensures_head_training(tmp_path):
    calls = {}

    class Trainer:
        experiment_dir = str(tmp_path)
        _training_args = TrainingArguments(
            experiment_dir=str(tmp_path),
            prediction_objective="clm_mlm_head",
            decoder_mlm_head_epochs=2,
            decoder_mlm_head_batch_size=3,
            decoder_mlm_head_max_size=4,
        )

        def train_decoder_only_mlm_head(self, model, **kwargs):
            calls["model"] = model
            calls["kwargs"] = kwargs

    trainer = Trainer()
    objective = resolve_prediction_objective(trainer)
    objective.ensure_training_resources(trainer, "base-model")

    assert calls["model"] == "base-model"
    assert calls["kwargs"]["epochs"] == 2
    assert calls["kwargs"]["batch_size"] == 3
    assert calls["kwargs"]["max_size"] == 4
    assert os.path.basename(calls["kwargs"]["output"]) == "decoder_mlm_head"
