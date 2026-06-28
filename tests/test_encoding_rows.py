"""Tests for gradient_entry_to_encoder_row / encode_dataset_to_rows."""

from gradiend.util.encoding_rows import gradient_entry_to_encoder_row


def test_gradient_entry_to_encoder_row_includes_masked_and_target_token():
    entry = {
        "label": 0,
        "factual_id": "neutral",
        "alternative_id": "neutral",
        "text": "The chef he added pepper quickly",
        "template": "The chef he added [MASK] quickly",
        "input_text": "The chef he added [MASK] quickly",
        "factual_token": "pepper",
        "alternative_token": "pepper",
    }
    row = gradient_entry_to_encoder_row(entry, encoded=0.12, input_type="factual")
    assert row["masked"] == "The chef he added [MASK] quickly"
    assert row["template"] == "The chef he added [MASK] quickly"
    assert row["factual_token"] == "pepper"
    assert row["source_token"] == "pepper"
    assert row["text"] == "The chef he added pepper quickly"
