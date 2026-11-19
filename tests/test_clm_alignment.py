"""Unit tests for CLM divergence-point alignment.

Tests coverage:
- Aligned multi-token pairs (e.g. GPT-2 anglicised/anglicized with k=2)
- Single-token regression (k=0, n=m=1)
- Multi-token with k=0
- Forbidden prefix overlap (k == min(n, m), e.g. bride/bridegroom)
- Identical target tokens
- Empty target tokenization
- Context-aware continuation and leading space handling
- audit_clm_alignment DataFrame and list diagnostics
- Informative ClmAlignmentError attributes
"""

import pytest
import pandas as pd
from transformers import AutoTokenizer

from gradiend.trainer.text.prediction.clm_alignment import (
    ClmAlignmentError,
    ClmDivergence,
    align_clm_divergence,
    audit_clm_alignment,
    continuation_ids_from_prefix,
)


@pytest.fixture(scope="module")
def gpt2_tok():
    """Load cached GPT-2 tokenizer for realistic BPE continuation tests."""
    return AutoTokenizer.from_pretrained("gpt2")


def test_align_multi_token_divergence(gpt2_tok):
    """anglicised vs anglicized share [' ang', 'lic'] and diverge at 'ised' vs 'ized'."""
    prefix = "The author's spelling was "
    div = align_clm_divergence(gpt2_tok, prefix, "anglicised", "anglicized")

    assert isinstance(div, ClmDivergence)
    assert div.k == 2
    assert div.n == 3
    assert div.m == 3
    assert len(div.shared_ids) == 2
    assert div.factual_id != div.alternative_id

    # The shared prefix tokens should decode to ' anglic'
    shared_text = gpt2_tok.decode(list(div.shared_ids))
    assert "ang" in shared_text and "lic" in shared_text

    # The divergence tokens should decode to 'ised' and 'ized'
    f_token = gpt2_tok.decode([div.factual_id])
    a_token = gpt2_tok.decode([div.alternative_id])
    assert f_token.strip() == "ised"
    assert a_token.strip() == "ized"


def test_align_single_token_regression(gpt2_tok):
    """Single-token words must have k=0, empty shared_ids, and n=m=1."""
    prefix = "The weather today is "
    div = align_clm_divergence(gpt2_tok, prefix, "good", "bad")

    assert div.k == 0
    assert div.shared_ids == ()
    assert div.n == 1
    assert div.m == 1
    assert div.factual_id == gpt2_tok.encode(" good", add_special_tokens=False)[0]
    assert div.alternative_id == gpt2_tok.encode(" bad", add_special_tokens=False)[0]


def test_align_multi_token_k_zero(gpt2_tok):
    """Words that share no prefix (k=0) but have >1 tokens must diverge at index 0."""
    prefix = "The discussion was completely "
    div = align_clm_divergence(gpt2_tok, prefix, "unrelated", "different")

    assert div.k == 0
    assert div.shared_ids == ()
    assert div.n >= 1
    assert div.m >= 1
    assert div.factual_id != div.alternative_id


def test_forbidden_prefix_overlap_raises(gpt2_tok):
    """bride vs bridegroom has k == n (shorter side prefix of longer side) -> forbidden."""
    prefix = "In the photograph stood the "
    with pytest.raises(ClmAlignmentError) as exc_info:
        align_clm_divergence(gpt2_tok, prefix, "bride", "bridegroom")

    err = exc_info.value
    assert err.reason == "prefix_overlap"
    assert err.factual == "bride"
    assert err.alternative == "bridegroom"
    assert err.k > 0

    # Reverse direction must also raise
    with pytest.raises(ClmAlignmentError) as exc_info_rev:
        align_clm_divergence(gpt2_tok, prefix, "bridegroom", "bride")
    assert exc_info_rev.value.reason == "prefix_overlap"


def test_identical_targets_raise(gpt2_tok):
    """Identical targets have k == n == m -> forbidden zero contrast."""
    prefix = "The hospital hired a "
    with pytest.raises(ClmAlignmentError) as exc_info:
        align_clm_divergence(gpt2_tok, prefix, "nurse", "nurse")

    err = exc_info.value
    assert err.reason == "identical_tokens"


def test_empty_continuation_raises(gpt2_tok):
    """Empty string targets cannot be aligned."""
    prefix = "The sentence ends here. "
    with pytest.raises(ClmAlignmentError) as exc_info:
        align_clm_divergence(gpt2_tok, prefix, "", "hello")
    assert exc_info.value.reason == "empty_tokenization"


def test_continuation_ids_gap_handling(gpt2_tok):
    """continuation_ids_from_prefix handles space boundaries cleanly."""
    p_trailing = "The word is "
    ids_trailing = continuation_ids_from_prefix(gpt2_tok, p_trailing, "anglicised")
    assert ids_trailing[0] == gpt2_tok.encode(" ang", add_special_tokens=False)[0]
    assert len(ids_trailing) == 3

    # When continuation explicitly includes leading space
    ids_with_space = continuation_ids_from_prefix(gpt2_tok, "The word is", " anglicised")
    assert ids_with_space == ids_trailing


def test_audit_clm_alignment_dataframe(gpt2_tok):
    """audit_clm_alignment must classify single, aligned, and invalid rows accurately."""
    df = pd.DataFrame([
        {"masked": "The word is [MASK]", "factual": "anglicised", "alternative": "anglicized"},
        {"masked": "The result was [MASK]", "factual": "good", "alternative": "bad"},
        {"masked": "The ceremony had a [MASK]", "factual": "bride", "alternative": "bridegroom"},
        {"masked": "She saw a [MASK]", "factual": "cat", "alternative": "cat"},
    ])

    audit = audit_clm_alignment(gpt2_tok, df)
    assert len(audit) == 4
    assert list(audit["status"]) == ["aligned", "single", "invalid", "invalid"]
    assert audit.loc[0, "k"] == 2
    assert audit.loc[1, "k"] == 0
    assert audit.loc[2, "reason"] == "prefix_overlap"
    assert audit.loc[3, "reason"] == "identical_tokens"


def test_audit_clm_alignment_tuple_list(gpt2_tok):
    """audit_clm_alignment must accept a sequence of (factual, alternative) tuples."""
    pairs = [
        ("anglicised", "anglicized"),
        ("good", "bad"),
        ("bride", "bridegroom"),
    ]
    audit = audit_clm_alignment(gpt2_tok, pairs, context_prefix="The word is ")
    assert len(audit) == 3
    assert audit.loc[0, "status"] == "aligned"
    assert audit.loc[1, "status"] == "single"
    assert audit.loc[2, "status"] == "invalid"


class MockCharTokenizer:
    """Mock tokenizer mapping characters to integer ids for fast deterministic testing."""

    def __init__(self, padding_side="right", pad_id=0):
        self.padding_side = padding_side
        self.pad_token_id = pad_id
        self.eos_token_id = 99

    def __call__(self, text, add_special_tokens=False, **_):
        chars = [ord(c) for c in str(text)]
        return {"input_ids": chars}


def test_align_with_mock_tokenizer():
    """align_clm_divergence must work purely on token ids without HuggingFace model dependency."""
    tok = MockCharTokenizer()
    # "abcX" vs "abcY" -> shared prefix "abc" (k=3), divergence at 'X' vs 'Y'
    div = align_clm_divergence(tok, "pre ", "abcX", "abcY")
    assert div.k == 3
    assert div.factual_id == ord("X")
    assert div.alternative_id == ord("Y")
    assert div.shared_ids == (ord("a"), ord("b"), ord("c"))
