"""CLM divergence-point alignment for multi-token factual and alternative targets.

Decoder-only causal language models predict the next token given preceding context.
When factual and alternative target phrases share a common token prefix of length
k (where k < len(factual) and k < len(alternative)), the causal divergence point
occurs at token position k+1 (0-indexed position k). At this position, preceding
shared tokens act as context, and the contrastive supervision is applied to
predict the differing tokens factual[k] vs alternative[k].

For an Instruct model (a chat tokenizer, see :func:`~gradiend.trainer.text.common.chat_format.uses_chat_format`) the
context is the chat prompt, which ends with the generation prompt of the assistant turn, and the answer words are the
first thing the assistant says: they are tokenized on their own, without a leading space, and the row is
``chat prompt + shared tokens + slot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch

from gradiend.trainer.core.arguments import LABEL_TOKEN_PROTOCOLS
from gradiend.trainer.text.common.chat_format import chat_prompt_ids, raw_tokenizer, uses_chat_format
from gradiend.util.logging import suppress_tokenizer_length_warning
from gradiend.util.positions import assert_labels_on_real_tokens, last_real_token_positions


class ClmAlignmentError(ValueError):
    """Raised when factual and alternative targets cannot be contrastively aligned for CLM."""

    def __init__(
        self,
        message: str,
        *,
        factual: Optional[str] = None,
        alternative: Optional[str] = None,
        context_prefix: Optional[str] = None,
        factual_ids: Optional[Sequence[int]] = None,
        alternative_ids: Optional[Sequence[int]] = None,
        k: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.factual = factual
        self.alternative = alternative
        self.context_prefix = context_prefix
        self.factual_ids = tuple(factual_ids) if factual_ids is not None else ()
        self.alternative_ids = tuple(alternative_ids) if alternative_ids is not None else ()
        self.k = k
        self.reason = reason


@dataclass(frozen=True)
class ClmDivergence:
    """Result of aligning factual and alternative targets for decoder-only CLM supervision.

    Attributes:
        shared_ids: Token ids of the shared prefix between factual and alternative.
        factual_id: Divergence token id to predict for the factual target.
        alternative_id: Divergence token id to predict for the alternative target.
        n: Total number of token ids in the factual continuation.
        m: Total number of token ids in the alternative continuation.
    """

    shared_ids: Tuple[int, ...]
    factual_id: int
    alternative_id: int
    n: int
    m: int

    @property
    def k(self) -> int:
        """Length of the shared common prefix."""
        return len(self.shared_ids)


def _ids_for_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenize ``text`` without special tokens and return token ids.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to tokenize.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return tok(str(text), add_special_tokens=False)["input_ids"]


def continuation_ids_from_prefix(tokenizer: Any, prefix: str, continuation: str) -> List[int]:
    """Return token ids contributed by ``continuation`` after ``prefix``.

    Handles tokenizer gap / space mechanics by checking full text tokenization against prefix.

    Args:
        tokenizer: Tokenizer used for encoding.
        prefix: Text before the continuation.
        continuation: Candidate continuation text.
    """
    if not continuation:
        return []
    prefix_ids = _ids_for_text(tokenizer, prefix)
    full_ids = _ids_for_text(tokenizer, prefix + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    full_ids = _ids_for_text(tokenizer, prefix + " " + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    clean_prefix = prefix.rstrip()
    if clean_prefix and clean_prefix != prefix:
        clean_prefix_ids = _ids_for_text(tokenizer, clean_prefix)
        full_ids = _ids_for_text(tokenizer, prefix + continuation)
        if full_ids[: len(clean_prefix_ids)] == clean_prefix_ids:
            return full_ids[len(clean_prefix_ids) :]
        full_ids = _ids_for_text(tokenizer, clean_prefix + " " + continuation.lstrip())
        if full_ids[: len(clean_prefix_ids)] == clean_prefix_ids:
            return full_ids[len(clean_prefix_ids) :]
    return _ids_for_text(tokenizer, continuation)


def answer_ids(tokenizer: Any, word: str) -> List[int]:
    """Return the token ids of ``word`` as the first thing the assistant says after a chat prompt.

    The generation prompt ends with a newline, so the word starts the text: it is tokenized on its own, without
    a leading space. A leading token that decodes to blank space (a tokenizer artifact) is dropped.

    Args:
        tokenizer: A chat tokenizer or its raw tokenizer.
        word: Answer word.
    """
    raw = raw_tokenizer(tokenizer)
    ids = list(continuation_ids_from_prefix(raw, "", str(word)))
    if ids and raw.decode(ids[0]).strip() == "":
        ids = ids[1:]
    return ids


def label_token_ids(
    tokenizer: Any,
    prefix: str,
    word: Any,
    protocol: str = "first_differing_token",
    *,
    vocab: Optional[Any] = None,
) -> List[int]:
    """The token ids that label ``word`` after ``prefix`` under ``protocol``: THE label rule of decoder-only rows.

    Every place that needs to know which token(s) a word is after a prefix asks here: the training item, the
    single-row scorer ``create_inputs``, the divergence alignment (:func:`align_clm_divergence`), the alignment
    audit and the row-wise decoder evaluation. They keep no variant of their own.

    - ``"first_differing_token"``: the ids ``word`` contributes when it follows ``prefix`` (see

      :func:`continuation_ids_from_prefix`). After a prefix that ends in a space this is the *leading-space*
      token (``Ġshe`` / ``▁she``); when the slot is glued to the preceding character (``"("`` + ``[MASK]``) or the
      prefix is empty it is the literal token without a space. A chat tokenizer (Instruct model) gets the answer
      ids of the assistant turn (:func:`answer_ids`), whatever the prefix.

    - ``"no_leading_space"``: the rule of gradiend 0.2.5, independent of the prefix: ``vocab[word]``, else the first

      id of ``tokenizer(word)``.

    Args:
        tokenizer: Plain or chat tokenizer.
        prefix: Text in front of the slot (the part of the masked text before the mask).
        word: The label word.
        protocol: ``TrainingArguments.label_token_protocol``.
        vocab: Optional ``token -> id`` map used by ``"no_leading_space"`` (default ``tokenizer.vocab``).

    Returns:
        All ids of the word (callers apply their own multi-token policy); one id under ``"no_leading_space"``.

    Raises:
        ValueError: If ``protocol`` is unknown.
    """
    if protocol == "first_differing_token":
        if uses_chat_format(tokenizer):
            return answer_ids(tokenizer, word)
        return continuation_ids_from_prefix(tokenizer, prefix, str(word))
    if protocol != "no_leading_space":
        raise ValueError(f"label_token_protocol must be one of {LABEL_TOKEN_PROTOCOLS}, got {protocol!r}")
    word = str(word)
    if vocab is None:
        vocab = getattr(tokenizer, "vocab", None)
    if vocab is not None and word in vocab:
        return [int(vocab[word])]
    return [int(tokenizer(word, add_special_tokens=False)["input_ids"][0])]


def label_token_id(
    tokenizer: Any,
    prefix: str,
    word: Any,
    protocol: str = "first_differing_token",
    *,
    vocab: Optional[Any] = None,
) -> int:
    """The one id a row of ``word`` after ``prefix`` supervises: the first id of :func:`label_token_ids`.

    Raises:
        ValueError: If ``protocol`` is unknown or ``word`` has no token.
    """
    ids = label_token_ids(tokenizer, prefix, word, protocol, vocab=vocab)
    if not ids:
        raise ValueError(f"Could not tokenize label {word!r} to any token id.")
    return int(ids[0])


def align_clm_divergence(
    tokenizer: Any,
    context_prefix: str,
    factual: str,
    alternative: str,
) -> ClmDivergence:
    """Align factual and alternative continuations to find their first divergence point.

    Tokenizes ``factual`` and ``alternative`` as continuations of ``context_prefix``
    (respecting leading-space and BPE boundary behaviors); for a chat tokenizer they are tokenized as answer
    words that open the assistant turn (see :func:`answer_ids`), whatever ``context_prefix`` is. Let n = len(factual_ids),
    m = len(alternative_ids), and k = length of their longest common prefix.

    The alignment is valid iff strict inequality holds: ``k < n AND k < m``.
    When valid:

    - ``shared_ids`` = factual_ids[:k] (identical to alternative_ids[:k])
    - ``factual_id`` = factual_ids[k]
    - ``alternative_id`` = alternative_ids[k]

    If ``k == n`` or ``k == m``, one phrase is a token prefix of the other (e.g.
    'bride' vs 'bridegroom'). If ``factual_ids == alternative_ids``, the tokens are
    identical. Both cases are invalid and raise :class:`ClmAlignmentError`.

    Args:
        tokenizer: Tokenizer or model with a tokenizer.
        context_prefix: Preceding sentence context (or prefix before the mask).
        factual: Factual target string.
        alternative: Alternative target string.

    Returns:
        ClmDivergence instance containing shared_ids, factual_id, alternative_id, n, m.

    Raises:
        ClmAlignmentError: If targets cannot be contrastively aligned.
    """
    f_str = str(factual)
    a_str = str(alternative)
    f_ids = label_token_ids(tokenizer, context_prefix, f_str)
    a_ids = label_token_ids(tokenizer, context_prefix, a_str)
    n = len(f_ids)
    m = len(a_ids)

    if n == 0 or m == 0:
        raise ClmAlignmentError(
            f"Cannot align empty continuation: factual={f_str!r} (ids={f_ids}), "
            f"alternative={a_str!r} (ids={a_ids}) after prefix={context_prefix!r}.",
            factual=f_str,
            alternative=a_str,
            context_prefix=context_prefix,
            factual_ids=f_ids,
            alternative_ids=a_ids,
            k=0,
            reason="empty_tokenization",
        )

    k = 0
    while k < n and k < m and f_ids[k] == a_ids[k]:
        k += 1

    if k == n and k == m:
        raise ClmAlignmentError(
            f"Factual {f_str!r} and alternative {a_str!r} tokenize to identical token "
            f"sequences {f_ids} after prefix {context_prefix!r}; cannot supervise zero contrast.",
            factual=f_str,
            alternative=a_str,
            context_prefix=context_prefix,
            factual_ids=f_ids,
            alternative_ids=a_ids,
            k=k,
            reason="identical_tokens",
        )

    if k == n or k == m:
        shorter = "factual" if n < m else "alternative"
        longer = "alternative" if n < m else "factual"
        raise ClmAlignmentError(
            f"One target is a token-prefix of the other: {shorter} (n={n}, ids={f_ids if n < m else a_ids}) "
            f"is a prefix of {longer} (m={m}, ids={a_ids if n < m else f_ids}) with common prefix k={k}. "
            f"The shorter side has no valid next-token contrast.",
            factual=f_str,
            alternative=a_str,
            context_prefix=context_prefix,
            factual_ids=f_ids,
            alternative_ids=a_ids,
            k=k,
            reason="prefix_overlap",
        )

    return ClmDivergence(
        shared_ids=tuple(f_ids[:k]),
        factual_id=f_ids[k],
        alternative_id=a_ids[k],
        n=n,
        m=m,
    )


def resolve_clm_labels(
    tokenizer: Any, prefix: str, factual: Any, alternative: Optional[Any] = None
) -> Tuple[Tuple[int, ...], int]:
    """The ids that make up the label of one decoder-only row under ``label_token_protocol="first_differing_token"``.

    With an ``alternative`` that differs from ``factual`` the row is a contrast row: the result is the tokens the two
    share (they join the context) and the first token where they differ (:func:`align_clm_divergence`; unalignable
    pairs raise :class:`ClmAlignmentError`). Without one (or with the very same string, a neutral identity row) nothing
    is shared and the label is the first id of :func:`label_token_ids`. Both cases read the word ids from the same
    function, so a row with and without an alternative gets the same token variant for the same prefix.

    Args:
        tokenizer: Plain or chat tokenizer.
        prefix: Text in front of the slot.
        factual: Factual target word.
        alternative: Alternative target word, or None.

    Returns:
        ``(shared_ids, label_id)``.
    """
    if alternative is not None and str(alternative) != str(factual):
        divergence = align_clm_divergence(tokenizer, prefix, str(factual), str(alternative))
        return divergence.shared_ids, divergence.factual_id
    return (), label_token_id(tokenizer, prefix, factual)


def clm_context_ids(tokenizer: Any, prefix: str, *, max_length: Optional[int] = None) -> List[int]:
    """Return the token ids of the context in front of the shared tokens of a divergence row.

    This is the one definition of that context, used by the training item, ``create_inputs`` and the decoder
    evaluation: ``tokenizer(prefix.rstrip())`` with the tokenizer's own special tokens, or, for a chat tokenizer,
    the chat prompt of ``prefix.rstrip()`` (system prompt, user turn, generation prompt; see
    :func:`~gradiend.trainer.text.common.chat_format.chat_prompt_ids`). Trailing whitespace of the prefix is
    dropped in both cases: the answer is a separate token (a plain model) or a separate turn (a chat model), so a
    prefix with and without a trailing space gives the same row.

    Args:
        tokenizer: Plain or chat tokenizer (or wrapper with ``.tokenizer``).
        prefix: Text before the target slot.
        max_length: Maximum number of context tokens; only a chat prompt is shortened to it (see
            :func:`~gradiend.trainer.text.common.chat_format.chat_prompt_ids`).
    """
    if uses_chat_format(tokenizer):
        return chat_prompt_ids(tokenizer, str(prefix).rstrip(), max_length=max_length)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    with suppress_tokenizer_length_warning():
        return list(tok(str(prefix).rstrip(), add_special_tokens=True, truncation=False)["input_ids"])


def build_shared_prefix_item(
    tokenizer: Any,
    prefix: str,
    shared_ids: Sequence[int],
    label_id: int,
    *,
    max_length: int,
    pad_to_max_length: bool,
) -> Dict[str, torch.Tensor]:
    """Build the one decoder-only item whose label sits on the token after a shared prefix.

    The sequence is ``tokenizer(prefix.rstrip()) + shared_ids + [slot]``; ``label_id`` is written onto the
    slot (the last real token) and, because causal-LM loss shifts by one, is predicted from the token
    before it. The prefix is encoded without trailing whitespace so no lone space token separates it from
    the shared tokens. The slot token only carries the label; no prediction reads its input id.

    If the sequence is longer than ``max_length`` the TAIL is kept (the label sits at the end), whatever
    the tokenizer's truncation side. A row with no real token before the slot (empty prefix, no shared
    tokens) follows the dataset's empty-row convention when padding: an all-padding row with the label
    on its last position.

    For a chat tokenizer (Instruct model) the context is the chat prompt of ``prefix.rstrip()``: the
    sequence is ``chat prompt + shared_ids + [slot]``, so the label is predicted from the position that
    ends the generation prompt (or the last shared token), the position the decoder evaluation reads. A
    prefix that does not fit is shortened from its left, never the system prompt or the generation prompt;
    a ``max_length`` too small for the chat frame raises.

    Args:
        tokenizer: Tokenizer (or wrapper with ``.tokenizer``).
        prefix: Text before the target slot.
        shared_ids: Token ids shared by factual and alternative (may be empty).
        label_id: Token id to supervise.
        max_length: Maximum sequence length.
        pad_to_max_length: Pad to ``max_length`` (training item) or return the unpadded sequence.

    Returns:
        Dict of ``(1, L)`` tensors ``input_ids``, ``attention_mask``, ``labels``.

    Raises:
        ValueError: If ``tokenizer`` is a chat tokenizer and ``max_length`` cannot hold the chat frame.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    budget = int(max_length) - len(shared_ids) - 1
    ctx_ids = clm_context_ids(tokenizer, prefix, max_length=budget)
    if uses_chat_format(tokenizer) and len(ctx_ids) > budget:
        raise ValueError(
            f"max_length={max_length} is too small for the chat prompt of {getattr(tok, 'name_or_path', tok)!r}: "
            f"its system prompt and headers take {len(ctx_ids)} tokens, plus {len(shared_ids) + 1} for the answer."
        )
    slot_id = getattr(tok, "eos_token_id", None)
    if slot_id is None:
        slot_id = getattr(tok, "pad_token_id", None)
    if slot_id is None:
        slot_id = 0
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = slot_id

    empty_row = pad_to_max_length and not ctx_ids and not shared_ids
    tokens = [] if empty_row else ctx_ids + list(shared_ids) + [slot_id]
    if len(tokens) > max_length:
        tokens = tokens[-max_length:]
    if pad_to_max_length:
        pad_len = max_length - len(tokens)
        if getattr(tok, "padding_side", "right") == "left":
            ids = [pad_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
        else:
            ids = tokens + [pad_id] * pad_len
            mask = [1] * len(tokens) + [0] * pad_len
    else:
        ids, mask = tokens, [1] * len(tokens)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.tensor([mask], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    label_pos = -1 if empty_row else int(last_real_token_positions(attention_mask)[0])
    labels[0, label_pos] = int(label_id)
    assert_labels_on_real_tokens(labels, attention_mask, where="shared-prefix item", allow_empty_rows=True)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def audit_clm_alignment(
    tokenizer: Any,
    pairs_or_df: Any,
    context_prefix: str = "",
) -> pd.DataFrame:
    """Audit target pairs or DataFrame rows for decoder-only CLM divergence alignment.

    Diagnostic utility to inspect which rows are single-token, multi-token aligned,
    or unalignable (invalid).

    Args:
        tokenizer: Tokenizer used for evaluation/training.
        pairs_or_df: DataFrame with factual/alternative (and optionally masked) columns,
            or an iterable of (factual, alternative) string tuples or dicts.
        context_prefix: Default context prefix if not present in the row.

    Returns:
        DataFrame with columns ['factual', 'alternative', 'prefix', 'factual_ids',
        'alternative_ids', 'n', 'm', 'k', 'status', 'reason'].
        'status' is one of:

        - 'single': n=1, m=1, k=0 (standard single-token)
        - 'aligned': k < n and k < m (multi-token divergence aligned)
        - 'invalid': prefix overlap, identical tokens, or empty tokens
    """
    records: List[Dict[str, Any]] = []

    if isinstance(pairs_or_df, pd.DataFrame):
        df = pairs_or_df
        fact_col = "factual" if "factual" in df.columns else ("label" if "label" in df.columns else None)
        alt_col = "alternative" if "alternative" in df.columns else None
        mask_col = "masked" if "masked" in df.columns else ("text" if "text" in df.columns else None)
        for _, row in df.iterrows():
            f_val = str(row[fact_col]) if fact_col and fact_col in row else ""
            a_val = str(row[alt_col]) if alt_col and alt_col in row else ""
            prefix = context_prefix
            if mask_col and mask_col in row and pd.notna(row[mask_col]):
                raw_text = str(row[mask_col])
                if "[MASK]" in raw_text:
                    prefix = raw_text.split("[MASK]")[0]
                elif not context_prefix:
                    prefix = raw_text
            records.append({"factual": f_val, "alternative": a_val, "prefix": prefix})
    elif isinstance(pairs_or_df, (list, tuple)):
        for item in pairs_or_df:
            if isinstance(item, dict):
                f_val = str(item.get("factual", item.get("label", "")))
                a_val = str(item.get("alternative", ""))
                prefix = item.get("prefix", item.get("masked", context_prefix))
                if "[MASK]" in prefix:
                    prefix = prefix.split("[MASK]")[0]
                records.append({"factual": f_val, "alternative": a_val, "prefix": prefix})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                records.append({"factual": str(item[0]), "alternative": str(item[1]), "prefix": context_prefix})
            else:
                raise ValueError(f"Unrecognized pair format: {item!r}")
    else:
        raise TypeError(f"pairs_or_df must be DataFrame, list, or tuple, got {type(pairs_or_df).__name__}")

    results: List[Dict[str, Any]] = []
    for rec in records:
        f = rec["factual"]
        a = rec["alternative"]
        p = rec["prefix"]
        f_ids = label_token_ids(tokenizer, p, f)
        a_ids = label_token_ids(tokenizer, p, a)
        n = len(f_ids)
        m = len(a_ids)

        if n == 0 or m == 0:
            status = "invalid"
            reason = "empty_tokenization"
            k = 0
        else:
            k = 0
            while k < n and k < m and f_ids[k] == a_ids[k]:
                k += 1
            if k == n and k == m:
                status = "invalid"
                reason = "identical_tokens"
            elif k == n or k == m:
                status = "invalid"
                reason = "prefix_overlap"
            elif n == 1 and m == 1 and k == 0:
                status = "single"
                reason = "single_token"
            else:
                status = "aligned"
                reason = "divergence_aligned"

        results.append({
            "factual": f,
            "alternative": a,
            "prefix": p,
            "factual_ids": f_ids,
            "alternative_ids": a_ids,
            "n": n,
            "m": m,
            "k": k,
            "status": status,
            "reason": reason,
        })

    return pd.DataFrame(results)
