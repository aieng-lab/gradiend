"""Encoder-side MLM helpers for encoder-decoder models (T5, BART, mT5, ...)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from gradiend.trainer.text.prediction.objective_hints import format_seq2seq_objective_hint


def mask_placeholder_for_tokenizer(masked_text: str, tokenizer: Any) -> str:
    """Replace ``[MASK]`` with the tokenizer's mask token or a T5-style sentinel."""
    if "[MASK]" not in masked_text:
        return masked_text
    mask_token = getattr(tokenizer, "mask_token", None)
    if mask_token:
        return masked_text.replace("[MASK]", mask_token)
    return masked_text.replace("[MASK]", "<extra_id_0>")


def tokenize_prediction_label(tokenizer: Any, label: str) -> list[int]:
    """Tokenize a prediction target, with SentencePiece leading-space fallback."""
    label = str(label)
    ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) == 1:
        return [int(ids[0])]
    spaced = tokenizer(" " + label.strip(), add_special_tokens=False)["input_ids"]
    if isinstance(spaced, torch.Tensor):
        spaced = spaced.tolist()
    if len(spaced) == 1:
        return [int(spaced[0])]
    return [int(i) for i in ids]


def seq2seq_mask_token_ids(tokenizer: Any, count: int) -> list[int]:
    """Mask/sentinel token ids for ``count`` consecutive prediction positions."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is not None:
        return [int(mask_id)] * count
    return [int(tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")) for i in range(count)]


def expand_mask_placeholders_in_text(masked_text: str, tokenizer: Any, num_tokens: int) -> str:
    """Expand one mask placeholder into ``num_tokens`` mask/sentinel slots (BERT-style)."""
    if num_tokens <= 1:
        return mask_placeholder_for_tokenizer(masked_text, tokenizer)
    mask_token = getattr(tokenizer, "mask_token", None)
    if mask_token and mask_token in masked_text:
        return masked_text.replace(mask_token, " ".join([mask_token] * num_tokens), 1)
    if "[MASK]" in masked_text:
        if mask_token:
            replacement = " ".join([mask_token] * num_tokens)
        else:
            replacement = " ".join(f"<extra_id_{i}>" for i in range(num_tokens))
        return masked_text.replace("[MASK]", replacement, 1)
    if "<extra_id_0>" in masked_text:
        replacement = " ".join(f"<extra_id_{i}>" for i in range(num_tokens))
        return masked_text.replace("<extra_id_0>", replacement, 1)
    return masked_text


def seq2seq_mask_token_id(tokenizer: Any) -> int:
    """Token id used as the mask/sentinel position for encoder-side MLM."""
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is not None:
        return int(mask_id)
    sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if sentinel_id is None or (
        getattr(tokenizer, "unk_token_id", None) is not None and sentinel_id == tokenizer.unk_token_id
    ):
        raise ValueError("Could not resolve a mask/sentinel token id for encoder-decoder MLM.")
    return int(sentinel_id)


SEQ2SEQ_DECODER_SEQUENCE_CLOZE = "seq2seq_decoder_sequence_cloze"


def _sentinel_token_id(tokenizer: Any, index: int) -> Optional[int]:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert(f"<extra_id_{index}>")
    if isinstance(token_id, (list, tuple)):
        token_id = token_id[0] if token_id else None
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and int(token_id) == int(unk_id):
        return None
    return int(token_id)


def _uses_t5_span_targets(tokenizer: Any) -> bool:
    return getattr(tokenizer, "mask_token_id", None) is None and _sentinel_token_id(tokenizer, 0) is not None


def _seq2seq_decoder_target_ids(
    tokenizer: Any,
    continuation_ids: list[int],
    *,
    include_span_sentinels: bool,
) -> list[int]:
    if not include_span_sentinels or not _uses_t5_span_targets(tokenizer):
        return continuation_ids
    start = _sentinel_token_id(tokenizer, 0)
    end = _sentinel_token_id(tokenizer, 1)
    if start is None or end is None:
        return continuation_ids
    return [start, *continuation_ids, end]


def _ids_for_text(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(i) for i in ids]


def _continuation_ids_from_prefix(tokenizer: Any, prefix: str, continuation: str) -> list[int]:
    prefix_ids = _ids_for_text(tokenizer, prefix)
    full_ids = _ids_for_text(tokenizer, prefix + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    full_ids = _ids_for_text(tokenizer, prefix + " " + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    return _ids_for_text(tokenizer, continuation)


def _limit_rhs_by_tokens(tokenizer: Any, rhs: str, rhs_window: int) -> str:
    if rhs_window is None or rhs_window < 0:
        return rhs
    if rhs_window == 0:
        return ""
    ids = _ids_for_text(tokenizer, rhs)
    if len(ids) <= rhs_window:
        return rhs
    return tokenizer.decode(ids[:rhs_window], skip_special_tokens=True)


def _effective_max_length(tokenizer: Any, base_model: Optional[Any] = None) -> int:
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len > 100_000:
        max_len = 512
    if base_model is not None:
        config = getattr(base_model, "config", None)
        if config is not None:
            for attr in ("max_position_embeddings", "n_positions", "max_length"):
                val = getattr(config, attr, None)
                if val is not None:
                    max_len = min(max_len, int(val))
    return int(max_len)


def _get_encoder_module(model: Any):
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        return encoder
    inner = getattr(model, "model", None)
    if inner is not None:
        encoder = getattr(inner, "encoder", None)
        if encoder is not None:
            return encoder
    raise TypeError(f"Cannot find encoder module on {type(model).__name__}")


def _get_lm_head(model: Any):
    head = getattr(model, "lm_head", None)
    if head is not None:
        return head
    cls = getattr(model, "cls", None)
    if cls is not None and hasattr(cls, "predictions"):
        return cls.predictions
    raise TypeError(f"Cannot find lm_head on {type(model).__name__}")


def seq2seq_encoder_mlm_logits(model: Any, item: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Vocab logits at every encoder position (batch, seq, vocab)."""
    encoder = _get_encoder_module(model)
    encoder_outputs = encoder(
        input_ids=item["input_ids"],
        attention_mask=item.get("attention_mask"),
    )
    hidden = (
        encoder_outputs.last_hidden_state
        if hasattr(encoder_outputs, "last_hidden_state")
        else encoder_outputs[0]
    )
    return _get_lm_head(model)(hidden)


def seq2seq_encoder_mlm_loss(model: Any, item: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy loss at mask positions only, matching BERT-style MLM labels."""
    logits = seq2seq_encoder_mlm_logits(model, item)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        item["labels"].view(-1),
        ignore_index=-100,
    )


def seq2seq_mlm_probs_at_mask(
    model: Any,
    tokenizer: Any,
    encoder_texts: list,
    device: torch.device,
) -> torch.Tensor:
    """Softmax vocab distribution at the mask/sentinel token for each example."""
    mask_id = seq2seq_mask_token_id(tokenizer)
    texts = [mask_placeholder_for_tokenizer(str(t), tokenizer) for t in encoder_texts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = seq2seq_encoder_mlm_logits(model, enc)
    probs = torch.softmax(logits, dim=-1)
    input_ids = enc["input_ids"]
    rows = []
    for b in range(input_ids.shape[0]):
        pos = (input_ids[b] == mask_id).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            rows.append(torch.zeros(logits.size(-1), device=device))
        else:
            rows.append(probs[b, int(pos[0].item()), :])
    return torch.stack(rows, dim=0)


def create_seq2seq_mlm_item(
    masked_text: str,
    label: str,
    tokenizer: Any,
    *,
    base_model: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build encoder inputs with BERT-style labels at the mask/sentinel position(s)."""
    max_len = _effective_max_length(tokenizer, base_model)
    target_tokens = tokenize_prediction_label(tokenizer, label)
    if not target_tokens:
        raise ValueError(f"Could not tokenize prediction label={label!r}")
    encoder_text = expand_mask_placeholders_in_text(masked_text, tokenizer, len(target_tokens))
    enc = tokenizer(
        encoder_text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    mask_ids = seq2seq_mask_token_ids(tokenizer, len(target_tokens))
    labels = enc["input_ids"].clone()
    labels[:] = -100
    input_ids = enc["input_ids"]
    if len(set(mask_ids)) == 1:
        mask_positions = (input_ids == mask_ids[0]).nonzero(as_tuple=False)
        for i, idx in enumerate(mask_positions):
            b, pos = idx.tolist()
            labels[b, pos] = target_tokens[min(i, len(target_tokens) - 1)]
    else:
        for i, mask_id in enumerate(mask_ids):
            pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                labels[0, int(pos[0].item())] = target_tokens[i]
    item = dict(enc)
    item["labels"] = labels
    if device is not None:
        item = {k: v.to(device) for k, v in item.items()}
    return item


# Backward-compatible alias
create_seq2seq_item = create_seq2seq_mlm_item


def create_seq2seq_decoder_sequence_item(
    masked_text: str,
    label: str,
    tokenizer: Any,
    *,
    base_model: Optional[Any] = None,
    device: Optional[torch.device] = None,
    rhs_window: int = -1,
    include_span_sentinels: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build encoder-decoder inputs for multi-token continuation training.

    Encoder sees ``prefix + mask sentinel + rhs`` (rhs truncated by ``rhs_window``).
    Decoder is teacher-forced on the continuation tokens for ``label``. For
    T5-style sentinel tokenizers, sequence cloze uses the natural span-corruption
    target ``<extra_id_0> continuation <extra_id_1>``.
    """
    if "[MASK]" not in masked_text:
        raise ValueError("seq2seq_decoder_sequence_cloze requires a [MASK] placeholder in the template.")
    prefix, rhs = masked_text.split("[MASK]", 1)
    rhs = _limit_rhs_by_tokens(tokenizer, rhs, rhs_window)
    continuation_ids = _continuation_ids_from_prefix(tokenizer, prefix, str(label))
    if not continuation_ids:
        continuation_ids = tokenize_prediction_label(tokenizer, label)
    if not continuation_ids:
        raise ValueError(f"Could not tokenize continuation label={label!r}")

    max_len = _effective_max_length(tokenizer, base_model)
    encoder_text = mask_placeholder_for_tokenizer(f"{prefix}[MASK]{rhs}", tokenizer)
    enc = tokenizer(
        encoder_text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    target_ids = _seq2seq_decoder_target_ids(
        tokenizer,
        continuation_ids,
        include_span_sentinels=include_span_sentinels,
    )
    label_ids = torch.tensor([target_ids], dtype=torch.long)
    item = dict(enc)
    item["labels"] = label_ids
    if device is not None:
        item = {k: v.to(device) for k, v in item.items()}
    return item


def create_seq2seq_decoder_item(
    masked_text: str,
    label: str,
    tokenizer: Any,
    *,
    base_model: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build encoder inputs + single-token decoder labels for full seq2seq training."""
    if "[MASK]" in masked_text:
        item = create_seq2seq_decoder_sequence_item(
            masked_text,
            label,
            tokenizer,
            base_model=base_model,
            device=device,
            rhs_window=0,
            include_span_sentinels=False,
        )
        if item["labels"].shape[-1] != 1:
            raise ValueError(
                f"Seq2seq decoder mode (prediction_objective='seq2seq_decoder') expects a single "
                f"target token, got {item['labels'].shape[-1]} for label={label!r}. "
                f"{format_seq2seq_objective_hint(prefix='Use')}"
            )
        return item

    max_len = _effective_max_length(tokenizer, base_model)
    encoder_text = mask_placeholder_for_tokenizer(masked_text, tokenizer)
    enc = tokenizer(
        encoder_text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    target_tokens = tokenize_prediction_label(tokenizer, label)
    if len(target_tokens) != 1:
        raise ValueError(
            f"Seq2seq decoder mode (prediction_objective='seq2seq_decoder') expects a single target "
            f"token, got {len(target_tokens)} for label={label!r}. "
            f"{format_seq2seq_objective_hint(prefix='Use')}"
        )
    label_ids = torch.tensor([target_tokens], dtype=torch.long)
    item = dict(enc)
    item["labels"] = label_ids
    if device is not None:
        item = {k: v.to(device) for k, v in item.items()}
    return item


def score_seq2seq_continuation_logprob(
    model: Any,
    tokenizer: Any,
    prefix: str,
    continuation: str,
    device: torch.device,
    *,
    rhs: str = "",
    include_span_sentinels: bool = True,
) -> float:
    """
    Sum log-probs of a teacher-forced seq2seq decoder target.

    Encoder sees ``prefix + mask sentinel + rhs``. For T5-style sentinel
    tokenizers, the default decoder target is the natural span-corruption form
    ``<extra_id_0> continuation <extra_id_1>``.
    """
    continuation_ids = _continuation_ids_from_prefix(tokenizer, prefix, continuation)
    if not continuation_ids:
        continuation_ids = tokenize_prediction_label(tokenizer, continuation)
    if not continuation_ids:
        return float("-inf")

    encoder_text = mask_placeholder_for_tokenizer(f"{prefix}[MASK]{rhs}", tokenizer)
    enc = tokenizer(encoder_text, return_tensors="pt", max_length=512, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    decoder_start = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
    if decoder_start is None:
        decoder_start = getattr(tokenizer, "pad_token_id", None) or 0

    target_ids = _seq2seq_decoder_target_ids(
        tokenizer,
        continuation_ids,
        include_span_sentinels=include_span_sentinels,
    )

    decoder_input_ids = torch.tensor(
        [[int(decoder_start)] + target_ids[:-1]],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor([target_ids], dtype=torch.long, device=device)
    outputs = model(**enc, decoder_input_ids=decoder_input_ids, labels=labels)
    if outputs.loss is None or outputs.logits is None:
        return float("-inf")

    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    logprob = 0.0
    for i, token_id in enumerate(target_ids):
        logprob += float(log_probs[0, i, token_id].item())
    return logprob
