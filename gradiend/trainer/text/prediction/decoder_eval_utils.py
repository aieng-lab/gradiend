"""
Decoder evaluation helpers for text prediction (LMS, feature/target token scores).

Used by TextPredictionTrainer.evaluate_base_model. No dependency on
trainer.core.feature_definition to avoid circular imports.

Row-wise mode: for each row, P(dataset_class) = P(factual token), P(other_class) = P(alternative token).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch

from gradiend.model.utils import is_decoder_only_model, prediction_eval_kind
from gradiend.trainer.text.prediction.seq2seq import (
    mask_placeholder_for_tokenizer,
    score_seq2seq_continuation_logprob,
    seq2seq_mlm_probs_at_mask,
)
from gradiend.util.logging import get_logger

from gradiend.trainer.text.common.lm_eval import evaluate_mlm, evaluate_clm_perplexity, evaluate_seq2seq_perplexity, compute_lms


logger = get_logger(__name__)

# Column names for row-wise eval (unified schema)
DEFAULT_FACTUAL_COL = "factual"
DEFAULT_ALTERNATIVE_COL = "alternative"
DEFAULT_DATASET_CLASS_COL = "factual_id"
DEFAULT_OTHER_CLASS_COL = "alternative_id"


def _normalize_token_string(value: str) -> str:
    return str(value).lstrip().lower() if value is not None else ""


def _token_to_single_id(tokenizer, token_str: str, vocab_norm_map: Dict[str, List[str]]) -> Optional[int]:
    """Resolve a single token string to one token id (single token or first subword). Returns None if invalid."""
    if token_str is None or (isinstance(token_str, float) and np.isnan(token_str)):
        return None
    s = str(token_str).strip()
    if not s:
        return None
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    norm = _normalize_token_string(s)
    candidates = vocab_norm_map.get(norm, []) or [s]
    for cand in candidates:
        tokenized = _tokenizer.tokenize(cand)
        if len(tokenized) == 1:
            return _tokenizer.convert_tokens_to_ids(tokenized[0])
        if len(tokenized) > 1:
            return _tokenizer.convert_tokens_to_ids(tokenized[0])
    return _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(s)[0]) if _tokenizer.tokenize(s) else None


def _build_vocab_norm_map(tokenizer) -> Dict[str, List[str]]:
    vocab = None
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None
    if vocab is None and hasattr(tokenizer, "vocab"):
        vocab = getattr(tokenizer, "vocab", None)
    if not vocab:
        return {}
    norm_map: Dict[str, List[str]] = defaultdict(list)
    for token in vocab.keys():
        norm_map[_normalize_token_string(token)].append(token)
    return norm_map


def annotate_text_probability_rows(
    model,
    tokenizer,
    df: pd.DataFrame,
    targets: Dict[str, List[str]],
    *,
    key_text: str = "masked",
    batch_size: int = 16,
    safe_token_map: Optional[Dict[str, str]] = None,
    prefix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Return per-row target-token and class probabilities for text prediction data.

    The resulting DataFrame contains columns:
    - ``{prefix}p_target_{safe_token}``
    - ``{prefix}p_class_{class_name}``
    """
    if key_text not in df.columns:
        raise ValueError(
            f"annotate_text_probability_rows() requires column {key_text!r}. Available: {list(df.columns)}"
        )

    token_set = sorted({str(token) for values in targets.values() for token in values if token is not None})
    if safe_token_map is None:
        safe_token_map = {str(idx): token for idx, token in enumerate(token_set)}
    token_to_safe = {token: safe for safe, token in safe_token_map.items()}

    model.eval()
    device = model.device
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    eval_kind = prediction_eval_kind(model)
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)
    mask_token = getattr(tokenizer, "mask_token", None) or getattr(_tokenizer, "mask_token", None)
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or getattr(_tokenizer, "mask_token_id", None)

    token_id_map: Dict[str, int] = {}
    for token in token_set:
        token_id = _token_to_single_id(_tokenizer, token, vocab_norm_map)
        if token_id is None:
            logger.warning("Skipping unresolved annotation token %r", token)
            continue
        token_id_map[token] = token_id
    if not token_id_map:
        raise ValueError("annotate_text_probability_rows() could not resolve any target token ids.")

    prob_rows: List[Dict[str, float]] = []
    rows = df.to_dict("records")
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if eval_kind == "clm_next_token":
                prefixes = [str(r[key_text]).split("[MASK]")[0] for r in batch]
                valid = [(i, p) for i, p in enumerate(prefixes) if p and p.strip()]
                example_probs = {}
                if valid:
                    idxs, prefix_texts = zip(*valid)
                    inputs = tokenizer(
                        list(prefix_texts), return_tensors="pt", padding=True, truncation=True, max_length=512
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    example_probs = {i: probs[j] for j, i in enumerate(idxs)}
            elif eval_kind == "seq2seq_encoder_mlm":
                texts = [str(r[key_text]) for r in batch]
                probs = seq2seq_mlm_probs_at_mask(model, _tokenizer, texts, device)
                example_probs = {i: probs[i] for i in range(len(batch))}
            else:
                if mask_token is None or mask_token_id is None:
                    raise ValueError("annotate_text_probability_rows() requires tokenizer.mask_token for MLM models.")
                texts = [str(r[key_text]).replace("[MASK]", mask_token) for r in batch]
                inputs = tokenizer(
                    list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                mask_idxs = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)
                example_probs = {}
                if len(mask_idxs) > 0:
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    for b_idx, pos in mask_idxs.tolist():
                        example_probs[b_idx] = probs[b_idx, pos, :]

            for b_idx, _row in enumerate(batch):
                p = example_probs.get(b_idx)
                if p is None:
                    prob_rows.append({})
                    continue
                record: Dict[str, float] = {}
                token_probs: Dict[str, float] = {}
                for token, token_id in token_id_map.items():
                    safe = token_to_safe[token]
                    value = float(p[token_id].item() if hasattr(p[token_id], "item") else p[token_id])
                    token_probs[token] = value
                    record[f"{prefix}p_target_{safe}"] = value
                for class_name, class_tokens in targets.items():
                    record[f"{prefix}p_class_{class_name}"] = float(
                        sum(token_probs.get(str(token), 0.0) for token in class_tokens)
                    )
                prob_rows.append(record)

    return pd.DataFrame(prob_rows), safe_token_map


def compute_probability_shift_score_row_wise(
    model,
    tokenizer,
    df,
    key_text: str = "masked",
    batch_size: int = 16,
    factual_col: str = DEFAULT_FACTUAL_COL,
    alternative_col: str = DEFAULT_ALTERNATIVE_COL,
    dataset_class_col: str = DEFAULT_DATASET_CLASS_COL,
    other_class_col: str = DEFAULT_OTHER_CLASS_COL,
    return_per_row_df: bool = False,
) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], pd.DataFrame]]:
    """
    Row-wise decoder eval: for each row, P(dataset_class) = P(factual), P(other_class) = P(alternative).

    DataFrame must have: key_text (masked), factual_col, alternative_col, dataset_class_col (e.g. factual_id),
    other_class_col (e.g. alternative_id). Returns same shape as static eval: {dataset_class: {class_name: mean_prob}}.
    When return_per_row_df=True, also returns a DataFrame with one row per sample: masked, factual, alternative,
    factual_id, alternative_id, dataset_class, other_class, P_factual, P_alternative.
    """
    model.eval()
    device = model.device
    eval_kind = prediction_eval_kind(model)
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or getattr(_tokenizer, "mask_token_id", None)

    for col in (key_text, factual_col, alternative_col, dataset_class_col, other_class_col):
        if col not in df.columns:
            raise ValueError(
                f"Row-wise decoder eval requires column '{col}' in the DataFrame. "
                f"Available: {list(df.columns)}."
            )

    probs_by_dataset: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_row_records: List[Dict[str, object]] = []
    rows = df.to_dict("records")

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if eval_kind == "clm_next_token":
                prefixes = [r[key_text].split("[MASK]")[0] for r in batch]
                valid = [(i, p) for i, p in enumerate(prefixes) if p and p.strip()]
                if not valid:
                    continue
                idxs, prefix_texts = zip(*valid)
                inputs = tokenizer(
                    list(prefix_texts), return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                example_probs = {i: probs[j] for j, i in enumerate(idxs)}
            elif eval_kind == "seq2seq_encoder_mlm":
                texts = [str(r[key_text]) for r in batch]
                probs = seq2seq_mlm_probs_at_mask(model, _tokenizer, texts, device)
                example_probs = {i: probs[i] for i in range(len(batch))}
            else:
                mask_tok = getattr(tokenizer, "mask_token", None) or getattr(_tokenizer, "mask_token", None)
                texts = [r[key_text].replace("[MASK]", mask_tok) for r in batch]
                inputs = tokenizer(
                    list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if mask_token_id is None:
                    continue
                mask_idxs = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)
                if len(mask_idxs) == 0:
                    continue
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                example_probs = {}
                for b_idx, pos in mask_idxs.tolist():
                    example_probs[b_idx] = probs[b_idx, pos, :]

            for b_idx, row in enumerate(batch):
                if b_idx not in example_probs:
                    continue
                p = example_probs[b_idx]
                ds_class = row.get(dataset_class_col)
                other_class = row.get(other_class_col)
                if ds_class is None or other_class is None:
                    continue
                fid = _token_to_single_id(_tokenizer, row.get(factual_col), vocab_norm_map)
                aid = _token_to_single_id(_tokenizer, row.get(alternative_col), vocab_norm_map)
                if fid is None or aid is None:
                    continue
                p_f = float(p[fid].item() if hasattr(p[fid], "item") else p[fid])
                p_a = float(p[aid].item() if hasattr(p[aid], "item") else p[aid])
                probs_by_dataset[ds_class][ds_class].append(p_f)
                probs_by_dataset[ds_class][other_class].append(p_a)
                if return_per_row_df:
                    per_row_records.append({
                        "masked": row.get(key_text),
                        "factual": row.get(factual_col),
                        "alternative": row.get(alternative_col),
                        "factual_id": ds_class,
                        "alternative_id": other_class,
                        "dataset_class": ds_class,
                        "other_class": other_class,
                        "P_factual": p_f,
                        "P_alternative": p_a,
                    })

    probs_by_dataset_means = {}
    for ds_cls, class_probs in probs_by_dataset.items():
        probs_by_dataset_means[ds_cls] = {
            cls: float(np.mean(vals)) if vals else 0.0 for cls, vals in class_probs.items()
        }
    if not probs_by_dataset_means:
        raise ValueError(
            "Row-wise decoder eval produced no probabilities. Ensure the DataFrame has columns "
            "factual, alternative, factual_id, alternative_id and that tokenizer can tokenize the tokens."
        )
    logger.debug(
        "Decoder eval (row-wise) probability means: %s",
        ", ".join(f"{ds}: {list(p.keys())}" for ds, p in sorted(probs_by_dataset_means.items())),
    )
    if return_per_row_df and per_row_records:
        per_row_df = pd.DataFrame(per_row_records)
        return (probs_by_dataset_means, per_row_df)
    if return_per_row_df:
        return (probs_by_dataset_means, pd.DataFrame())
    return probs_by_dataset_means


def compute_probability_shift_score_clm(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    dataset_class_col=None,
):
    """
    Compute probabilities for all classes evaluated on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        df: DataFrame with evaluation data
        targets: Dict mapping class_name -> list of tokens
        key_text: Column name for masked text
        batch_size: Batch size for evaluation
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, uses "label_class" if available, else "factual_id".

    Returns:
        Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
    """
    model.eval()
    device = model.device
    eval_kind = prediction_eval_kind(model)
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)

    def create_token_ids(words):
        toks = []
        for w in set(words):
            fallback_id = None
            fallback_token = None
            norm = _normalize_token_string(w)
            candidates = vocab_norm_map.get(norm, []) or [str(w).lstrip()]
            for cand in candidates:
                tokenized = _tokenizer.tokenize(cand)
                if len(tokenized) == 1:
                    toks.append(_tokenizer.convert_tokens_to_ids(tokenized[0]))
                elif len(tokenized) > 1 and fallback_id is None:
                    fallback_token = tokenized[0]
                    fallback_id = _tokenizer.convert_tokens_to_ids(tokenized[0])
            if fallback_id is not None and not any(
                _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(c)) == [fallback_token]
                for c in candidates
            ):
                first_token_str = _tokenizer.convert_tokens_to_string([fallback_token]).strip()
                if len(first_token_str) / max(len(str(w)), 1) < 0.5:
                    logger.warning(f"word '{w}' tokenized into multiple tokens; first token '{first_token_str}' is less than 50% of the word.")
                else:
                    logger.warning(f"word '{w}' tokenized into multiple tokens; using first token only.")
                toks.append(fallback_id)
        return list(set(toks))

    targets_ids = {g: create_token_ids(ws) for g, ws in targets.items()}
    for g, ids in targets_ids.items():
        if not ids:
            raise ValueError(f"Target group '{g}' has no valid token ids; check your target words and tokenizer.")

    # Determine dataset class column
    if dataset_class_col is None:
        if "label_class" in df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in df.columns:
            dataset_class_col = "factual_id"
        else:
            dataset_class_col = None

    # Group probabilities by dataset class: {dataset_class: {class_name: [probs]}}
    probs_by_dataset = defaultdict(lambda: defaultdict(list))
    rows = df.to_dict("records")
    n_batches = (len(rows) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]

            if eval_kind == "clm_next_token":
                # Decoder/CLM: use next-token logits only (no MLM head)
                prefixes = [r[key_text].split('[MASK]')[0] for r in batch]
                valid = [(i, p) for i, p in enumerate(prefixes) if p.strip()]
                if not valid:
                    continue
                idxs, prefix_texts = zip(*valid)
                inputs = tokenizer(list(prefix_texts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # CLM next-token logits
                probs = torch.softmax(logits, dim=-1)
                example_probs = {i: probs[j] for j, i in enumerate(idxs)}
            elif eval_kind == "seq2seq_encoder_mlm":
                texts = [str(r[key_text]) for r in batch]
                probs = seq2seq_mlm_probs_at_mask(model, _tokenizer, texts, device)
                example_probs = {i: probs[i] for i in range(len(batch))}
            else:
                texts = [r[key_text].replace('[MASK]', tokenizer.mask_token) for r in batch]
                inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                mask_idxs = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
                if len(mask_idxs) == 0:
                    continue
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                example_probs = {}
                for b_idx, pos in mask_idxs.tolist():
                    example_probs[b_idx] = probs[b_idx, pos, :]

            for b_idx, row in enumerate(batch):
                if b_idx not in example_probs:
                    continue
                p = example_probs[b_idx]
                
                # Get dataset class for this row
                dataset_class = None
                if dataset_class_col:
                    dataset_class = row.get(dataset_class_col)
                
                # Compute probabilities for all classes
                for g, ids in targets_ids.items():
                    if ids:
                        prob = float(p[ids].sum())  # Sum multiple tokens per class
                        if dataset_class is not None:
                            probs_by_dataset[dataset_class][g].append(prob)
                        else:
                            # Fallback: use class name as dataset identifier
                            probs_by_dataset[g][g].append(prob)

    # Compute means per dataset class
    probs_by_dataset_means = {}
    for dataset_class, class_probs in probs_by_dataset.items():
        probs_by_dataset_means[dataset_class] = {
            class_name: float(np.mean(probs)) if probs else 0.0
            for class_name, probs in class_probs.items()
        }

    # Return probs_by_dataset structure; selection filtering is done in evaluate_base_model
    if probs_by_dataset_means:
        means_str = ", ".join(
            f"{ds_cls}: {', '.join(f'{cls}={prob:.4f}' for cls, prob in sorted(cls_probs.items()))}"
            for ds_cls, cls_probs in sorted(probs_by_dataset_means.items())
        )
        logger.debug(f"Decoder eval probability means by dataset: {means_str}")
        return probs_by_dataset_means
    
    raise ValueError("No valid group probabilities computed; check your data and targets.")


_token_cache = {}


def compute_probability_shift_score_mlm(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    dataset_class_col=None,
):
    """
    Compute probabilities for all classes evaluated on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        df: DataFrame with evaluation data
        targets: Dict mapping class_name -> list of tokens
        key_text: Column name for masked text
        batch_size: Batch size for evaluation
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, uses "label_class" if available, else "factual_id".

    Returns:
        Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
    """
    model.eval()
    device = model.device
    if is_decoder_only_model(model):
        return compute_probability_shift_score_clm(model, tokenizer, df, targets, key_text, batch_size, dataset_class_col)

    eval_kind = prediction_eval_kind(model)
    if eval_kind == "seq2seq_encoder_mlm":
        return compute_probability_shift_score_clm(
            model, tokenizer, df, targets, key_text, batch_size, dataset_class_col
        )

    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)

    def create_token_ids(words):
        cache_id = (tokenizer.name_or_path, hash(tuple(sorted(set(words)))))
        if cache_id in _token_cache:
            return _token_cache[cache_id]
        result = []
        seen = set()
        for w in set(words):
            norm = _normalize_token_string(w)
            candidates = vocab_norm_map.get(norm, []) or [str(w).lstrip()]
            for cand in candidates:
                ids = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(cand))
                if not ids:
                    continue
                tup = tuple(ids)
                if tup in seen:
                    continue
                seen.add(tup)
                result.append(list(ids))
        _token_cache[cache_id] = result
        return result

    targets_ids = {g: create_token_ids(ws) for g, ws in targets.items()}
    for g, id_lists in targets_ids.items():
        if not id_lists:
            raise ValueError(f"Target group '{g}' has no valid token ids; check your targets.")

    # Determine dataset class column
    if dataset_class_col is None:
        if "label_class" in df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in df.columns:
            dataset_class_col = "factual_id"
        else:
            dataset_class_col = None

    # Group probabilities by dataset class: {dataset_class: {class_name: [probs]}}
    probs_by_dataset = defaultdict(lambda: defaultdict(list))
    rows = df.to_dict("records")
    n_batches = (len(rows) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [r[key_text] for r in batch]
            targets_by_len = defaultdict(list)
            for g, id_lists in targets_ids.items():
                for ids in id_lists:
                    targets_by_len[len(ids)].append((g, ids))
            example_probs = defaultdict(dict)
            for k, g_and_ids in targets_by_len.items():
                expanded_texts = []
                example_map = []
                for b_idx, text in enumerate(texts):
                    if "[MASK]" not in text:
                        continue
                    masked_text = text.replace("[MASK]", " ".join([tokenizer.mask_token] * k))
                    expanded_texts.append(masked_text)
                    example_map.append(b_idx)
                if not expanded_texts:
                    continue
                inputs = tokenizer(expanded_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                for local_idx, b_idx in enumerate(example_map):
                    mask_positions = (inputs["input_ids"][local_idx] == tokenizer.mask_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
                    for g, ids in g_and_ids:
                        if len(ids) > len(mask_positions):
                            continue
                        logp = 0.0
                        for j, tok_id in enumerate(ids):
                            pos = mask_positions[j]
                            logp += torch.log(probs[local_idx, pos, tok_id] + 1e-12)
                        p = torch.exp(logp).item()
                        example_probs[b_idx][g] = example_probs[b_idx].get(g, 0.0) + p
            
            for b_idx, row in enumerate(batch):
                per_group_probs = example_probs.get(b_idx, {})
                
                # Get dataset class for this row
                dataset_class = None
                if dataset_class_col:
                    dataset_class = row.get(dataset_class_col)
                
                # Store probabilities grouped by dataset class
                for g, p in per_group_probs.items():
                    if dataset_class is not None:
                        probs_by_dataset[dataset_class][g].append(p)
                    else:
                        # Fallback: use class name as dataset identifier
                        probs_by_dataset[g][g].append(p)

    # Compute means per dataset class
    probs_by_dataset_means = {}
    for dataset_class, class_probs in probs_by_dataset.items():
        probs_by_dataset_means[dataset_class] = {
            class_name: float(np.mean(probs)) if probs else 0.0
            for class_name, probs in class_probs.items()
        }

    # Return probs_by_dataset structure; selection filtering is done in evaluate_base_model
    if probs_by_dataset_means:
        means_str = ", ".join(
            f"{ds_cls}: {', '.join(f'{cls}={prob:.4f}' for cls, prob in sorted(cls_probs.items()))}"
            for ds_cls, cls_probs in sorted(probs_by_dataset_means.items())
        )
        logger.debug(f"Decoder eval probability means by dataset: {means_str}")
        return probs_by_dataset_means
    
    raise ValueError("No valid group probabilities computed; check your data and targets.")


def _ids_for_text(tokenizer, text: str) -> List[int]:
    return tokenizer(str(text), add_special_tokens=False)["input_ids"]


def _continuation_ids_from_prefix(tokenizer, prefix: str, continuation: str) -> List[int]:
    prefix_ids = _ids_for_text(tokenizer, prefix)
    full_ids = _ids_for_text(tokenizer, prefix + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    full_ids = _ids_for_text(tokenizer, prefix + " " + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    return _ids_for_text(tokenizer, continuation)


def _limit_rhs_by_tokens(tokenizer, rhs: str, rhs_window: int) -> str:
    if rhs_window is None or rhs_window < 0:
        return rhs
    if rhs_window == 0:
        return ""
    ids = _ids_for_text(tokenizer, rhs)
    if len(ids) <= rhs_window:
        return rhs
    return tokenizer.decode(ids[:rhs_window], skip_special_tokens=True)


def _score_clm_continuation_logprob(model, tokenizer, prefix: str, continuation: str, device) -> float:
    prefix_ids = _ids_for_text(tokenizer, prefix)
    continuation_ids = _continuation_ids_from_prefix(tokenizer, prefix, continuation)
    if not prefix_ids or not continuation_ids:
        return float("-inf")
    input_ids = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    logprob = 0.0
    for offset, token_id in enumerate(continuation_ids):
        pred_pos = len(prefix_ids) + offset - 1
        logprob += float(log_probs[0, pred_pos, token_id].item())
    return logprob


def compute_probability_shift_score_clm_sequence(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    dataset_class_col=None,
    rhs_window: int = -1,
    use_row_wise: bool = False,
    return_per_row_df: bool = False,
    score_continuation=None,
):
    """
    Decoder-only sequence-cloze scoring.

    For each candidate, score log P(candidate + RHS_window | prefix), where prefix
    and RHS come from splitting key_text at [MASK]. Aggregates candidate-softmax
    probabilities so downstream probability-shift code can keep its existing shape.
    """
    model.eval()
    device = model.device
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    if dataset_class_col is None:
        if "label_class" in df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in df.columns:
            dataset_class_col = "factual_id"

    rows = df.to_dict("records")
    probs_by_dataset = defaultdict(lambda: defaultdict(list))
    per_row_records: List[Dict[str, object]] = []

    def _default_score(m, tok, prefix_text, token_text, rhs_text, dev):
        return _score_clm_continuation_logprob(m, tok, prefix_text, token_text + rhs_text, dev)

    score_fn = score_continuation or _default_score

    with torch.no_grad():
        for row in rows:
            text = str(row.get(key_text, ""))
            if "[MASK]" not in text:
                continue
            prefix, rhs = text.split("[MASK]", 1)
            rhs = _limit_rhs_by_tokens(_tokenizer, rhs, rhs_window)

            if use_row_wise:
                candidate_items = [
                    (str(row.get(DEFAULT_DATASET_CLASS_COL)), str(row.get(DEFAULT_FACTUAL_COL))),
                    (str(row.get(DEFAULT_OTHER_CLASS_COL)), str(row.get(DEFAULT_ALTERNATIVE_COL))),
                ]
            else:
                candidate_items = [
                    (str(class_name), str(token))
                    for class_name, tokens in targets.items()
                    for token in tokens
                    if token is not None
                ]
            candidate_items = [
                (cls, tok)
                for cls, tok in candidate_items
                if cls and cls != "None" and tok and tok != "None"
            ]
            if not candidate_items:
                continue

            scored_items = [
                (
                    cls,
                    score_fn(model, _tokenizer, prefix, token, rhs, device),
                )
                for cls, token in candidate_items
            ]
            finite_items = [(cls, logp) for cls, logp in scored_items if np.isfinite(logp)]
            if not finite_items:
                continue
            classes, values = zip(*finite_items)
            probs = torch.softmax(torch.tensor(values), dim=0).tolist()
            class_probs: Dict[str, float] = defaultdict(float)
            for cls, prob in zip(classes, probs):
                class_probs[cls] += float(prob)

            dataset_class = row.get(dataset_class_col) if dataset_class_col else None
            if dataset_class is None and use_row_wise:
                dataset_class = row.get(DEFAULT_DATASET_CLASS_COL)
            if dataset_class is None:
                for cls, prob in class_probs.items():
                    probs_by_dataset[cls][cls].append(prob)
            else:
                for cls, prob in class_probs.items():
                    probs_by_dataset[dataset_class][cls].append(prob)

            if return_per_row_df and use_row_wise:
                ds_cls = row.get(DEFAULT_DATASET_CLASS_COL)
                other_cls = row.get(DEFAULT_OTHER_CLASS_COL)
                per_row_records.append({
                    "masked": row.get(key_text),
                    "factual": row.get(DEFAULT_FACTUAL_COL),
                    "alternative": row.get(DEFAULT_ALTERNATIVE_COL),
                    "factual_id": ds_cls,
                    "alternative_id": other_cls,
                    "dataset_class": ds_cls,
                    "other_class": other_cls,
                    "P_factual": class_probs.get(str(ds_cls), 0.0),
                    "P_alternative": class_probs.get(str(other_cls), 0.0),
                    "score_kind": "clm_sequence_cloze_candidate_softmax",
                })

    probs_by_dataset_means = {
        dataset_class: {
            class_name: float(np.mean(probs)) if probs else 0.0
            for class_name, probs in class_probs.items()
        }
        for dataset_class, class_probs in probs_by_dataset.items()
    }
    if not probs_by_dataset_means:
        raise ValueError("No valid CLM sequence-cloze probabilities computed; check data, targets, and [MASK].")
    if return_per_row_df:
        return probs_by_dataset_means, pd.DataFrame(per_row_records)
    return probs_by_dataset_means


def evaluate_probability_shift_score(
    model,
    tokenizer,
    targets,
    eval_data_df,
    key_text="masked",
    dataset_class_col=None,
    use_row_wise: bool = False,
    return_per_row_df: bool = False,
    objective: str = "auto",
    rhs_window: int = -1,
) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], pd.DataFrame]]:
    """
    Compute probabilities for all classes on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        targets: Dict mapping class_name -> list of tokens (ignored if use_row_wise=True)
        eval_data_df: DataFrame with evaluation data (must have factual/alternative if use_row_wise=True)
        key_text: Column name for masked text
        dataset_class_col: Column name identifying dataset class (ignored if use_row_wise=True)
        use_row_wise: If True, compute P(factual) and P(alternative) per row by dataset class.
        return_per_row_df: If True and use_row_wise, return (result_dict, per_row_DataFrame) for CSV export.

    Returns:
        Dict or (Dict, DataFrame): {dataset_class: {class_name: prob, ...}}. If return_per_row_df and use_row_wise,
        returns (dict, DataFrame) with columns masked, factual, alternative, factual_id, alternative_id,
        dataset_class, other_class, P_factual, P_alternative.
    """
    if objective == "auto":
        objective = prediction_eval_kind(model)
    if objective == "clm_sequence_cloze":
        return compute_probability_shift_score_clm_sequence(
            model,
            tokenizer,
            eval_data_df,
            targets=targets,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
            use_row_wise=use_row_wise,
            return_per_row_df=return_per_row_df,
            rhs_window=rhs_window,
        )
    if objective == "seq2seq_decoder" or objective == "seq2seq_decoder_sequence_cloze":
        include_span_sentinels = objective == "seq2seq_decoder_sequence_cloze"
        return compute_probability_shift_score_clm_sequence(
            model,
            tokenizer,
            eval_data_df,
            targets=targets,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
            use_row_wise=use_row_wise,
            return_per_row_df=return_per_row_df,
            rhs_window=rhs_window,
            score_continuation=lambda m, tok, prefix, token, rhs, dev: score_seq2seq_continuation_logprob(
                m, tok, prefix, token, dev, rhs=rhs, include_span_sentinels=include_span_sentinels
            ),
        )
    if objective == "seq2seq_encoder_mlm":
        if use_row_wise:
            return compute_probability_shift_score_row_wise(
                model,
                tokenizer,
                eval_data_df,
                key_text=key_text,
                factual_col=DEFAULT_FACTUAL_COL,
                alternative_col=DEFAULT_ALTERNATIVE_COL,
                dataset_class_col=DEFAULT_DATASET_CLASS_COL,
                other_class_col=DEFAULT_OTHER_CLASS_COL,
                return_per_row_df=return_per_row_df,
            )
        return compute_probability_shift_score_clm(
            model,
            tokenizer,
            eval_data_df,
            targets=targets,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
        )
    if use_row_wise:
        out = compute_probability_shift_score_row_wise(
            model,
            tokenizer,
            eval_data_df,
            key_text=key_text,
            factual_col=DEFAULT_FACTUAL_COL,
            alternative_col=DEFAULT_ALTERNATIVE_COL,
            dataset_class_col=DEFAULT_DATASET_CLASS_COL,
            other_class_col=DEFAULT_OTHER_CLASS_COL,
            return_per_row_df=return_per_row_df,
        )
        return out
    return compute_probability_shift_score_mlm(
        model, tokenizer, eval_data_df, targets=targets, key_text=key_text,
        dataset_class_col=dataset_class_col
    )


__all__ = [
    "annotate_text_probability_rows",
    "compute_probability_shift_score_row_wise",
    "compute_probability_shift_score_mlm",
    "compute_probability_shift_score_clm",
    "compute_probability_shift_score_clm_sequence",
    "compute_lms",
    "evaluate_probability_shift_score",
]
