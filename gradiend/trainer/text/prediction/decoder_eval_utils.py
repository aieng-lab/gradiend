"""
Decoder evaluation helpers for text prediction (LMS, feature/target token scores).

Used by TextPredictionTrainer.evaluate_base_model. No dependency on
trainer.core.feature_definition to avoid circular imports.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

from gradiend.model.utils import is_decoder_only_model
from gradiend.util.logging import get_logger

from gradiend.trainer.text.common.lm_eval import evaluate_mlm, evaluate_clm_perplexity


logger = get_logger(__name__)


def _normalize_token_string(value: str) -> str:
    return str(value).lstrip().lower() if value is not None else ""


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


def compute_probability_shift_score_clm(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    data_class_col=None,
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
        data_class_col: DEPRECATED - kept for backward compatibility. If set, filters results
            to counterfactual probabilities only (for selection logic).
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, uses "label_class" if available, else "factual_id".

    Returns:
        If dataset_class_col is provided or can be inferred:
            Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
        Otherwise (backward compatibility):
            Dict[str, float]: {class_name: prob, ...} (filtered by data_class_col if provided)
    """
    model.eval()
    device = model.device
    is_decoder_only = 'ForMaskedLM' not in model.__class__.__name__
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
        for start in tqdm(range(0, len(rows), batch_size), desc="Decoder prob (CLM)", total=n_batches):
            batch = rows[start:start + batch_size]

            if is_decoder_only:
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

    # Always return probs_by_dataset structure
    # Backward compatibility filtering happens in evaluate_base_model
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
    data_class_col=None,
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
        data_class_col: DEPRECATED - kept for backward compatibility. If set, filters results
            to counterfactual probabilities only (for selection logic).
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, uses "label_class" if available, else "factual_id".

    Returns:
        If dataset_class_col is provided or can be inferred:
            Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
        Otherwise (backward compatibility):
            Dict[str, float]: {class_name: prob, ...} (filtered by data_class_col if provided)
    """
    model.eval()
    device = model.device
    if is_decoder_only_model(model):
        return compute_probability_shift_score_clm(model, tokenizer, df, targets, key_text, batch_size, data_class_col)

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
        for start in tqdm(range(0, len(rows), batch_size), desc="Decoder prob (MLM)", total=n_batches):
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

    # Always return probs_by_dataset structure
    # Backward compatibility filtering happens in evaluate_base_model
    if probs_by_dataset_means:
        means_str = ", ".join(
            f"{ds_cls}: {', '.join(f'{cls}={prob:.4f}' for cls, prob in sorted(cls_probs.items()))}"
            for ds_cls, cls_probs in sorted(probs_by_dataset_means.items())
        )
        logger.debug(f"Decoder eval probability means by dataset: {means_str}")
        return probs_by_dataset_means
    
    raise ValueError("No valid group probabilities computed; check your data and targets.")


def compute_lms(model, tokenizer, texts, ignore, max_texts=None, batch_size=32):
    """
    Compute language modeling score (LMS) on a sample of texts.
    
    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        texts: List of texts to evaluate.
        ignore: List of tokens to ignore during evaluation.
        max_texts: Max number of texts; None = use all.
        batch_size: Batch size for evaluation (used for both MLM and CLM). Default 32.
    
    Returns:
        Evaluation result dictionary.
    """
    limited = texts[:max_texts] if max_texts is not None else texts
    if is_decoder_only_model(tokenizer):
        return evaluate_clm_perplexity(model, tokenizer, limited, ignore=ignore, batch_size=batch_size)
    result, _ = evaluate_mlm(model, tokenizer, limited, verbose=False, ignore=ignore, batch_size=batch_size)
    return result


def evaluate_probability_shift_score(model, tokenizer, targets, eval_data_df, key_text='masked', data_class_col=None, dataset_class_col=None):
    """
    Generic feature score evaluation: computes probabilities for all classes on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        targets: Dict mapping class_name -> list of tokens
        eval_data_df: DataFrame with evaluation data
        key_text: Column name for masked text
        data_class_col: DEPRECATED - kept for backward compatibility. If set, returns filtered
            counterfactual probabilities (for selection logic).
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, auto-detects from dataframe columns.

    Returns:
        If dataset_class_col is provided or can be inferred:
            Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
        Otherwise (backward compatibility):
            Dict[str, float]: {class_name: prob, ...} (filtered by data_class_col if provided)

    Probabilities are taken from the model's forward only: for decoder/generative models
    we use next-token (CLM) logits at the position after the prefix; for MaskedLM we use
    logits at mask positions. Decoder analysis never uses a separate MLM head—callers must
    pass the CLM (base decoder) when using a decoder-only MLM-head model.
    """
    return compute_probability_shift_score_mlm(
        model, tokenizer, eval_data_df, targets=targets, key_text=key_text, 
        data_class_col=data_class_col, dataset_class_col=dataset_class_col
    )


__all__ = [
    "compute_probability_shift_score_mlm",
    "compute_probability_shift_score_clm",
    "compute_lms",
    "evaluate_probability_shift_score",
]
