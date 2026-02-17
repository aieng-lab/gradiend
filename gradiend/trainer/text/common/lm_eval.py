"""
Language-model evaluation for text: MLM and CLM (perplexity, accuracy, etc.).

Used by prediction and classification decoder evaluation (e.g. compute_lms in decoder_eval_utils).
"""

import json
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from gradiend.trainer.text.common.loading import InstructTokenizerWrapper
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _calculate_classification_metrics(
    true_labels: List[str],
    predicted_labels: List[str],
    correct_predictions: int,
    total_predictions: int,
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score from labels."""
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
    precision = precision_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    recall = recall_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _save_results(
    result: Dict[str, Any],
    stats: Optional[pd.DataFrame],
    file: Optional[str],
) -> None:
    """Save evaluation results to files."""
    if file:
        if stats is not None:
            stats.to_csv(file + ".csv", index=False)
        with open(file + ".json", "w+", encoding="utf8") as f:
            json.dump(result, f, indent=2)


def _select_clm_target_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
) -> List[Optional[int]]:
    """Select target positions for CLM evaluation (random position in second half)."""
    target_positions = []
    for i in range(batch_size):
        token_positions = torch.where(attention_mask[i] != 0)[0].tolist()
        if len(token_positions) > 1:
            chosen_pos = random.choice(
                token_positions[len(token_positions) // 2 :]
            )
            target_positions.append(chosen_pos)
        else:
            target_positions.append(None)
    return target_positions


def _select_clm_instruction_target_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    tokenizer: InstructTokenizerWrapper,
) -> List[Optional[int]]:
    """Select target positions for CLM instruction evaluation."""
    end_header_token_id = tokenizer.tokenizer.convert_tokens_to_ids(
        tokenizer.tokenizer.END
    )
    target_positions = []
    for i in range(batch_size):
        token_positions = torch.where(attention_mask[i] != 0)[0].tolist()
        ids = input_ids[i].tolist()
        matches = [j for j, token_id in enumerate(ids) if token_id == end_header_token_id]
        if len(matches) >= 2:
            second_last_index = matches[-2]
            start_of_user_prompt = second_last_index + 1
        else:
            start_of_user_prompt = 0
        remaining_token_count = len(token_positions) - start_of_user_prompt
        if remaining_token_count > 1:
            chosen_pos = (
                random.choice(
                    token_positions[
                        start_of_user_prompt + remaining_token_count // 2 :
                    ]
                )
                - 1
            )
            target_positions.append(chosen_pos)
        else:
            target_positions.append(None)
    return target_positions


def _select_mlm_target_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    tokenizer: Any,
) -> Tuple[List[List[int]], torch.Tensor]:
    """Select and mask target positions for MLM evaluation."""
    mask_positions_batch = []
    masked_input_ids_batch = input_ids.clone()
    
    for i in range(batch_size):
        token_positions = torch.where(
            (input_ids[i] != tokenizer.all_special_ids) & (attention_mask[i] != 0)
        )[0].tolist()
        num_to_mask = max(1, int(0.15 * len(token_positions)))
        mask_positions = random.sample(token_positions, num_to_mask)
        masked_input_ids_batch[i, mask_positions] = tokenizer.mask_token_id
        mask_positions_batch.append(mask_positions)
    
    return mask_positions_batch, masked_input_ids_batch


def _process_batch_for_clm(
    model: Any,
    tokenizer: Any,
    batch_sentences: List[str],
    target_positions: List[Optional[int]],
    device: torch.device,
    ignore_ids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], int, int, List[str], List[str]]:
    """Process a batch for CLM evaluation and extract predictions."""
    batch_tokenized_input = tokenizer(
        batch_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    )
    input_ids = batch_tokenized_input["input_ids"].to(device)
    attention_mask = batch_tokenized_input["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = logits.argmax(dim=-1)
    
    stats_data = []
    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    predicted_labels = []
    
    for i in range(len(batch_sentences)):
        target_pos = target_positions[i]
        if target_pos is not None:
            predicted_token = tokenizer.decode(
                [predictions[i, target_pos]], skip_special_tokens=True
            )
            # For instruction tokenizer, true token is at target_pos + 1
            true_pos = target_pos + 1 if isinstance(tokenizer, InstructTokenizerWrapper) else target_pos
            true_token = tokenizer.decode(
                [input_ids[i, true_pos]], skip_special_tokens=True
            )
            correct = predicted_token.lower() == true_token.lower()
            if correct:
                correct_predictions += 1
            total_predictions += 1
            true_labels.append(true_token.lower())
            predicted_labels.append(predicted_token.lower())
            stats_data.append({
                "sentence": batch_sentences[i],
                "token_index": target_pos,
                "true": true_token,
                "predicted": predicted_token,
                "correct": correct,
                "score": logits[i, target_pos].max().item(),
            })
    
    return stats_data, correct_predictions, total_predictions, true_labels, predicted_labels


def _process_batch_for_mlm(
    model: Any,
    tokenizer: Any,
    batch_sentences: List[str],
    mask_positions_batch: List[List[int]],
    masked_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    original_input_ids: torch.Tensor,
    device: torch.device,
    ignore_ids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], int, int, List[str], List[str]]:
    """Process a batch for MLM evaluation and extract predictions."""
    with torch.no_grad():
        outputs = model(
            input_ids=masked_input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = logits.argmax(dim=-1)
    
    stats_data = []
    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    predicted_labels = []
    
    for i in range(len(batch_sentences)):
        sentence = batch_sentences[i]
        original_tokens = original_input_ids[i]
        predicted_tokens_batch = predictions[i]
        mask_positions = mask_positions_batch[i]
        logits_for_sentence = logits[i]
        for mask_position in mask_positions:
            token_id = original_tokens[mask_position].item()
            if ignore_ids and token_id in ignore_ids:
                continue
            predicted_token = tokenizer.decode(
                [predicted_tokens_batch[mask_position]], skip_special_tokens=True
            )
            original_token = tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            correct = predicted_token.lower() == original_token.lower()
            correct_predictions += int(correct)
            total_predictions += 1
            true_labels.append(original_token.lower())
            predicted_labels.append(predicted_token.lower())
            stats_data.append({
                "sentence": sentence,
                "token_index": mask_position,
                "true": original_token,
                "predicted": predicted_token,
                "correct": correct,
                "score": logits_for_sentence[mask_position].max().item(),
            })
    
    return stats_data, correct_predictions, total_predictions, true_labels, predicted_labels


def _evaluate_with_common_workflow(
    model: Any,
    tokenizer: Any,
    text_data: Union[List[str], Any],
    batch_size: int,
    process_batch_fn: Callable,
    cleanup_fn: Optional[Callable[[torch.device], None]] = None,
    ignore: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Common workflow for CLM/MLM evaluation."""
    random.seed(42)
    model.eval()
    device = model.device
    
    ignore_ids: Optional[set] = None
    if ignore:
        ignore_ids = set()
        for ig_text in ignore:
            ig_tokens = tokenizer.encode(ig_text, add_special_tokens=False)
            ignore_ids.update(ig_tokens)
    
    correct_predictions = 0
    total_predictions = 0
    true_labels: List[str] = []
    predicted_labels: List[str] = []
    stats_data: List[Dict[str, Any]] = []
    start = time.time()
    n = len(text_data)

    for start_idx in tqdm(range(0, n, batch_size), desc="LMS", total=max(1, (n + batch_size - 1) // batch_size), leave=True):
        end_idx = min(start_idx + batch_size, n)
        batch_sentences = text_data[start_idx:end_idx]
        if hasattr(batch_sentences, "tolist"):
            batch_sentences = batch_sentences.tolist()

        logger.debug(f"Processing batch {start_idx + 1}-{end_idx}/{n}")
        
        batch_stats, batch_correct, batch_total, batch_true, batch_pred = process_batch_fn(
            model, tokenizer, batch_sentences, ignore_ids, device
        )
        
        stats_data.extend(batch_stats)
        correct_predictions += batch_correct
        total_predictions += batch_total
        true_labels.extend(batch_true)
        predicted_labels.extend(batch_pred)
        
        if cleanup_fn:
            cleanup_fn(device)
    
    stats = pd.DataFrame(stats_data)
    metrics = _calculate_classification_metrics(
        true_labels, predicted_labels, correct_predictions, total_predictions
    )
    logger.debug(f"Evaluated {n} sentences in {time.time() - start:.2f} seconds")
    logger.debug("Accuracy:", metrics["accuracy"], "Precision:", metrics["precision"], 
                 "Recall:", metrics["recall"], "F1 Score:", metrics["f1"])
    
    return metrics, stats


def evaluate_clm_perplexity(
    model: Any,
    tokenizer: Any,
    text_data: Union[List[str], Any],
    file: Optional[str] = None,
    batch_size: int = 32,
    ignore: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate CLM perplexity over text_data. Returns dict with lms, perplexity, total_log_likelihood, total_tokens."""
    logger.debug("Evaluating CLM Perplexity")
    ignore = ignore or []
    model.eval()
    device = model.device
    total_log_likelihood = 0.0
    total_token_count = 0

    ignore_ids: set = set()
    if ignore:
        for ig_text in ignore:
            ig_tokens = tokenizer.encode(ig_text, add_special_tokens=False)
            ignore_ids.update(ig_tokens)

    start = time.time()
    n = len(text_data)

    for start_idx in tqdm(range(0, n, batch_size), desc="LMS (CLM)", total=(n + batch_size - 1) // batch_size, leave=True):
        end_idx = min(start_idx + batch_size, n)
        batch_sentences = text_data[start_idx:end_idx]
        if hasattr(batch_sentences, "tolist"):
            batch_sentences = batch_sentences.tolist()

        logger.debug(f"Processing batch {start_idx + 1}-{end_idx}/{n}")

        encoded = tokenizer(
            batch_sentences, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if ignore_ids:
            mask_ignore = torch.isin(
                labels, torch.tensor(list(ignore_ids), device=labels.device)
            )
            labels[mask_ignore] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )

        non_ignored_tokens = (labels != -100).sum().item()
        total_log_likelihood += loss.item()
        total_token_count += non_ignored_tokens

    avg_neg_log_likelihood = total_log_likelihood / total_token_count
    perplexity = math.exp(avg_neg_log_likelihood)
    logger.debug(f"Evaluated {n} sentences in {time.time() - start:.2f}s")
    logger.debug(f"Perplexity: {perplexity:.2f}")

    lms = 1 / (1 + avg_neg_log_likelihood)
    result = {
        "lms": lms,
        "perplexity": perplexity,
        "total_log_likelihood": total_log_likelihood,
        "total_tokens": total_token_count,
    }
    if file:
        with open(file + ".json", "w+", encoding="utf8") as f:
            json.dump(result, f, indent=2)
    return result


def evaluate_clm(
    model: Any,
    tokenizer: Any,
    text_data: Union[List[str], Any],
    file: Optional[str] = None,
    batch_size: int = 128,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate CLM next-token accuracy/precision/recall/F1. Returns (result dict, stats DataFrame)."""
    if isinstance(tokenizer, InstructTokenizerWrapper):
        return evaluate_clm_instruction(
            model, tokenizer, text_data, file=file, batch_size=32
        )
    
    def process_clm_batch(model, tokenizer, batch_sentences, ignore_ids, device):
        target_positions = None  # Will be set after tokenization
        batch_tokenized_input = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = batch_tokenized_input["input_ids"].to(device)
        attention_mask = batch_tokenized_input["attention_mask"].to(device)
        target_positions = _select_clm_target_positions(input_ids, attention_mask, len(batch_sentences))
        return _process_batch_for_clm(model, tokenizer, batch_sentences, target_positions, device, ignore_ids)
    
    result, stats = _evaluate_with_common_workflow(
        model, tokenizer, text_data, batch_size,
        process_batch_fn=process_clm_batch,
    )
    
    _save_results(result, stats, file)
    return result, stats


def evaluate_clm_instruction(
    model: Any,
    tokenizer: InstructTokenizerWrapper,
    text_data: Union[List[str], Any],
    file: Optional[str] = None,
    verbose: bool = True,
    batch_size: int = 1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate CLM with instruction tokenizer."""
    def process_clm_instruction_batch(model, tokenizer, batch_sentences, ignore_ids, device):
        batch_tokenized_input = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = batch_tokenized_input["input_ids"].to(device)
        attention_mask = batch_tokenized_input["attention_mask"].to(device)
        target_positions = _select_clm_instruction_target_positions(
            input_ids, attention_mask, len(batch_sentences), tokenizer
        )
        return _process_batch_for_clm(model, tokenizer, batch_sentences, target_positions, device, ignore_ids)
    
    def cleanup_cuda(device):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    result, stats = _evaluate_with_common_workflow(
        model, tokenizer, text_data, batch_size,
        process_batch_fn=process_clm_instruction_batch,
        cleanup_fn=cleanup_cuda,
    )
    
    _save_results(result, stats, file)
    return result, stats


def evaluate_mlm(
    model: Any,
    tokenizer: Any,
    text_data: Union[List[str], Any],
    file: Optional[str] = None,
    verbose: bool = True,
    batch_size: int = 32,
    ignore: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate MLM (masked LM) accuracy/precision/recall/F1. Returns (result dict, stats DataFrame)."""
    def process_mlm_batch(model, tokenizer, batch_sentences, ignore_ids, device):
        batch_tokenized_input = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=512,
        )
        input_ids = batch_tokenized_input["input_ids"].to(device)
        attention_mask = batch_tokenized_input["attention_mask"].to(device)
        mask_positions_batch, masked_input_ids_batch = _select_mlm_target_positions(
            input_ids, attention_mask, len(batch_sentences), tokenizer
        )
        return _process_batch_for_mlm(
            model, tokenizer, batch_sentences, mask_positions_batch,
            masked_input_ids_batch, attention_mask, input_ids, device, ignore_ids
        )
    
    def cleanup_cuda(device):
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    result, stats = _evaluate_with_common_workflow(
        model, tokenizer, text_data, batch_size,
        process_batch_fn=process_mlm_batch,
        cleanup_fn=cleanup_cuda,
        ignore=ignore,
    )
    
    result["lms"] = result["accuracy"]
    _save_results(result, stats, file)
    return result, stats
