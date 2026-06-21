"""
Train and save AutoModelForSequenceClassification for use as base model in GRADIEND.

Saves to experiment_dir/classification_head with config_classification_head.json marker
so resolve_model_path uses this path when present.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from gradiend.trainer.text.classification.data import tokenize_for_classification, _normalize_text_value
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

CONFIG_CLASSIFICATION_HEAD = "config_classification_head.json"


class _ClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        label2id: Dict[Union[str, int], int],
        text_col: str = "text",
        label_col: str = "label",
        text_a_col: Optional[str] = None,
        text_b_col: Optional[str] = None,
        max_length: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col
        self.text_a_col = text_a_col
        self.text_b_col = text_b_col
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = _normalize_text_value(row[self.text_col])
        label_id = self.label2id.get(str(row[self.label_col]), self.label2id.get(row[self.label_col], 0))
        if self.text_a_col and self.text_b_col and self.text_a_col in row and self.text_b_col in row:
            text_or_pair = _normalize_text_value((row[self.text_a_col], row[self.text_b_col]))
        else:
            text_or_pair = text
        return tokenize_for_classification(
            self.tokenizer, text_or_pair, label_id, max_length=self.max_length, device=None
        )


def train_classification_head(
    base_model: str,
    train_df: pd.DataFrame,
    output_path: str,
    *,
    text_col: str = "text",
    label_col: str = "label",
    text_a_col: Optional[str] = None,
    text_b_col: Optional[str] = None,
    split_col: Optional[str] = None,
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    batch_size: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    max_length: int = 512,
    trust_remote_code: bool = False,
    device: Optional[torch.device] = None,
    use_cache: bool = False,
) -> str:
    """
    Train AutoModelForSequenceClassification on (text, label) data and save to output_path.

    When split_col is provided and present in train_df, use rows with split=="train" for
    training and split=="validation" for evaluation each epoch.

    Args:
        base_model: HuggingFace model id or path.
        train_df: DataFrame with text_col and label_col; optionally text_a_col/text_b_col for
            sequence-pair classification and split_col for train/val.
        output_path: Directory to save the trained model.
        split_col: If set and in train_df, filter to train split for training and eval on validation.
        num_labels: Number of classes. Inferred from data if None.
        id2label: Optional mapping id -> label name.
        label2id: Optional mapping label name -> id. Inferred from data if None with id2label.
        device: Device for training. Default: cuda if available.

    Returns:
        output_path
    """
    if text_col not in train_df.columns or label_col not in train_df.columns:
        raise ValueError(f"train_df must have columns {text_col!r} and {label_col!r}")

    if split_col is not None and split_col in train_df.columns:
        train_sub = train_df[
            train_df[split_col].astype(str).str.lower() == "train"
        ].copy()
        val_df = train_df[
            train_df[split_col].astype(str).str.lower() == "validation"
        ].copy()
        if len(train_sub) == 0:
            train_sub = train_df.copy()
            val_df = None
        else:
            val_df = val_df if len(val_df) > 0 else None
    else:
        train_sub = train_df.copy()
        val_df = None

    # Optional disk-level cache: when enabled and a classification head already
    # exists at output_path, skip re-training and reuse the saved model.
    if use_cache and os.path.isdir(output_path):
        marker = os.path.join(output_path, CONFIG_CLASSIFICATION_HEAD)
        model_files = [
            os.path.join(output_path, "pytorch_model.bin"),
            os.path.join(output_path, "model.safetensors"),
        ]
        if os.path.isfile(marker) and any(os.path.isfile(mf) for mf in model_files):
            logger.info(
                "Classification head already exists at %s, skipping training (use_cache=True).",
                output_path,
            )
            return output_path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique = train_sub[label_col].dropna().astype(str).unique().tolist()
    if id2label is not None:
        label2id = label2id or {v: k for k, v in id2label.items()}
        n = len(id2label)
    elif label2id is not None:
        id2label = id2label or {v: k for k, v in label2id.items()}
        n = len(label2id)
    else:
        unique = sorted(set(unique))
        label2id = {str(l): i for i, l in enumerate(unique)}
        id2label = {i: l for l, i in label2id.items()}
        n = len(label2id)

    if num_labels is not None and num_labels != n:
        n = num_labels

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=n,
        id2label={str(k): v for k, v in id2label.items()},
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.train()

    train_dataset = _ClassificationDataset(
        train_sub,
        tokenizer,
        label2id,
        text_col=text_col,
        label_col=label_col,
        text_a_col=text_a_col,
        text_b_col=text_b_col,
        max_length=max_length,
    )
    val_dataset = (
        _ClassificationDataset(
            val_df,
            tokenizer,
            label2id,
            text_col=text_col,
            label_col=label_col,
            text_a_col=text_a_col,
            text_b_col=text_b_col,
            max_length=max_length,
        )
        if val_df is not None and len(val_df) > 0
        else None
    )
    def _collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        )
        if val_dataset is not None
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_per_class: Dict[int, int] = {i: 0 for i in range(n)}
        count_per_class: Dict[int, int] = {i: 0 for i in range(n)}
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            with torch.no_grad():
                logits = outputs.logits.detach()
                preds = torch.argmax(logits, dim=-1)
                labels = batch["labels"].view(-1)
                for cls_id in range(n):
                    mask = labels == cls_id
                    if mask.any():
                        count_per_class[cls_id] += int(mask.sum().item())
                        correct_per_class[cls_id] += int((preds[mask] == labels[mask]).sum().item())

        avg_loss = total_loss / max(len(train_loader), 1)
        overall_correct = sum(correct_per_class.values())
        overall_count = max(sum(count_per_class.values()), 1)
        overall_acc = overall_correct / overall_count
        classwise_acc = {
            id2label.get(i, str(i)): (
                correct_per_class[i] / count_per_class[i] if count_per_class[i] > 0 else 0.0
            )
            for i in range(n)
        }
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item() * batch["labels"].size(0)
                    preds = torch.argmax(outputs.logits, dim=-1).view(-1)
                    labels = batch["labels"].view(-1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
            val_loss = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            logger.info(
                "Classification head epoch %d: train_loss=%.4f, train_acc=%.4f, classwise_acc=%s, val_loss=%.4f, val_acc=%.4f",
                epoch + 1,
                avg_loss,
                overall_acc,
                classwise_acc,
                val_loss,
                val_acc,
            )
        else:
            logger.info(
                "Classification head epoch %d: loss=%.4f, overall_acc=%.4f, classwise_acc=%s",
                epoch + 1,
                avg_loss,
                overall_acc,
                classwise_acc,
            )

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    meta = {
        "num_labels": n,
        "id2label": {str(k): v for k, v in id2label.items()},
        "base_model": base_model,
    }
    with open(os.path.join(output_path, CONFIG_CLASSIFICATION_HEAD), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved classification head to %s", output_path)
    return output_path
