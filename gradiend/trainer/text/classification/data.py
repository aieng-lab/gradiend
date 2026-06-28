"""
Unified classification data: build (factual, alternative) pairs for GRADIEND.

We expect a single set of column names (text, text_alternative, label, label_alternative).
Missing columns are auto-filled: same text => use text for both; missing alternative label
=> derive (binary: other class; multi-class: sample). Naming uses "alternative" throughout.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

# Internal unified schema (output of build_classification_training_df)
TEXT_FACTUAL = "text_factual"
TEXT_ALTERNATIVE = "text_alternative"
LABEL_FACTUAL = "label_factual"
LABEL_ALTERNATIVE = "label_alternative"
FACTUAL_ID = "factual_id"
ALTERNATIVE_ID = "alternative_id"
FACTUAL_CLS = "factual_cls"
ALTERNATIVE_CLS = "alternative_cls"


def _normalize_text_value(value: Any) -> Union[str, Tuple[str, str]]:
    """Preserve tuple/list pair inputs while normalizing leaf values to strings."""
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            return str(value[0]) if len(value) == 1 else ""
        return str(value[0]), str(value[1])
    return str(value)


def _resolve_label(
    label: Any,
    label2id: Dict[Union[str, int], int],
) -> int:
    """Map label (str or int) to integer id."""
    if isinstance(label, int):
        if label in label2id.values():
            return label
        if 0 <= label < len(label2id):
            return list(label2id.values())[label]
        return label
    return label2id.get(str(label), label2id.get(label, -1))


def _derive_alternative_label(
    factual_id: int,
    num_classes: int,
    id2label: Dict[int, str],
    rng: random.Random,
) -> int:
    """
    Derive alternative label when not provided.
    Binary (2 classes): return the other class. Multi-class: sample uniformly from others.
    """
    others = [i for i in range(num_classes) if i != factual_id]
    if not others:
        return factual_id
    if num_classes == 2:
        return others[0]
    return rng.choice(others)


def build_classification_training_df(
    config: TextClassificationConfig,
    df: pd.DataFrame,
    label2id: Optional[Dict[Union[str, int], int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[Union[str, int], int], Dict[int, str]]:
    """
    Build unified DataFrame with unified text, task-label, and semantic-class columns.

    **Auto-detection:** Only factual text and label are required (config.text_col, config.label_col).

    - If text_alternative_col is missing in df → use text_col (same text).
    - If label_alternative_col is missing → derive: binary = other class; multi = sample from others.

    Returns:
        (dataframe with internal schema, label2id, id2label)
    """
    tc = config.text_col
    ta = config.text_alternative_col
    lc = config.label_col
    la = config.label_alternative_col
    fc = config.factual_cls_col
    ac = config.alternative_cls_col
    split_col = config.split_col

    # Require at least factual text and label
    if tc not in df.columns:
        raise ValueError(
            f"DataFrame must have a column for factual text. "
            f"Default name is {tc!r}; provide it or set config.text_col to your column name."
        )
    if lc not in df.columns:
        raise ValueError(
            f"DataFrame must have a column for factual label. "
            f"Default name is {lc!r}; provide it or set config.label_col to your column name."
        )

    # Infer label2id / id2label from all labels present
    if label2id is None or id2label is None:
        labels_from_factual = df[lc].dropna().astype(str).unique().tolist()
        labels_from_alt = (
            df[la].dropna().astype(str).unique().tolist()
            if la in df.columns
            else []
        )
        unique_labels = sorted(set(labels_from_factual) | set(labels_from_alt))
        if config.id2label:
            id2label = {i: str(v) for i, v in config.id2label.items()}
            label2id = {v: k for k, v in id2label.items()}
        elif config.label2id:
            label2id = {str(k): v for k, v in config.label2id.items()}
            id2label = {v: k for k, v in label2id.items()}
        else:
            label2id = {str(l): i for i, l in enumerate(unique_labels)}
            id2label = {i: l for l, i in label2id.items()}
    num_classes = len(label2id)
    rng = random.Random(seed or getattr(config, "seed", 42))

    has_text_alternative = ta in df.columns
    has_label_alternative = la in df.columns
    has_factual_cls = fc in df.columns
    has_alternative_cls = ac in df.columns

    rows = []
    for _, row in df.iterrows():
        text_f = _normalize_text_value(row[tc])
        text_a = _normalize_text_value(row[ta]) if has_text_alternative else text_f
        lf = _resolve_label(row[lc], label2id)
        if has_label_alternative:
            la_id = _resolve_label(row[la], label2id)
        else:
            la_id = _derive_alternative_label(lf, num_classes, id2label, rng)

        factual_name = id2label.get(lf, str(lf))
        alternative_name = id2label.get(la_id, str(la_id))
        factual_cls = str(row[fc]) if has_factual_cls and pd.notna(row[fc]) else factual_name
        alternative_cls = str(row[ac]) if has_alternative_cls and pd.notna(row[ac]) else alternative_name
        rows.append({
            TEXT_FACTUAL: text_f,
            TEXT_ALTERNATIVE: text_a,
            LABEL_FACTUAL: lf,
            LABEL_ALTERNATIVE: la_id,
            FACTUAL_ID: factual_name,
            ALTERNATIVE_ID: alternative_name,
            FACTUAL_CLS: factual_cls,
            ALTERNATIVE_CLS: alternative_cls,
        })

    out = pd.DataFrame(rows)
    if split_col in df.columns:
        out[split_col] = df[split_col].values
    return out, label2id, id2label


def build_classification_head_df_from_pairs(
    combined_df: pd.DataFrame,
    target_classes: List[str],
    *,
    id2label: Optional[Dict[int, str]] = None,
    split_col: str = "split",
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Build a (text, label, split) DataFrame for classification-head training/eval
    from (factual, alternative) pairs. Use when data has same factual/alternative
    label (e.g. case #3: different texts, same label).

    Each row yields two (text, label) pairs:

    - (text_factual, factual_label)
    - (text_alternative, other_label)  where other_label is the other class in

      target_classes (or "not-{factual}" if only one target and we need a second).

    Preserves split so the result has train/validation/test for head training and eval.
    """
    if not target_classes:
        raise ValueError("target_classes must be non-empty for build_classification_head_df_from_pairs")
    target_classes = [str(c) for c in target_classes]
    other_by_factual: Dict[str, str] = {}
    if len(target_classes) >= 2:
        for i, c in enumerate(target_classes):
            other_by_factual[c] = target_classes[(i + 1) % len(target_classes)]
    else:
        other_by_factual[target_classes[0]] = f"not-{target_classes[0]}"

    rows: List[Dict[str, Any]] = []
    for _, row in combined_df.iterrows():
        text_f = _normalize_text_value(row[TEXT_FACTUAL])
        text_a = _normalize_text_value(row[TEXT_ALTERNATIVE])
        factual_label = row.get(LABEL_FACTUAL, "")
        if id2label:
            try:
                factual_lookup = int(factual_label)
            except (TypeError, ValueError):
                factual_lookup = factual_label
            factual_name = str(id2label.get(factual_lookup, factual_label))
        else:
            factual_name = str(factual_label)
        other_name = other_by_factual.get(factual_name, other_by_factual.get(target_classes[0], f"not-{factual_name}"))
        split_val = row.get(split_col, "train")
        rows.append({text_col: text_f, label_col: factual_name, split_col: split_val})
        rows.append({text_col: text_a, label_col: other_name, split_col: split_val})

    out = pd.DataFrame(rows)
    return out


def tokenize_for_classification(
    tokenizer: Any,
    text_or_pair: Union[str, Tuple[str, str]],
    label_id: int,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize one classification input (single text or pair) and add labels.

    For pairs, uses tokenizer(text_a, text_pair=text_b, ...) (BERT-style SEP).
    Returns dict with input_ids, attention_mask, labels.
    """
    if isinstance(text_or_pair, (list, tuple)) and len(text_or_pair) >= 2:
        text_a, text_b = str(text_or_pair[0]), str(text_or_pair[1])
        enc = tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    else:
        text = str(text_or_pair)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    labels = torch.full((enc["input_ids"].size(0),), label_id, dtype=torch.long)
    out = {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "labels": labels,
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


class ClassificationTrainingDataset(torch.utils.data.Dataset):
    """
    Dataset that yields items with factual and alternative classification inputs.

    Consumes the unified schema (text_factual, text_alternative, label_factual, label_alternative).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        label2id: Dict[Union[str, int], int],
        *,
        text_factual_col: str = TEXT_FACTUAL,
        text_alternative_col: str = TEXT_ALTERNATIVE,
        label_factual_col: str = LABEL_FACTUAL,
        label_alternative_col: str = LABEL_ALTERNATIVE,
        max_length: int = 512,
        device: Optional[torch.device] = None,
        pair: Optional[Tuple[str, str]] = None,
        feature_classes: Optional[List[str]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.text_factual_col = text_factual_col
        self.text_alternative_col = text_alternative_col
        self.label_factual_col = label_factual_col
        self.label_alternative_col = label_alternative_col
        self.max_length = max_length
        self.device = device
        self.pair = pair
        ordered_feature_classes = [str(v) for v in (feature_classes or [])]
        if FACTUAL_CLS in self.data.columns:
            observed = pd.concat([self.data[FACTUAL_CLS], self.data.get(ALTERNATIVE_CLS, pd.Series(dtype=object))]).dropna()
            for val in observed.astype(str).unique().tolist():
                if val not in ordered_feature_classes:
                    ordered_feature_classes.append(val)
        self.feature_class_to_id = {name: idx for idx, name in enumerate(ordered_feature_classes)}

    def __len__(self) -> int:
        return len(self.data)

    def _text_to_input(self, text: Union[str, Tuple[str, str]], label_id: int) -> Dict[str, torch.Tensor]:
        return tokenize_for_classification(
            self.tokenizer,
            text,
            label_id,
            max_length=self.max_length,
            device=self.device,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        text_f = _normalize_text_value(row[self.text_factual_col])
        text_a = _normalize_text_value(row[self.text_alternative_col])
        lf = int(row[self.label_factual_col])
        la = int(row[self.label_alternative_col])
        factual = self._text_to_input(text_f, lf)
        alternative = self._text_to_input(text_a, la)
        fid = str(row.get(FACTUAL_CLS, row.get(FACTUAL_ID, str(lf))))
        aid = str(row.get(ALTERNATIVE_CLS, row.get(ALTERNATIVE_ID, str(la))))
        if self.pair is not None:
            label_scalar = 1.0 if fid == self.pair[0] else (-1.0 if fid == self.pair[1] else 0.0)
        else:
            label_scalar = float(self.feature_class_to_id.get(fid, lf))
        return {
            "factual": factual,
            "alternative": alternative,
            "input_text": text_f,
            "label": label_scalar,
            FACTUAL_ID: fid,
            ALTERNATIVE_ID: aid,
            "feature_class_id": fid,
        }
