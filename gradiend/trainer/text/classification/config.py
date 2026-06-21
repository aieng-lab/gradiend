"""
Configuration for TextClassificationTrainer.

Unified data schema: (text_factual, text_alternative, label_factual, label_alternative).
User provides a DataFrame with a subset of columns; we auto-detect and derive the rest.
Naming uses "alternative" (aligned with TextPrediction), not "counterfactual".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from gradiend.trainer.config import TrainerConfig


@dataclass
class TextClassificationConfig(TrainerConfig):
    """
    Configuration for TextClassificationTrainer.

    **Unified schema (internal):** Every row becomes (text_factual, text_alternative,
    label_factual, label_alternative). The user's DataFrame can be incomplete; we
    auto-detect and fill defaults.

    **Supported use cases (auto-detected from which columns exist):**

    - **Same text, different labels:** User provides `text` + `label` only.
      We set text_alternative = text. If only two classes in data, we set
      label_alternative = the other class; otherwise we sample from other classes.

    - **Different text, same label:** User provides `text`, `text_alternative`, `label`.
      We set label_alternative = label.

    - **Full explicit:** User provides `text`, `text_alternative`, `label`,
      `label_alternative` (and optionally `split`). No derivation.

    **Column name defaults** (user can override if their DF uses different names):

    - text_col: column for factual text (default "text").
    - text_alternative_col: column for alternative text (default "text_alternative").
      If missing in DataFrame, we use factual text (same text).
    - label_col: column for factual label (default "label").
    - label_alternative_col: column for alternative label (default "label_alternative").
      If missing, we derive: binary = other class; multi-class = sample from others.
    - split_col: train/validation/test (default "split").
    """

    run_id: Optional[str] = None
    data: Optional[Union[pd.DataFrame, str, Path]] = None

    # Column names (defaults; override only if your DataFrame uses different names)
    text_col: str = "text"
    text_alternative_col: str = "text_alternative"
    label_col: str = "label"
    label_alternative_col: str = "label_alternative"
    factual_cls_col: str = "factual_cls"
    alternative_cls_col: str = "alternative_cls"
    split_col: str = "split"

    target_classes: Optional[List[str]] = None
    num_labels: Optional[int] = None
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None

    n_features: int = 1

    # Decoder evaluation
    decoder_eval_restrict_to_target_classes: bool = True
    decoder_eval_export_row_wise_csv: bool = False
    decoder_lms_mode: Optional[str] = None
    """
    How to compute the decoder 'LMS' score for sequence classification.
    - "classification_accuracy": accuracy on non-target classes (>=4 classes).
    - "lm": LM score (MLM/CLM) when model supports it.
    - "both": average when both available.
    If None, tries "lm" then "classification_accuracy"; raises if neither works.
    """

    eval_neutral_data: Optional[Union[pd.DataFrame, str, Path]] = None
    eval_neutral_max_rows: Optional[int] = None
