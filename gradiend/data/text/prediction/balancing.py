"""Balancing helpers for text-prediction data creation."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd

from gradiend.data.core import balance_dataframe_per_target_with_floor
from gradiend.data.text import TextFilterConfig
from gradiend.data.text.prediction.ids import _class_id
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _apply_balance(
    class_dfs: Dict[str, pd.DataFrame],
    max_size_per_class: Optional[int],
    balance: Union[bool, str],
    min_rows_per_target_for_balance: int,
    feature_targets: List[TextFilterConfig],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """Apply non-destructive try balancing or strict per-target/per-class balancing."""
    if not class_dfs:
        return class_dfs

    config_by_id = {_class_id(cfg, i): cfg for i, cfg in enumerate(feature_targets)}
    class_ids = list(class_dfs.keys())
    weights = [
        config_by_id[cid].weight if cid in config_by_id else 1.0
        for cid in class_ids
    ]
    weights = [max(0.001, w) for w in weights]

    if balance == "strict":
        for cid in class_ids:
            df = class_dfs[cid]
            if df is None or df.empty or "label" not in df.columns:
                continue
            before = len(df)
            class_dfs[cid] = balance_dataframe_per_target_with_floor(
                df,
                target_col="label",
                min_rows_per_target=min_rows_per_target_for_balance,
                max_size=max_size_per_class,
                seed=seed,
            )
            after = len(class_dfs[cid])
            if after != before:
                logger.info(
                    "  %s: strict per-label balance %s -> %s rows (%s distinct labels, min_rows_per_target=%s)",
                    cid,
                    before,
                    after,
                    class_dfs[cid]["label"].nunique(),
                    min_rows_per_target_for_balance,
                )

    shuffled: Dict[str, pd.DataFrame] = {}
    for cid in class_ids:
        df = class_dfs[cid].copy()
        shuffled[cid] = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if balance == "strict":
        min_len = min(len(shuffled[cid]) for cid in class_ids)
        cap = min(min_len, max_size_per_class) if max_size_per_class is not None else min_len
        capped: Dict[str, pd.DataFrame] = {}
        for cid in class_ids:
            df = shuffled[cid]
            if len(df) <= cap:
                capped[cid] = df.reset_index(drop=True)
            else:
                # Re-balance down to cap; blind iloc[:cap] breaks per-target equality.
                capped[cid] = balance_dataframe_per_target_with_floor(
                    df,
                    target_col="label",
                    min_rows_per_target=1,
                    max_size=cap,
                    seed=seed,
                )
        return capped

    if balance == "try":
        cycle = []
        for i, cid in enumerate(class_ids):
            n = max(1, int(round(weights[i])))
            cycle.extend([cid] * n)
        if not cycle:
            cycle = class_ids

        indices = {cid: 0 for cid in class_ids}
        caps = {
            cid: min(len(shuffled[cid]), max_size_per_class or len(shuffled[cid]))
            for cid in class_ids
        }
        result_rows: Dict[str, List[dict]] = {cid: [] for cid in class_ids}
        exhausted = set()
        added_any = True
        while added_any:
            added_any = False
            for cid in cycle:
                if cid in exhausted:
                    continue
                if len(result_rows[cid]) >= caps[cid]:
                    exhausted.add(cid)
                    continue
                idx = indices[cid]
                if idx >= len(shuffled[cid]):
                    exhausted.add(cid)
                    continue
                row = shuffled[cid].iloc[idx].to_dict()
                result_rows[cid].append(row)
                indices[cid] = idx + 1
                added_any = True
                if len(result_rows[cid]) >= caps[cid]:
                    exhausted.add(cid)
            if exhausted and len(exhausted) == len(class_ids):
                break

        return {
            cid: pd.DataFrame(rows) if rows else pd.DataFrame()
            for cid, rows in result_rows.items()
        }

    return class_dfs

__all__ = ["_apply_balance"]
