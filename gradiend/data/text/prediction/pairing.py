"""Word-pair lookup for :class:`~gradiend.data.text.filter_config.PairedTextFilterConfig`.

Paired groups are expanded into ordinary per-class ``TextFilterConfig`` objects, so filtering, balancing and
splitting stay the generic per-class code. This module only builds the word lookup used to add the alternative
columns to the generated rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from gradiend.data.core import SplitGroupKey, split_dataframe_per_group
from gradiend.data.text.filter_config import PairedTextFilterConfig, TextFilterConfig


@dataclass(frozen=True)
class PairTable:
    """All paired words: ``(class id, lower-cased word) -> the pair, one word per class (in ids order)``."""

    ids: Tuple[str, ...]
    pair_of: Dict[Tuple[str, str], Tuple[str, ...]]

    def row_alternatives(self, class_id: str, labels: Sequence[str]) -> Optional[Dict[str, str]]:
        """Alternative columns (one per other class) for a row whose masked words are ``labels``.

        Returns ``None`` if the row has no unique alternative (several different pair words in one sentence).
        """
        found = {self.pair_of.get((class_id, label.lower())) for label in labels}
        if len(found) != 1 or None in found:
            return None
        (pair,) = found
        return {cid: word for cid, word in zip(self.ids, pair) if cid != class_id}


def expand_feature_targets(
    feature_targets: Sequence[Union[TextFilterConfig, PairedTextFilterConfig]],
) -> Tuple[List[TextFilterConfig], Optional[PairTable]]:
    """Expand paired groups into per-class configs and build the word-pair lookup.

    Without paired groups the configs are returned unchanged and the lookup is ``None``.

    Raises:
        TypeError: If an entry is neither a ``TextFilterConfig`` nor a ``PairedTextFilterConfig``.
        ValueError: If paired groups are mixed with plain configs, groups use different ids, or a word is in two
            different pairs.
    """
    for entry in feature_targets:
        if not isinstance(entry, (TextFilterConfig, PairedTextFilterConfig)):
            raise TypeError(f"feature_targets entries must be TextFilterConfig or PairedTextFilterConfig, got {type(entry).__name__}")
    groups = [entry for entry in feature_targets if isinstance(entry, PairedTextFilterConfig)]
    if not groups:
        return list(feature_targets), None
    if len(groups) != len(feature_targets):
        raise ValueError("PairedTextFilterConfig cannot be combined with plain TextFilterConfig in one creator")
    ids = groups[0].ids
    pair_of: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for group in groups:
        if set(group.ids) != set(ids):
            raise ValueError(f"All PairedTextFilterConfig groups need the same ids, got {group.ids!r} and {ids!r}")
        for pair in group.pairs:
            by_id = dict(zip(group.ids, pair))
            ordered = tuple(by_id[cid] for cid in ids)
            for cid, word in by_id.items():
                known = pair_of.setdefault((cid, word.lower()), ordered)
                if [w.lower() for w in known] != [w.lower() for w in ordered]:
                    raise ValueError(f"'{word}' in class '{cid}' is in two different pairs: {known!r} and {ordered!r}")
    return [cfg for group in groups for cfg in group.to_configs()], PairTable(ids=ids, pair_of=pair_of)


def split_by_pair(
    class_dfs: Dict[str, pd.DataFrame],
    ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    split_group_key: SplitGroupKey = None,
) -> None:
    """Vocabulary-held-out split in which a pair is one group: both words of a pair land in the same split.

    Same as ``split_dataframe_per_group(..., split_group_col="label")``, but the groups are shared across the
    classes instead of being formed per class. ``ids`` are the class ids; every class table has one column per id.
    Replaces the values of ``class_dfs`` in place.
    """
    names = [cid for cid, df in class_dfs.items() if len(df)]
    joint = pd.concat([class_dfs[cid].assign(_class=cid) for cid in names], ignore_index=True)
    joint["_pair"] = joint[list(ids)].astype(str).apply(lambda column: column.str.lower()).agg("|".join, axis=1)
    holder = {"all": joint}
    split_dataframe_per_group(
        holder, train_ratio, val_ratio, test_ratio, seed, split_group_col="_pair", split_group_key=split_group_key
    )
    for cid in names:
        part = holder["all"][holder["all"]["_class"] == cid]
        class_dfs[cid] = part.drop(columns=["_class", "_pair"]).reset_index(drop=True)
