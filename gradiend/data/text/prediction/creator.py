"""
TextPredictionDataCreator: build training and neutral datasets for text prediction.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from gradiend.data.core.base_loader import resolve_base_data
from gradiend.data.core.spacy_util import load_spacy_model
from gradiend.data.text import (
    SpacyTagSpec,
    TextFilterConfig,
    TextPreprocessConfig,
    iter_sentences_from_texts,
)
from gradiend.data.text.prediction.filter_engine import (
    filter_sentences_multi,
    mask_sentence,
    token_matches_tags,
)
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

def _save_data(
    path: str,
    fmt: Literal["csv", "parquet", "hf"],
    df: Optional[pd.DataFrame] = None,
    class_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Save data to path. For training with per_class and fmt hf, pass class_dfs to save as DatasetDict (subsets)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        data = df if df is not None else (_to_merged(class_dfs) if class_dfs else None)
        if data is not None:
            data.to_csv(path, index=False)
        logger.info("Wrote data to %s (csv)", path)
        return
    if fmt == "parquet":
        data = df if df is not None else (_to_merged(class_dfs) if class_dfs else None)
        if data is not None:
            data.to_parquet(path, index=False)
        logger.info("Wrote data to %s (parquet)", path)
        return
    if fmt == "hf":
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            logger.warning(
                "output_format='hf' requires 'datasets'. Install with: pip install datasets. Falling back to csv."
            )
            ext = path_obj.suffix.lower()
            if not ext or path_obj.suffix == "":
                path = str(path_obj.with_suffix(".csv"))
            _save_data(path, "csv", df=df, class_dfs=class_dfs)
            return
        if class_dfs:
            d = DatasetDict(
                {cid: Dataset.from_pandas(df_c, preserve_index=False) for cid, df_c in class_dfs.items()}
            )
            d.save_to_disk(path)
            logger.info("Wrote data to %s (HuggingFace DatasetDict, subsets per class)", path)
        else:
            data = df if df is not None else None
            if data is not None:
                Dataset.from_pandas(data, preserve_index=False).save_to_disk(path)
                logger.info("Wrote data to %s (HuggingFace Dataset)", path)


def _class_id(cfg: TextFilterConfig, index: int) -> str:
    """Get class id from config; use id if set, else first target, else index fallback."""
    if cfg.id is not None:
        return cfg.id
    first = next((t for t in cfg.targets or [] if isinstance(t, str)), None)
    return first if first is not None else f"_class_{index}"


class TextPredictionDataCreator:
    """Creates training and neutral data for text prediction from base corpora."""

    def __init__(
        self,
        base_data: Union[str, pd.DataFrame, List[str]],
        text_column: str = "text",
        base_max_size: Optional[int] = None,
        split: str = "train",
        hf_config: Optional[str] = None,
        trust_remote_code: bool = False,
        preprocess: Optional[TextPreprocessConfig] = None,
        spacy_model: Optional[str] = None,
        feature_targets: Optional[List[TextFilterConfig]] = None,
        seed: int = 42,
        download_if_missing: bool = True,
        output_dir: Optional[str] = None,
        training_basename: str = "training",
        neutral_basename: str = "neutral",
        output_format: Literal["csv", "parquet", "hf"] = "csv",
    ) -> None:
        """Initialize with shared config for both generate methods.

        Args:
            base_data: HF id, pandas df, csv path, or List[str].
            text_column: Column name for text (default "text").
            base_max_size: Cap on base data (after shuffle, before preprocessing).
            split: HF split (default "train").
            hf_config: HF dataset config/subset (e.g. "20220301.en" for wikipedia).
            trust_remote_code: Passed to load_dataset when base_data is HF id. Default False.
            preprocess: Optional TextPreprocessConfig.
            spacy_model: Spacy model name (e.g. "de_core_news_sm"); lazy-loaded.
            feature_targets: List of TextFilterConfig. Each config's id (or first target) names the class.
            seed: Random seed for shuffle and sampling.
            download_if_missing: If True, auto-download spacy model when not found.
            output_dir: If set, generate_training_data/generate_neutral_data write to this folder
                when output= is not passed. Default filenames: training_basename + ext, neutral_basename + ext.
            training_basename: Base name for training output (default "training"); extension from output_format.
            neutral_basename: Base name for neutral output (default "neutral").
            output_format: "csv" (default), "parquet", or "hf" (HuggingFace datasets; per_class saves as subsets).
                "hf" requires the datasets library; falls back to csv with a warning if not installed.
        """
        self.base_data = base_data
        self.text_column = text_column
        self.base_max_size = base_max_size
        self.split = split
        self.hf_config = hf_config
        self.trust_remote_code = trust_remote_code
        self.preprocess = preprocess
        self.spacy_model = spacy_model
        self.feature_targets = feature_targets or []
        self.seed = seed
        self.download_if_missing = download_if_missing
        self.output_dir = Path(output_dir) if output_dir else None
        self.training_basename = training_basename
        self.neutral_basename = neutral_basename
        self.output_format = output_format
        self._texts_cache: Optional[List[str]] = None

    def _resolve_output_path(self, name: Literal["training", "neutral"], explicit: Optional[str]) -> Optional[str]:
        """Resolve output path: explicit path, or output_dir + basename + extension."""
        if explicit is not None:
            return explicit
        if self.output_dir is None:
            return None
        basename = self.training_basename if name == "training" else self.neutral_basename
        if self.output_format == "csv":
            return str(self.output_dir / f"{basename}.csv")
        if self.output_format == "parquet":
            return str(self.output_dir / f"{basename}.parquet")
        # hf: directory, no extension
        return str(self.output_dir / basename)

    def _get_texts(
        self, base_override: Optional[Union[str, pd.DataFrame, List[str]]] = None
    ) -> List[str]:
        """Load base data as raw texts (no sentence splitting); cache when no override."""
        def _resolve(src: Union[str, pd.DataFrame, List[str]]) -> List[str]:
            is_hf_str = isinstance(src, str) and Path(src).suffix.lower() != ".csv"
            return resolve_base_data(
                src,
                text_column=self.text_column,
                max_size=self.base_max_size,
                split=self.split,
                seed=self.seed,
                hf_config=self.hf_config if is_hf_str else None,
                trust_remote_code=self.trust_remote_code if is_hf_str else False,
            )
        if base_override is not None:
            return _resolve(base_override)
        if self._texts_cache is not None:
            return self._texts_cache
        texts = _resolve(self.base_data)
        self._texts_cache = texts
        return texts

    def generate_training_data(
        self,
        max_size_per_class: Optional[int] = None,
        format: str = "per_class",
        split_name: str = "train",
        balance: Union[bool, str] = "try",
        output: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Generate training data by filtering and masking.

        Args:
            max_size_per_class: Cap per feature class.
            format: Return structure: "per_class" (dict), "unified", or "minimal".
            split_name: Value for split column when auto_split is not used (default "train").
            balance: "try" (default) attempt balance, fill with abundant; False no
                balancing; "strict" cap all to smallest. Uses TextFilterConfig.weight.
            output: If set, save the data to this path as CSV (unified table when
                format is "per_class", otherwise the returned DataFrame).
            train_ratio: Fraction of each class for train (default 0.8).
            val_ratio: Fraction of each class for validation (default 0.1).
            test_ratio: Fraction of each class for test (default 0.1). Must sum to 1.0 with train_ratio and val_ratio.

        Returns:
            Per format: dict of DataFrames, or single DataFrame.
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio, val_ratio, test_ratio must sum to 1.0")
        texts = self._get_texts()
        configs_with_ids = [
            (_class_id(cfg, i), cfg) for i, cfg in enumerate(self.feature_targets)
        ]
        config_by_id = {cid: cfg for cid, cfg in configs_with_ids}
        stream = iter_sentences_from_texts(
            texts,
            self.preprocess,
            self.spacy_model,
            download_if_missing=self.download_if_missing,
        )
        total_target = (
            len(self.feature_targets) * max_size_per_class
            if max_size_per_class is not None
            else None
        )
        filter_stats: Dict[str, int] = {}
        results_per_class, _ = filter_sentences_multi(
            stream,
            configs_with_ids,
            spacy_model=self.spacy_model,
            download_if_missing=self.download_if_missing,
            max_matches_per_class=max_size_per_class,
            total_target_overall=total_target,
            stats=filter_stats,
        )
        n_processed = filter_stats.get("sentences_processed", 0)
        class_dfs = {}
        stats_per_group = {cid: len(matches) for cid, matches in results_per_class.items()}
        total_so_far = sum(stats_per_group.values())
        success_rates = {}
        for class_id, matches in results_per_class.items():
            n = len(matches)
            rate = (n / n_processed * 100.0) if n_processed else 0.0
            success_rates[class_id] = rate
            cap_str = f"/{max_size_per_class}" if max_size_per_class is not None else ""
            tqdm.write(f"  {class_id}: {n}{cap_str} matches (success rate: {rate:.2f}%)")
            cfg = config_by_id[class_id]
            rows = []
            for sent, spans in matches:
                masked = mask_sentence(sent, spans, cfg.mask)
                labels = [m[2] for m in spans]
                label = labels[0] if labels else ""
                rows.append({
                    "text": sent,
                    "masked": masked,
                    "label": label,
                    "token_count": len(spans),
                })
            if not rows:
                logger.warning(f"No matches for class '{class_id}'")
                continue
            df = pd.DataFrame(rows)
            df["split"] = split_name
            df[class_id] = df["label"]
            class_dfs[class_id] = df

        if stats_per_group:
            tqdm.write(f"Training filter stats (instances per group): {stats_per_group}")
            if total_target is not None and total_target > 0:
                pct = 100.0 * total_so_far / total_target
                tqdm.write(f"Overall: {total_so_far}/{total_target} ({pct:.1f}%)")
            if success_rates:
                avg_rate = sum(success_rates.values()) / len(success_rates)
                tqdm.write(f"Success rate per class: {success_rates}")
                tqdm.write(f"Average success rate: {avg_rate:.2f}")

        if balance is not False:
            class_dfs = _apply_balance(
                class_dfs,
                max_size_per_class,
                balance,
                self.feature_targets,
                self.seed,
            )
        elif max_size_per_class is not None:
            for class_id in list(class_dfs.keys()):
                df = class_dfs[class_id]
                if len(df) > max_size_per_class:
                    class_dfs[class_id] = df.sample(
                        n=max_size_per_class, random_state=self.seed
                    ).reset_index(drop=True)

        _apply_auto_split(class_dfs, train_ratio, val_ratio, test_ratio, self.seed)

        if format == "per_class":
            result: Union[Dict[str, pd.DataFrame], pd.DataFrame] = class_dfs
            df_to_save = _to_merged(class_dfs)
        elif format == "minimal":
            result = _to_minimal(class_dfs)
            df_to_save = result
        elif format == "unified":
            result = _to_merged(class_dfs)
            df_to_save = result
        else:
            raise ValueError(f"format must be 'per_class', 'unified', or 'minimal'; got {format!r}")

        output = output or self._resolve_output_path("training", None)
        if output is not None:
            fmt = self.output_format
            if fmt == "hf":
                try:
                    from datasets import DatasetDict  # noqa: F401
                except ImportError:
                    fmt = "csv"
            _save_data(
                output,
                fmt,
                df=df_to_save if fmt != "hf" or not isinstance(result, dict) else None,
                class_dfs=class_dfs if fmt == "hf" and isinstance(result, dict) else None,
            )

        return result

    def _get_all_target_words(self) -> List[str]:
        """Collect all target strings from feature_targets (for neutral exclusion)."""
        words: List[str] = []
        for cfg in self.feature_targets:
            for t, _ in cfg.flatten_targets_with_tags():
                if t and t not in words:
                    words.append(t)
        return words

    def generate_neutral_data(
        self,
        base_data: Optional[Union[str, pd.DataFrame, List[str]]] = None,
        additional_excluded_words: Optional[List[str]] = None,
        excluded_spacy_tags: Optional[
            Union[SpacyTagSpec, List[SpacyTagSpec]]
        ] = None,
        max_size: Optional[int] = None,
        format: str = "minimal",
        output: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate neutral data by excluding sentences with target tokens.

        Excludes sentences containing:
        - Any token in (target words + additional_excluded_words), deduplicated
        - Any token matching any spec in excluded_spacy_tags

        Use excluded_spacy_tags=[{"pos": "DET"}, {"pos": "PRON", "Person": "3"}]
        to exclude determiners and third-person pronouns.

        Args:
            base_data: Optional override (otherwise uses creator's base).
            additional_excluded_words: Extra words to exclude (in addition to
                target words from feature_targets). E.g. gendered articles or pronouns.
            excluded_spacy_tags: Spacy tag spec(s); exclude if any token matches
                any spec. Use list for multiple: [{"pos": "DET"}, {"pos": "PRON", "Person": "3"}].
            max_size: Global cap for neutral dataset.
            format: Return format ("minimal" = text column for eval).
            output: If set, save neutral data to this path. When output_dir is set on the
                creator and output is None, uses output_dir/neutral_basename + extension.

        Returns:
            DataFrame with at least "text" column.
        """
        texts = self._get_texts(base_override=base_data)
        sentence_stream = iter_sentences_from_texts(
            texts,
            self.preprocess,
            self.spacy_model,
            download_if_missing=self.download_if_missing,
        )
        target_words = self._get_all_target_words()
        extra = list(additional_excluded_words or [])
        excluded_words = list(dict.fromkeys(target_words + extra))
        if not excluded_words and not excluded_spacy_tags:
            if max_size is not None:
                from itertools import islice
                neutral = list(islice(sentence_stream, max_size))
                tqdm.write(f"Neutral: no exclusion filters, stopped at max_size={max_size} (got {len(neutral)}).")
            else:
                neutral = list(sentence_stream)
                tqdm.write(f"Neutral: no exclusion filters, kept all {len(neutral)} sentences.")
        else:
            neutral, neutral_stats = _filter_neutral(
                sentence_stream,
                excluded_words,
                excluded_spacy_tags,
                self.spacy_model,
                self.download_if_missing,
                max_size=max_size,
            )
            total_sent = neutral_stats.get("total", 0)
            kept = neutral_stats.get("kept", 0)
            rate = (kept / total_sent) if total_sent else 0.0
            tqdm.write(f"Neutral filter stats: {neutral_stats} (success rate: {rate:.2f})")
        if neutral is None:
            neutral = []
        rows = [{"text": s} for s in neutral]
        df = pd.DataFrame(rows)

        out_path = output or self._resolve_output_path("neutral", None)
        if out_path is not None:
            fmt = self.output_format
            if fmt == "hf":
                try:
                    from datasets import Dataset  # noqa: F401
                except ImportError:
                    fmt = "csv"
            _save_data(out_path, fmt, df=df)

        return df


def _apply_balance(
    class_dfs: Dict[str, pd.DataFrame],
    max_size_per_class: Optional[int],
    balance: Union[bool, str],
    feature_targets: List[TextFilterConfig],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """Apply balancing: strict caps all to smallest; try does round-robin with weights then fills."""
    if not class_dfs:
        return class_dfs

    config_by_id = {_class_id(cfg, i): cfg for i, cfg in enumerate(feature_targets)}
    class_ids = list(class_dfs.keys())
    weights = [
        config_by_id[cid].weight if cid in config_by_id else 1.0
        for cid in class_ids
    ]
    weights = [max(0.001, w) for w in weights]

    shuffled: Dict[str, pd.DataFrame] = {}
    for cid in class_ids:
        df = class_dfs[cid].copy()
        shuffled[cid] = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if balance == "strict":
        min_len = min(len(shuffled[cid]) for cid in class_ids)
        cap = min(min_len, max_size_per_class) if max_size_per_class is not None else min_len
        return {
            cid: shuffled[cid].iloc[:cap].reset_index(drop=True)
            for cid in class_ids
        }

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


def _filter_neutral(
    sentences: Union[Iterable[str], List[str]],
    excluded_words: List[str],
    excluded_spacy_tags: Optional[Union[SpacyTagSpec, List[SpacyTagSpec]]],
    spacy_model: Optional[str],
    download_if_missing: bool = False,
    max_size: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Filter out sentences containing excluded words or spacy tags.

    Consumes the sentence iterable (e.g. from iter_sentences_from_texts).
    If max_size is set, stops after collecting that many kept sentences.

    Returns:
        (kept_sentences, stats) with stats e.g. {"kept": N, "excluded": M, "total": T}.
    """
    if excluded_spacy_tags is not None and spacy_model is None:
        raise ValueError("spacy_model required when excluded_spacy_tags is set")

    import re
    word_pattern = None
    if excluded_words:
        word_pattern = re.compile(
            "|".join(rf"\b{re.escape(w)}\b" for w in excluded_words),
            re.IGNORECASE,
        )

    specs = excluded_spacy_tags
    if specs is not None and isinstance(specs, dict):
        specs = [specs]

    from itertools import islice
    if not excluded_words and not specs:
        out = list(sentences) if max_size is None else list(islice(sentences, max_size))
        return out, {"kept": len(out), "excluded": 0, "total": len(out)}

    nlp = None
    if specs:
        nlp = load_spacy_model(spacy_model, download_if_missing=download_if_missing)

    out: List[str] = []
    excluded_count = 0
    total_count = 0
    it = iter(sentences)
    pbar = tqdm(it, desc="Neutral filter", unit="sent", leave=True, position=0, dynamic_ncols=True)
    for sent in pbar:
        total_count += 1
        if word_pattern and word_pattern.search(sent):
            excluded_count += 1
            pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=True)
            continue
        if nlp is not None and specs:
            doc = nlp(sent)
            for token in doc:
                if any(token_matches_tags(token, s) for s in specs):
                    excluded_count += 1
                    pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=True)
                    break
            else:
                out.append(sent)
                pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=True)
                if max_size is not None and len(out) >= max_size:
                    break
        else:
            out.append(sent)
            pbar.set_postfix_str(f"kept={len(out)} | excluded={excluded_count}", refresh=True)
            if max_size is not None and len(out) >= max_size:
                break
    stats = {"kept": len(out), "excluded": excluded_count, "total": total_count}
    return out, stats


def _apply_auto_split(
    class_dfs: Dict[str, pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    """Assign train/validation/test split per class by ratio. Modifies class_dfs in place."""
    for class_id in list(class_dfs.keys()):
        df = class_dfs[class_id]
        if len(df) == 0:
            continue
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_train = max(0, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        splits = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
        df["split"] = splits[:n]
        class_dfs[class_id] = df


def _to_minimal(class_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert per-class to minimal (masked, label, label_class, split)."""
    rows = []
    for class_id, df in class_dfs.items():
        for _, row in df.iterrows():
            rows.append({
                "masked": row["masked"],
                "label": row["label"],
                "label_class": class_id,
                "split": row["split"],
            })
    return pd.DataFrame(rows)


def _to_merged(class_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build training DataFrame (factual only): masked, split, label_class, label, feature_class_id.

    feature_class_id is the string id from TextFilterConfig (same as label_class per row).
    Splits are applied per feature class in _apply_auto_split.
    """
    from gradiend.trainer.text.prediction.unified_data import per_class_dict_to_unified
    from gradiend.trainer.text.prediction.unified_schema import (
        UNIFIED_FACTUAL,
        UNIFIED_FACTUAL_CLASS,
    )

    df = per_class_dict_to_unified(
        class_dfs,
        classes=list(class_dfs.keys()),
        masked_col="masked",
        split_col="split",
        use_class_names_as_columns=True,
    )
    # Map unified schema to merged column names expected by callers
    df = df.rename(columns={UNIFIED_FACTUAL_CLASS: "label_class", UNIFIED_FACTUAL: "label"})
    df["feature_class_id"] = df["label_class"]
    return df
