#!/usr/bin/env python3
"""
Warm the Hugging Face cache for datasets used by gradiend examples/experiments.

Why a separate env?
-------------------
Current gradiend installs ``datasets>=3.0``, which no longer runs repository
loading scripts (``genter.py``, ``geneutral.py``, …). Use a small **prefetch
env** with ``datasets<3`` to download once, then point your main gradiend env
at the same ``HF_HOME`` / ``HF_DATASETS_CACHE``.

Important limitation
--------------------
``datasets`` 3.x still refuses script-based hubs even if files exist in cache.
For ``aieng-lab/genter`` and ``aieng-lab/geneutral`` you should either:

  * export Parquet with ``--export-dir`` and load those paths in your code, or
  * keep ``datasets<3`` in the runtime env that calls ``gender_en.build_gender_trainer``.

Parquet / per-class datasets (race, religion, German gender, biasneutral, …)
work fine from cache with ``datasets`` 3.x once prefetched.

Quick start
-----------
::

    conda create -n gradiend-prefetch python=3.11 -y
    conda activate gradiend-prefetch
    pip install -r scripts/requirements-prefetch-datasets.txt

    # Use the same cache directory as your main gradiend env:
    export HF_HOME=/shared/drechsel/hf-cache
    export HF_DATASETS_CACHE=$HF_HOME/datasets

    # Optional: fix SSL / xet issues on minimal HPC images
    export HF_HUB_DISABLE_XET=1

    python scripts/prefetch_hf_datasets.py --profile multilingual_demo
    python scripts/prefetch_hf_datasets.py --profile gender_en --export-dir data/hf_exports

Then in your main env::

    export HF_HOME=/shared/drechsel/hf-cache
    export HF_DATASETS_CACHE=$HF_HOME/datasets
    python experiments/multilingual_gradiend_demo.py

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# Must be set before huggingface_hub / datasets import (SSL + xet workarounds).
from gradiend.util.hf_env import configure_hf_download_env

configure_hf_download_env()

from datasets import get_dataset_config_names, load_dataset  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    hf_config: Optional[str] = None
    trust_remote_code: bool = False
    per_class: bool = False
    splits: Optional[Sequence[str]] = None
    note: str = ""


MULTILINGUAL_DEMO_DATASETS: List[DatasetSpec] = [
    DatasetSpec("aieng-lab/de-gender-case-articles", per_class=True, note="German gender/case"),
    DatasetSpec("aieng-lab/wortschatz-leipzig-de-grammar-neutral", note="German neutral eval"),
    DatasetSpec("aieng-lab/gradiend_race_data", per_class=True, note="Race per-class"),
    DatasetSpec("aieng-lab/gradiend_religion_data", per_class=True, note="Religion per-class"),
    DatasetSpec("aieng-lab/biasneutral", note="Race/religion neutral eval"),
    DatasetSpec(
        "tweet_eval",
        hf_config="sentiment",
        splits=["train", "validation", "test"],
        note="Sentiment example (masked emotion words)",
    ),
]

GENDER_EN_DATASETS: List[DatasetSpec] = [
    DatasetSpec("aieng-lab/genter", trust_remote_code=True, note="Loading script — export recommended"),
    DatasetSpec("aieng-lab/geneutral", trust_remote_code=True, note="Loading script — export recommended"),
    DatasetSpec("aieng-lab/gentypes", note="Decoder eval names"),
    DatasetSpec("aieng-lab/namextend", note="Name augmentation"),
    DatasetSpec("aieng-lab/namexact", note="Name augmentation (all splits)"),
]

PROFILES = {
    "multilingual_demo": MULTILINGUAL_DEMO_DATASETS,
    "gender_en": GENDER_EN_DATASETS,
    "all": MULTILINGUAL_DEMO_DATASETS + GENDER_EN_DATASETS,
}


def _load_kwargs(trust_remote_code: bool) -> dict:
    if trust_remote_code:
        return {"trust_remote_code": True}
    return {}


def _save_dataset_dict(ds, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ds, "items"):
        for split_name, split_ds in ds.items():
            out = export_path.with_name(f"{export_path.stem}_{split_name}.parquet")
            split_ds.to_parquet(out)
    else:
        ds.to_parquet(export_path)


def prefetch_spec(spec: DatasetSpec, *, export_dir: Optional[Path]) -> None:
    print(f"\n=== {spec.repo_id} ===")
    if spec.note:
        print(f"    ({spec.note})")
    kwargs = _load_kwargs(spec.trust_remote_code)

    if spec.per_class:
        config_names = get_dataset_config_names(spec.repo_id, **kwargs)
        if not config_names:
            raise RuntimeError(f"No configs/subsets found for {spec.repo_id}")
        print(f"    subsets: {config_names}")
        for class_name in config_names:
            ds = load_dataset(spec.repo_id, class_name, **kwargs)
            n = sum(len(split) for split in ds.values()) if hasattr(ds, "values") else len(ds)
            print(f"    loaded {class_name}: {n} rows")
            if export_dir is not None:
                safe_repo = spec.repo_id.replace("/", "__")
                out = export_dir / safe_repo / f"{class_name}.parquet"
                _save_dataset_dict(ds, out)
                print(f"    exported -> {out}")
        return

    if spec.splits:
        for split in spec.splits:
            if spec.hf_config is not None:
                ds = load_dataset(spec.repo_id, spec.hf_config, split=split, **kwargs)
            else:
                ds = load_dataset(spec.repo_id, split=split, **kwargs)
            print(f"    loaded split {split}: {len(ds)} rows")
            if export_dir is not None:
                safe_repo = spec.repo_id.replace("/", "__")
                config_suffix = f"_{spec.hf_config}" if spec.hf_config else ""
                out = export_dir / safe_repo / f"{split}{config_suffix}.parquet"
                _save_dataset_dict(ds, out)
                print(f"    exported -> {out}")
        return

    if spec.hf_config is not None:
        ds = load_dataset(spec.repo_id, spec.hf_config, **kwargs)
    else:
        ds = load_dataset(spec.repo_id, **kwargs)
    if hasattr(ds, "items"):
        for split_name, split_ds in ds.items():
            print(f"    loaded split {split_name}: {len(split_ds)} rows")
    else:
        print(f"    loaded: {len(ds)} rows")
    if export_dir is not None:
        safe_repo = spec.repo_id.replace("/", "__")
        out = export_dir / safe_repo / "dataset.parquet"
        _save_dataset_dict(ds, out)
        print(f"    exported -> {out.parent}/")


def _check_datasets_version() -> None:
    import datasets

    major = int(datasets.__version__.split(".", 1)[0])
    if major >= 3:
        print(
            "ERROR: This script requires datasets<3 to prefetch script-based hubs "
            f"(installed: {datasets.__version__}).\n"
            "Create the prefetch env:\n"
            "  conda create -n gradiend-prefetch python=3.11 -y\n"
            "  conda activate gradiend-prefetch\n"
            "  pip install -r scripts/requirements-prefetch-datasets.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="all",
        help="Which dataset groups to prefetch (default: all).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory for Parquet exports (for script-based datasets).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="REPO_ID",
        help="Prefetch a single repo id in addition to the profile (repeatable).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True for --dataset entries.",
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Treat --dataset entries as per-class HF datasets.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _check_datasets_version()

    cache_root = os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE") or "~/.cache/huggingface"
    print(f"HF_HOME={os.environ.get('HF_HOME', '(unset)')}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '(unset)')}")
    print(f"Effective cache root hint: {cache_root}")
    print(f"datasets version: ", end="")
    import datasets

    print(datasets.__version__)

    specs: List[DatasetSpec] = list(PROFILES[args.profile])
    for repo_id in args.dataset:
        specs.append(
            DatasetSpec(
                repo_id=repo_id,
                trust_remote_code=args.trust_remote_code,
                per_class=args.per_class,
            )
        )

    export_dir = args.export_dir
    if export_dir is not None:
        export_dir = export_dir.resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"Export dir: {export_dir}")

    for spec in specs:
        prefetch_spec(spec, export_dir=export_dir)

    print("\nDone.")
    if export_dir is not None:
        print(
            "Parquet exports are script-free and can be loaded with pandas or datasets 3.x.\n"
            "For genter/geneutral, point gender_en at the exported files or keep datasets<3."
        )


if __name__ == "__main__":
    main()
