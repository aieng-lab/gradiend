"""
Plot only the 3-set English pronoun Venn diagram from the paper/demo workflow.

This is a lightweight extraction of the Venn section in
``experiments/multilingual_gradiend_demo.py``. It reuses the same pronoun
trainer IDs and experiment-directory convention, then writes only:

    topk_overlap_venn_three_train_english_pronouns.pdf

Run after the corresponding multilingual demo pronoun checkpoints exist:

    python experiments/paper_workflow_enxtended.py --plot-only
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from gradiend import plot_topk_overlap_venn

from experiments.multilingual_gradiend_demo import (
    DecoderEvalMode,
    MlmHeadScope,
    build_experiment_config,
    build_pronoun_suite,
    load_models_for_heatmap_from_cache,
    _pretty_label,
)


DEFAULT_VENN_PRONOUN_IDS = (
    "pronoun_1SG_3PL",
    "pronoun_1SG_3SG",
    "pronoun_3SG_3PL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot only the three-English-pronoun top-k overlap Venn diagram."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HF model id/path. Defaults exactly like multilingual_gradiend_demo.py: "
            "multilingual BERT in encoder mode, XGLM in decoder modes."
        ),
    )
    parser.add_argument(
        "--decoder-eval-mode",
        "--decoder-mode",
        choices=[mode.value for mode in DecoderEvalMode] + ["combined", "mlm-only"],
        default=DecoderEvalMode.NONE.value,
    )
    parser.add_argument(
        "--mlm-head-scope",
        choices=[scope.value for scope in MlmHeadScope],
        default=MlmHeadScope.PER_RUN.value,
    )
    parser.add_argument("--pre-prune-topk", type=float, default=0.1)
    parser.add_argument("--post-prune-topk", type=float, default=0.01)
    parser.add_argument("--three-seed", action="store_true")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--part", default="decoder-weight")
    parser.add_argument(
        "--pronoun-ids",
        nargs=3,
        default=list(DEFAULT_VENN_PRONOUN_IDS),
        metavar="RUN_ID",
        help="Exactly three pronoun run IDs to include in the Venn diagram.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Output PDF/PNG path. Defaults to the multilingual demo experiment "
            "directory's topk_overlap_venn_three_train_english_pronouns.pdf."
        ),
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the plot window while saving.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Document intent: this script only plots the Venn diagram. The flag is "
            "accepted for symmetry with multilingual_gradiend_demo.py."
        ),
    )
    parser.add_argument(
        "--problem-learning-rate",
        action="append",
        default=[],
        metavar="PROBLEM=LR",
        help="Forwarded to the multilingual demo config builder for path compatibility.",
    )
    parser.add_argument("--mlm-head-max-size", type=int, default=1000)
    parser.add_argument("--mlm-head-epochs", type=int, default=5)
    return parser.parse_args()


def _select_trainers(pronoun_suite, pronoun_ids: Sequence[str]):
    missing = [run_id for run_id in pronoun_ids if run_id not in pronoun_suite.trainers]
    if missing:
        available = ", ".join(sorted(pronoun_suite.trainers))
        raise ValueError(
            "Requested pronoun Venn run IDs are not available: "
            f"{', '.join(missing)}. Available pronoun runs: {available}"
        )
    return {run_id: pronoun_suite.trainers[run_id] for run_id in pronoun_ids}


def main() -> None:
    cli_args = parse_args()
    config = build_experiment_config(cli_args)
    pronoun_suite = build_pronoun_suite(config, retain_models_in_memory=False)
    selected_trainers = _select_trainers(pronoun_suite, cli_args.pronoun_ids)

    print("=== English pronoun Venn only ===")
    print(f"Experiment dir: {config.args.experiment_dir}")
    print(f"Pronoun runs: {', '.join(cli_args.pronoun_ids)}")

    models_for_venn = load_models_for_heatmap_from_cache(config, selected_trainers)
    labelled_models = {
        _pretty_label(run_id): models_for_venn[run_id]
        for run_id in cli_args.pronoun_ids
    }

    output_path = cli_args.output_path or os.path.join(
        config.args.experiment_dir,
        "topk_overlap_venn_three_train_english_pronouns.pdf",
    )
    plot_topk_overlap_venn(
        labelled_models,
        topk=cli_args.topk,
        part=cli_args.part,
        output_path=output_path,
        show=cli_args.show,
    )
    print(f"Venn plot saved to {output_path}")


if __name__ == "__main__":
    main()
