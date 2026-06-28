"""
Small multilingual GRADIEND demo for cross-encoding matrix debugging.

Subset of multilingual_gradiend_demo.py:
- Race (3 GRADIENDs: white/black, white/asian, black/asian)
- German case: all der/dem-related transitions (der<->dem, den<->dem, das<->dem)

Faster defaults (fewer steps, smaller data caps). Skips pronouns, religion, sentiment,
English gender, and the full German case grid.

Usage:
  python experiments/multilingual_gradiend_demo_small.py
  python experiments/multilingual_gradiend_demo_small.py --plot-only
  python experiments/multilingual_gradiend_demo_small.py --max-steps 100
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEMO_PATH = Path(__file__).resolve().parent / "multilingual_gradiend_demo.py"
_spec = importlib.util.spec_from_file_location("multilingual_gradiend_demo", _DEMO_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load {_DEMO_PATH}")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

from gradiend import (
    SuitePairDefinition,
    SymmetricTrainerSuite,
    TextPredictionTrainer,
    TrainerCollection,
    TrainingArguments,
    plot_cross_encoding_heatmap,
    plot_gradiend_transition_cross_encoding_heatmap,
)
from gradiend.comparison.feature_cross_encoding import (
    build_cross_task_encoder_summary,
    collect_unified_test_transitions,
)

DecoderEvalMode = demo.DecoderEvalMode
ExperimentConfig = demo.ExperimentConfig
MlmHeadScope = demo.MlmHeadScope
_default_post_prune_topk = demo._default_post_prune_topk
_resolve_decoder_eval_mode = demo._resolve_decoder_eval_mode
_suite_common_kwargs = demo._suite_common_kwargs
_train_suite = demo._train_suite
configs_de = demo.configs_de
prepare_global_mlm_head = demo.prepare_global_mlm_head
race_configs = demo.race_configs
DECODER_MODEL = demo.DECODER_MODEL
ENCODER_MODEL = demo.ENCODER_MODEL
DECODER_MLM_HEAD_ARGS = demo.DECODER_MLM_HEAD_ARGS

# German article groups that involve der (nom/dat/gen) and dem (dat).
CONFIGS_DE_DER_DEM: Dict[str, list] = {
    key: list(pairs)
    for key, pairs in configs_de.items()
    if key in {"der <-> dem", "den <-> dem", "das <-> dem"}
}

GERMAN_DER_DEM_CLASSES = sorted(
    {cls for pairs in CONFIGS_DE_DER_DEM.values() for pair in pairs for cls in pair}
)
SMALL_FEATURE_CLASSES = sorted(set(GERMAN_DER_DEM_CLASSES + ["white", "black", "asian"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small multilingual demo for cross-encoding matrix testing.",
    )
    parser.add_argument("--model", default=None, help="HF model id/path.")
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
    parser.add_argument("--source", choices=["alternative"], default="alternative")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Training steps per GRADIEND (default 250; full demo uses 1000).",
    )
    parser.add_argument(
        "--train-max-size",
        type=int,
        default=3000,
        help="Max training rows per feature class (default 3000).",
    )
    parser.add_argument(
        "--encoder-eval-max-size",
        type=int,
        default=30,
        help="Max test rows per factual class for cross-encoding (default 30).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; only plot cross-encoding (requires a prior run in the experiment dir).",
    )
    parser.add_argument(
        "--retain-models-in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pre-prune-topk", type=float, default=0.01)
    parser.add_argument("--post-prune-topk", type=float, default=None)
    return parser.parse_args()


def build_experiment_config(cli_args: argparse.Namespace) -> ExperimentConfig:
    from gradiend.trainer import PostPruneConfig, PrePruneConfig

    decoder_eval_mode = _resolve_decoder_eval_mode(cli_args.decoder_eval_mode)
    default_model = DECODER_MODEL if decoder_eval_mode is not DecoderEvalMode.NONE else ENCODER_MODEL
    model_name = cli_args.model or default_model
    mode_suffix = "" if decoder_eval_mode is DecoderEvalMode.NONE else f"_{decoder_eval_mode.value}"
    pre_prune_topk = cli_args.pre_prune_topk
    post_prune_topk = (
        cli_args.post_prune_topk
        if cli_args.post_prune_topk is not None
        else _default_post_prune_topk(pre_prune_topk)
    )
    pre_suffix = "" if pre_prune_topk == 0.01 else f"_pre{pre_prune_topk:g}"
    experiment_dir = (
        f"runs/multilingual_gradiend_demo_small_{model_name.split('/')[-1]}{mode_suffix}{pre_suffix}_v1"
    )

    base_args = TrainingArguments(
        experiment_dir=experiment_dir,
        train_batch_size=16,
        train_max_size=cli_args.train_max_size,
        eval_steps=100,
        max_steps=cli_args.max_steps,
        encoder_eval_max_size=cli_args.encoder_eval_max_size,
        decoder_eval_max_size_training_like=50,
        decoder_eval_max_size_neutral=200,
        num_train_epochs=3,
        max_seeds=3,
        min_convergent_seeds=1,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        pre_prune_config=PrePruneConfig(n_samples=16, topk=pre_prune_topk, source="diff"),
        post_prune_config=PostPruneConfig(topk=post_prune_topk, part="decoder-weight"),
        add_identity_for_other_classes=True,
        use_cache=True,
    )

    return ExperimentConfig(
        model_name=model_name,
        decoder_eval_mode=decoder_eval_mode,
        mlm_head_scope=MlmHeadScope(cli_args.mlm_head_scope),
        args=base_args,
        mlm_head_args=dict(DECODER_MLM_HEAD_ARGS),
    )


def build_race_suite(config: ExperimentConfig, *, retain_models_in_memory: bool) -> SymmetricTrainerSuite:
    pair_definitions = [
        SuitePairDefinition(
            target_classes=pair,
            child_id=f"{bias_type}_{pair[0]}_{pair[1]}",
            label=f"{pair[0]} <-> {pair[1]}",
        )
        for bias_type, pair, _other_classes in race_configs
    ]
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data="aieng-lab/gradiend_race_data",
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/biasneutral",
        pair_definitions=pair_definitions,
        **_suite_common_kwargs(config, "train_race_religion", retain_models_in_memory=retain_models_in_memory),
    )


def build_gender_de_der_dem_suite(
    config: ExperimentConfig,
    *,
    retain_models_in_memory: bool,
) -> SymmetricTrainerSuite:
    pair_definitions = [
        SuitePairDefinition(
            target_classes=pair,
            child_id=f"gender_de_{pair[0]}_{pair[1]}",
            label=f"{pair[0]} <-> {pair[1]}",
        )
        for pairs in CONFIGS_DE_DER_DEM.values()
        for pair in pairs
    ]
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data="aieng-lab/de-gender-case-articles",
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        pair_definitions=pair_definitions,
        **_suite_common_kwargs(config, "gender_de", retain_models_in_memory=retain_models_in_memory),
    )


def train_small(
    config: ExperimentConfig,
    *,
    race_suite: SymmetricTrainerSuite,
    gender_de_suite: SymmetricTrainerSuite,
) -> Dict[str, Any]:
    models_for_heatmap: Dict[str, Any] = {}
    os.makedirs(config.args.experiment_dir, exist_ok=True)

    all_trainers = list(TrainerCollection.merge(race_suite, gender_de_suite).values())
    prepare_global_mlm_head(config, all_trainers)

    print("=== Race (3 GRADIENDs) ===")
    _train_suite(config, problem="train_race_religion", suite=race_suite, models_for_heatmap=models_for_heatmap)

    print("=== German der/dem transitions ===")
    for group, pairs in CONFIGS_DE_DER_DEM.items():
        print(f"  {group}: {len(pairs)} pair(s)")
    _train_suite(config, problem="gender_de", suite=gender_de_suite, models_for_heatmap=models_for_heatmap)

    return models_for_heatmap


def plot_cross_encoding(
    config: ExperimentConfig,
    trainers_by_id: Dict[str, TextPredictionTrainer],
) -> None:
    from gradiend.visualizer.multilingual_demo_labels import (
        build_demo_feature_label_mapping,
        build_demo_feature_plot_groups,
        build_demo_trainer_label_mapping,
        build_demo_trainer_order_and_groups,
        build_demo_transition_label_mapping,
        demo_encoding_heatmap_normalized_style_kwargs,
        demo_encoding_heatmap_style_kwargs,
    )

    feature_order = list(SMALL_FEATURE_CLASSES)
    feature_groups = build_demo_feature_plot_groups(feature_order)
    trainer_order, trainer_groups = build_demo_trainer_order_and_groups(
        sorted(trainers_by_id.keys())
    )
    trainer_order = [tid for tid in trainer_order if tid in trainers_by_id]
    style = demo_encoding_heatmap_style_kwargs(annot=True, annot_fmt=".2f")
    normalized_style = demo_encoding_heatmap_normalized_style_kwargs()
    feature_labels = build_demo_feature_label_mapping(feature_order)
    trainer_labels = build_demo_trainer_label_mapping(trainer_order)

    encoder_summary = build_cross_task_encoder_summary(
        trainers_by_id,
        feature_order,
        split="test",
        max_size=config.args.encoder_eval_max_size,
    )
    transition_order = collect_unified_test_transitions(trainers_by_id, split="test")
    transition_labels = build_demo_transition_label_mapping(transition_order)

    gradiend_transition_path = os.path.join(
        config.args.experiment_dir,
        "cross_encoding_gradiend_by_transition_heatmap.pdf",
    )
    plot_gradiend_transition_cross_encoding_heatmap(
        trainers_by_id,
        trainer_order=trainer_order,
        transition_order=transition_order,
        encoder_summary=encoder_summary,
        split="test",
        max_size=config.args.encoder_eval_max_size,
        order=trainer_order,
        pretty_groups=trainer_groups,
        row_label_mapping=trainer_labels,
        column_label_mapping=transition_labels,
        output_path=gradiend_transition_path,
        title="GRADIEND × input transition (pre-anchor)",
        show=True,
        **style,
    )
    print(f"Saved {gradiend_transition_path}")

    for alignment, column_ids, title, suffix in (
        ("factual", feature_order, "Cross-encoding (factual)", "factual"),
        ("counterfactual", feature_order, "Cross-encoding (counterfactual)", "counterfactual"),
        ("transition", None, "Cross-encoding (transition)", "transition"),
    ):
        output_path = os.path.join(
            config.args.experiment_dir,
            f"cross_encoding_oriented_{suffix}_heatmap.pdf",
        )
        col_labels = transition_labels if alignment == "transition" else feature_labels
        plot_cross_encoding_heatmap(
            trainers_by_id,
            feature_order,
            alignment=alignment,
            column_ids=column_ids,
            encoder_summary=encoder_summary,
            split="test",
            max_size=config.args.encoder_eval_max_size,
            cross_task_eval=False,
            aggregate="mean",
            order=feature_order,
            pretty_groups=feature_groups,
            row_label_mapping=feature_labels,
            column_label_mapping=col_labels,
            output_path=output_path,
            title=title,
            show=True,
            **style,
        )
        print(f"Saved {output_path}")

    for alignment, column_ids, suffix in (
        ("factual", feature_order, "factual"),
        ("counterfactual", feature_order, "counterfactual"),
    ):
        col_labels = feature_labels
        normalized_output = os.path.join(
            config.args.experiment_dir,
            f"cross_encoding_oriented_{suffix}_row_normalized_heatmap.pdf",
        )
        plot_cross_encoding_heatmap(
            trainers_by_id,
            feature_order,
            alignment=alignment,
            column_ids=column_ids,
            encoder_summary=encoder_summary,
            split="test",
            max_size=config.args.encoder_eval_max_size,
            cross_task_eval=False,
            aggregate="mean",
            normalize=True,
            order=feature_order,
            pretty_groups=feature_groups,
            row_label_mapping=feature_labels,
            column_label_mapping=col_labels,
            output_path=normalized_output,
            title=False,
            show=True,
            **normalized_style,
        )
        print(f"Saved {normalized_output}")


def main() -> None:
    cli_args = parse_args()
    config = build_experiment_config(cli_args)

    print(f"Model: {config.model_name}")
    print(f"Experiment dir: {config.args.experiment_dir}")
    print(f"GRADIENDs: {len(race_configs)} race + {sum(len(p) for p in CONFIGS_DE_DER_DEM.values())} der/dem")
    print(f"Feature classes in matrix: {len(SMALL_FEATURE_CLASSES)}")

    race_suite = build_race_suite(config, retain_models_in_memory=cli_args.retain_models_in_memory)
    gender_de_suite = build_gender_de_der_dem_suite(
        config, retain_models_in_memory=cli_args.retain_models_in_memory
    )

    trainers_by_id = TrainerCollection.merge(race_suite, gender_de_suite).trainers

    if not cli_args.plot_only:
        train_small(
            config,
            race_suite=race_suite,
            gender_de_suite=gender_de_suite,
        )
    else:
        print("--plot-only: skipping training (using cached checkpoints where available)")


    plot_cross_encoding(config, trainers_by_id)


if __name__ == "__main__":
    main()
