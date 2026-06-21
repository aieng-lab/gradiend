"""
Multilingual GRADIEND comparison for system demo: one large heatmap.

Trains GRADIENDs for:
- German gender/case (train_gender_de_detailed configs: multiple pairs)
- English gender (gender_en: one GRADIEND M vs F)
- Race (3 GRADIENDs: white-black, white-asian, black-asian)
- Religion (3 GRADIENDs: christian-muslim, christian-jewish, muslim-jewish)
- English pronouns (10 GRADIENDs: all pairs from 1SG, 1PL, 2SGPL, 3SG, 3PL)
- English pronoun merged groups (4 GRADIENDs: Number SG vs PL; Person 1vs2, 1vs3, 2vs3)
- English sentiment (1 GRADIEND: positive vs negative, masked emotion words from tweet_eval)
- English formality (disabled until GYAFC data is available; see ENABLE_FORMALITY)

Uses a single multilingual model and the same TrainingArguments as
train_gender_de_detailed. Encoder mode defaults to multilingual BERT; decoder-only
modes default to XGLM-564M. Pairwise training is orchestrated via
SymmetricTrainerSuite where possible.

At the end, plots in the experiment directory root:
- topk_overlap_heatmap_all_1000.pdf
- cross_encoding_oriented_factual_heatmap.pdf,
  cross_encoding_oriented_counterfactual_heatmap.pdf, and
  cross_encoding_oriented_transition_heatmap.pdf
  (oriented cross-encoding under the three alignment conventions)
- topk_overlap_venn_three_train_english_pronouns.pdf (when enough pronoun runs exist)
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from gradiend import (
    SuitePairDefinition,
    SymmetricTrainerSuite,
    TextPredictionTrainer,
    TrainingArguments,
    plot_anchor_aligned_encoding_heatmap,
    plot_topk_overlap_heatmap,
    plot_topk_overlap_venn,
)
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.examples.create_english_pronoun_data import ensure_english_pronoun_data
from gradiend.trainer.text.prediction.decoder_only_mlm import train_mlm_head
from gradiend.util.paths import has_saved_decoder_mlm_head, resolve_decoder_mlm_head_dir


ENCODER_MODEL = "google-bert/bert-base-multilingual-cased"
DECODER_MODEL = "facebook/xglm-564M"


class DecoderEvalMode(str, Enum):
    """How decoder-only models are evaluated for gradient training."""

    NONE = "none"
    DEFAULT = "default"
    LEFT_CONTEXT = "left_context"
    MLM_HEAD = "mlm_head"


class MlmHeadScope(str, Enum):
    """Where to train/share custom decoder MLM heads."""

    PER_RUN = "per_run"
    PER_GROUP = "per_group"
    GLOBAL = "global"


PROBLEM_DECODER_HEAD_POLICY = {
    # German article/case labels often precede the noun, so right context is
    # part of the actual problem. Pure CLM only sees the prefix before [MASK].
    "gender_de": True,
    # These demos use target tokens that can be modeled as the next token after
    # the prefix. In default mode we keep the native causal LM head here.
    "gender_en": False,
    "train_race_religion": False,
    "pronoun": False,
    "pronoun_merged": False,
    "sentiment": False,
    "formality": False,
}

MLM_HEAD_GROUP_BY_PROBLEM = {
    "gender_de": "gender_de",
    "gender_en": "gender_en",
    "train_race_religion": "train_race_religion",
    "pronoun": "pronoun",
    "pronoun_merged": "pronoun",
    "sentiment": "sentiment",
    "formality": "formality",
}

DECODER_MLM_HEAD_ARGS = {
    "batch_size": 4,
    "epochs": 3,
    "lr": 1e-4,
    "max_size": 1000,
}

# German gender/case configs (from train_gender_de_detailed)
configs_de = {
    "die <-> das": [
        ("fem_acc", "neut_acc"),
        ("fem_nom", "neut_nom"),
    ],
    "der <-> die": [
        ("masc_nom", "fem_nom"),
        ("fem_nom", "fem_dat"),
        ("fem_nom", "fem_gen"),
        ("fem_acc", "fem_dat"),
        ("fem_acc", "fem_gen"),
    ],
    "der <-> das": [
        ("masc_nom", "neut_nom"),
    ],
    "der <-> den": [
        ("masc_nom", "masc_acc"),
    ],
    "der <-> dem": [
        ("masc_nom", "masc_dat"),
        ("fem_dat", "masc_dat"),
        ("fem_dat", "neut_dat"),
    ],
    "der <-> des": [
        ("masc_nom", "masc_gen"),
        ("fem_gen", "masc_gen"),
        ("fem_gen", "neut_gen"),
    ],
    "die <-> den": [
        ("fem_acc", "masc_acc"),
    ],
    "das <-> dem": [
        ("neut_nom", "neut_dat"),
        ("neut_acc", "neut_dat"),
    ],
    "das <-> des": [
        ("neut_nom", "neut_gen"),
        ("neut_acc", "neut_gen"),
    ],
    "den <-> dem": [
        ("masc_acc", "masc_dat"),
    ],
    "dem <-> des": [
        ("masc_dat", "masc_gen"),
        ("neut_dat", "neut_gen"),
    ],
}

race_configs = [
    ("race", ("white", "black"), ["asian"]),
    ("race", ("white", "asian"), ["black"]),
    ("race", ("black", "asian"), ["white"]),
]

religion_configs = [
    ("religion", ("christian", "muslim"), ["jewish"]),
    ("religion", ("christian", "jewish"), ["muslim"]),
    ("religion", ("muslim", "jewish"), ["christian"]),
]

PRONOUN_DATA_DIR = "data/english_pronouns"
SENTIMENT_DATA_DIR = "data/sentiment_tweets"
FORMALITY_DATA_DIR = "data/formality_gyafc"
# Re-enable when training.csv and neutral.csv exist under FORMALITY_DATA_DIR.
ENABLE_FORMALITY = False
PRONOUN_CLASSES = ["1SG", "1PL", "2SGPL", "3SG", "3PL"]
SENTIMENT_CLASSES = ["positive", "negative"]
FORMALITY_CLASSES = ["informal", "formal"]
pronoun_pairs = [
    (PRONOUN_CLASSES[i], PRONOUN_CLASSES[j])
    for i in range(len(PRONOUN_CLASSES))
    for j in range(i + 1, len(PRONOUN_CLASSES))
]

GERMAN_CASE_CLASSES = sorted({cls for pairs in configs_de.values() for pair in pairs for cls in pair})
MULTILINGUAL_FEATURE_CLASSES = sorted(
    set(
        GERMAN_CASE_CLASSES
        + ["M", "F"]
        + ["white", "black", "asian"]
        + ["christian", "muslim", "jewish"]
        + PRONOUN_CLASSES
        + SENTIMENT_CLASSES
        #+ FORMALITY_CLASSES
    )
)

pronoun_merged_configs = [
    ("pronoun_number_singular_plural", {"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]}, "English Number SG↔PL", [["1SG", "1PL"], ["3SG", "3PL"]]),
    ("pronoun_person_1vs2", {"1st": ["1SG", "1PL"], "2nd": ["2SGPL"]}, "English Person 1vs2", None),
    ("pronoun_person_1vs3", {"1st": ["1SG", "1PL"], "3rd": ["3SG", "3PL"]}, "English Person 1vs3", [["1SG", "3SG"], ["1PL", "3PL"]]),
    ("pronoun_person_2vs3", {"2nd": ["2SGPL"], "3rd": ["3SG", "3PL"]}, "English Person 2vs3", None),
]


class ExperimentConfig:
    model_name: str
    decoder_eval_mode: DecoderEvalMode
    mlm_head_scope: MlmHeadScope
    args: TrainingArguments
    mlm_head_args: Dict[str, Any]

    def __init__(
        self,
        *,
        model_name: str,
        decoder_eval_mode: DecoderEvalMode,
        mlm_head_scope: MlmHeadScope,
        args: TrainingArguments,
        mlm_head_args: Dict[str, Any],
    ) -> None:
        self.model_name = model_name
        self.decoder_eval_mode = decoder_eval_mode
        self.mlm_head_scope = mlm_head_scope
        self.args = args
        self.mlm_head_args = mlm_head_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multilingual GRADIEND demo.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HF model id/path. Defaults to the multilingual BERT encoder in encoder mode "
            f"and to {DECODER_MODEL!r} in decoder modes."
        ),
    )
    parser.add_argument(
        "--decoder-eval-mode",
        "--decoder-mode",
        choices=[mode.value for mode in DecoderEvalMode] + ["combined", "mlm-only"],
        default=DecoderEvalMode.NONE.value,
        help=(
            "'default' (alias: 'combined') keeps task-specific decoder behavior (MLM head for "
            "German articles, left context elsewhere). 'left_context' uses CLM next-token scoring "
            "for every task. 'mlm_head' (alias: 'mlm-only') trains a custom MLM head "
            "(see --mlm-head-scope)."
        ),
    )
    parser.add_argument(
        "--mlm-head-scope",
        choices=[scope.value for scope in MlmHeadScope],
        default=MlmHeadScope.PER_RUN.value,
        help=(
            "How to share decoder-only MLM heads when --decoder-eval-mode=mlm_head or when "
            "default mode needs a head: one per run, one per task group, or one global head."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["alternative"],
        default="alternative",
        help="GRADIEND input source. Multilingual cross-encoding requires alternative.",
    )
    parser.add_argument(
        "--mlm-head-max-size",
        type=int,
        default=DECODER_MLM_HEAD_ARGS["max_size"],
        help="Per-label max rows used to train each decoder-only custom MLM head.",
    )
    parser.add_argument(
        "--mlm-head-epochs",
        type=int,
        default=DECODER_MLM_HEAD_ARGS["epochs"],
        help="Epochs for each decoder-only custom MLM head.",
    )
    parser.add_argument(
        "--retain-models-in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep every trained model loaded while plotting (can OOM on large suites).",
    )
    parser.add_argument(
        "--pre-prune-topk",
        type=float,
        default=0.01,
        help="Pre-prune keep fraction (float in (0,1]) or absolute dim count (int). Default: 0.01.",
    )
    parser.add_argument(
        "--post-prune-topk",
        type=float,
        default=None,
        help="Post-prune keep fraction after training. Default: 0.05 with pre-prune 0.01, else 0.01.",
    )
    return parser.parse_args()


def _default_post_prune_topk(pre_prune_topk: float) -> float:
    return 0.05 if pre_prune_topk == 0.01 else 0.01


def _resolve_decoder_eval_mode(raw: str) -> DecoderEvalMode:
    aliases = {
        "combined": DecoderEvalMode.DEFAULT,
        "mlm-only": DecoderEvalMode.MLM_HEAD,
    }
    if raw in aliases:
        return aliases[raw]
    return DecoderEvalMode(raw)


def build_experiment_config(cli_args: argparse.Namespace) -> ExperimentConfig:
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
    experiment_dir = f"runs/multilingual_gradiend_demo_{model_name.split('/')[-1]}{mode_suffix}{pre_suffix}_v5"

    base_args = TrainingArguments(
        experiment_dir=experiment_dir,
        train_batch_size=16,
        train_max_size=20000,
        eval_steps=250,
        max_steps=1000,
        encoder_eval_max_size=50,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=500,
        num_train_epochs=3,
        max_seeds=10,
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
    mlm_head_args = dict(DECODER_MLM_HEAD_ARGS)
    mlm_head_args["epochs"] = cli_args.mlm_head_epochs
    mlm_head_args["max_size"] = cli_args.mlm_head_max_size
    return ExperimentConfig(
        model_name=model_name,
        decoder_eval_mode=decoder_eval_mode,
        mlm_head_scope=MlmHeadScope(cli_args.mlm_head_scope),
        args=base_args,
        mlm_head_args=mlm_head_args,
    )


def problem_args(config: ExperimentConfig, **overrides: Any) -> TrainingArguments:
    return replace(config.args, **overrides)


def needs_mlm_head_for_problem(config: ExperimentConfig, problem: str) -> bool:
    if config.decoder_eval_mode is DecoderEvalMode.NONE:
        return False
    if config.decoder_eval_mode is DecoderEvalMode.MLM_HEAD:
        return True
    if config.decoder_eval_mode is DecoderEvalMode.LEFT_CONTEXT:
        return False
    return PROBLEM_DECODER_HEAD_POLICY[problem]


def prediction_objective_for_problem(config: ExperimentConfig, problem: str) -> Optional[str]:
    if config.decoder_eval_mode is DecoderEvalMode.NONE:
        return None
    if config.decoder_eval_mode is DecoderEvalMode.LEFT_CONTEXT:
        return "clm_next_token"
    if config.decoder_eval_mode is DecoderEvalMode.MLM_HEAD:
        return "clm_mlm_head"
    if needs_mlm_head_for_problem(config, problem):
        return "clm_mlm_head"
    return "clm_next_token"


def shared_mlm_head_path(config: ExperimentConfig, *, problem: str) -> Optional[str]:
    if config.mlm_head_scope is MlmHeadScope.GLOBAL:
        return os.path.join(config.args.experiment_dir, "decoder_mlm_heads", "global")
    if config.mlm_head_scope is MlmHeadScope.PER_GROUP:
        group = MLM_HEAD_GROUP_BY_PROBLEM[problem]
        return os.path.join(config.args.experiment_dir, "decoder_mlm_heads", group)
    return None


def _install_shared_mlm_head(trainer: TextPredictionTrainer, shared_path: str) -> None:
    dest = resolve_decoder_mlm_head_dir(trainer.experiment_dir)
    if dest is None:
        raise ValueError(f"Cannot resolve decoder MLM head directory for trainer {trainer.run_id!r}")
    if os.path.normcase(dest) == os.path.normcase(shared_path):
        return
    if has_saved_decoder_mlm_head(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(dest):
        return
    if os.name == "nt":
        shutil.copytree(shared_path, dest)
    else:
        os.symlink(shared_path, dest, target_is_directory=True)


def _merge_decoder_mlm_training_data(
    trainers: Iterable[TextPredictionTrainer],
    *,
    max_size: Optional[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for trainer in trainers:
        frames.append(trainer.get_decoder_mlm_training_data(max_size=max_size))
    if not frames:
        raise ValueError("No trainers provided for shared MLM-head training data")
    return pd.concat(frames, ignore_index=True)


def _train_shared_mlm_head(
    config: ExperimentConfig,
    *,
    problem: str,
    trainers: Sequence[TextPredictionTrainer],
) -> str:
    output_path = shared_mlm_head_path(config, problem=problem)
    if output_path is None:
        raise ValueError("shared MLM-head path requires per_group or global scope")
    if has_saved_decoder_mlm_head(output_path):
        print(f"  reusing shared decoder MLM head at {output_path}")
        return output_path
    print(f"  training shared decoder MLM head for {problem!r} -> {output_path}")
    train_df = _merge_decoder_mlm_training_data(trainers, max_size=config.mlm_head_args["max_size"])
    train_mlm_head(
        base_model=config.model_name,
        train_df=train_df,
        output_path=output_path,
        batch_size=config.mlm_head_args["batch_size"],
        epochs=config.mlm_head_args["epochs"],
        lr=config.mlm_head_args["lr"],
        use_cache=False,
    )
    return output_path


def prepare_decoder_resources(
    config: ExperimentConfig,
    *,
    problem: str,
    trainers: Sequence[TextPredictionTrainer],
) -> None:
    if not needs_mlm_head_for_problem(config, problem):
        return
    if config.mlm_head_scope is MlmHeadScope.PER_RUN:
        for trainer in trainers:
            print(f"  training decoder-only custom MLM head for {trainer.run_id}")
            trainer.train_decoder_only_mlm_head(config.model_name, **config.mlm_head_args)
        return
    if config.mlm_head_scope is MlmHeadScope.GLOBAL:
        shared_path = shared_mlm_head_path(config, problem=problem)
        if shared_path is None:
            raise ValueError("global MLM-head path could not be resolved")
        for trainer in trainers:
            _install_shared_mlm_head(trainer, shared_path)
        return
    shared_path = _train_shared_mlm_head(config, problem=problem, trainers=trainers)
    for trainer in trainers:
        _install_shared_mlm_head(trainer, shared_path)


def prepare_global_mlm_head(
    config: ExperimentConfig,
    trainers: Sequence[TextPredictionTrainer],
) -> None:
    if config.decoder_eval_mode is DecoderEvalMode.NONE:
        return
    if config.mlm_head_scope is not MlmHeadScope.GLOBAL:
        return
    if config.decoder_eval_mode is DecoderEvalMode.LEFT_CONTEXT:
        return
    if config.decoder_eval_mode is DecoderEvalMode.MLM_HEAD:
        relevant = list(trainers)
    else:
        relevant = [
            trainer
            for trainer in trainers
            if needs_mlm_head_for_problem(config, _problem_for_trainer(trainer))
        ]
    if not relevant:
        return
    _train_shared_mlm_head(config, problem="global", trainers=relevant)


def _problem_for_trainer(trainer: TextPredictionTrainer) -> str:
    run_id = str(trainer.run_id or "")
    if run_id.startswith("gender_de_"):
        return "gender_de"
    if run_id == "gender_en":
        return "gender_en"
    if run_id.startswith("race_") or run_id.startswith("religion_"):
        return "train_race_religion"
    if run_id.startswith("pronoun_number_") or run_id.startswith("pronoun_person_"):
        return "pronoun_merged"
    if run_id.startswith("pronoun_"):
        return "pronoun"
    if run_id.startswith("sentiment_"):
        return "sentiment"
    if run_id.startswith("formality_"):
        return "formality"
    return "pronoun"


def _pronoun_data_paths() -> Tuple[Path, Path]:
    return ensure_english_pronoun_data(output_dir=PRONOUN_DATA_DIR)


def _require_local_prediction_data(data_dir: str, *, label: str, creation_module: str) -> Tuple[Path, Path]:
    base = Path(data_dir)
    training_path = base / "training.csv"
    neutral_path = base / "neutral.csv"
    if not training_path.is_file() or not neutral_path.is_file():
        raise FileNotFoundError(
            f"{label} data not found at {data_dir}. "
            f"Run python -m gradiend.examples.{creation_module} to generate training.csv and neutral.csv."
        )
    return training_path, neutral_path


def _suite_common_kwargs(
    config: ExperimentConfig,
    problem: str,
    *,
    retain_models_in_memory: bool,
    **overrides: Any,
) -> Dict[str, Any]:
    objective = prediction_objective_for_problem(config, problem)
    args = problem_args(config, **overrides)
    if objective is not None:
        args = replace(args, prediction_objective=objective)
    return {
        "model": config.model_name,
        "args": args,
        "retain_models_in_memory": retain_models_in_memory,
    }


def _train_suite(
    config: ExperimentConfig,
    *,
    problem: str,
    suite: SymmetricTrainerSuite,
    models_for_heatmap: Dict[str, Any],
) -> None:
    trainers = list(suite.values())
    prepare_decoder_resources(config, problem=problem, trainers=trainers)
    suite.train(use_cache=config.args.use_cache)
    for child_id, trainer in suite.items():
        stats = trainer.get_training_stats()
        ts = stats.get("training_stats", {}) if stats else {}
        print(f"  {child_id}: correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
        trainer.cpu()
        models_for_heatmap[child_id] = trainer.get_model()


def build_gender_de_suite(config: ExperimentConfig, *, retain_models_in_memory: bool) -> SymmetricTrainerSuite:
    pair_definitions = [
        SuitePairDefinition(
            target_classes=pair,
            child_id=f"gender_de_{pair[0]}_{pair[1]}",
            label=f"{pair[0]} <-> {pair[1]}",
        )
        for pairs in configs_de.values()
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


def build_train_race_religion_suite(
    config: ExperimentConfig,
    *,
    retain_models_in_memory: bool,
) -> Tuple[SymmetricTrainerSuite, SymmetricTrainerSuite]:
    pair_definitions = [
        SuitePairDefinition(
            target_classes=pair,
            child_id=f"{bias_type}_{pair[0]}_{pair[1]}",
            label=f"{pair[0]} <-> {pair[1]}",
        )
        for bias_type, pair, _other_classes in race_configs + religion_configs
    ]
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data="aieng-lab/gradiend_race_data",
        masked_col="masked",
        eval_neutral_data="aieng-lab/biasneutral",
        pair_definitions=pair_definitions[:3],
        **_suite_common_kwargs(config, "train_race_religion", retain_models_in_memory=retain_models_in_memory),
    ), SymmetricTrainerSuite(
        TextPredictionTrainer,
        data="aieng-lab/gradiend_religion_data",
        masked_col="masked",
        eval_neutral_data="aieng-lab/biasneutral",
        pair_definitions=pair_definitions[3:],
        **_suite_common_kwargs(config, "train_race_religion", retain_models_in_memory=retain_models_in_memory),
    )


def build_pronoun_suite(config: ExperimentConfig, *, retain_models_in_memory: bool) -> SymmetricTrainerSuite:
    training_path, neutral_path = _pronoun_data_paths()
    pair_definitions = [
        SuitePairDefinition(
            target_classes=(c1, c2),
            child_id=f"pronoun_{c1}_{c2}",
            label=f"{c1} <-> {c2}",
        )
        for c1, c2 in pronoun_pairs
    ]
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data=str(training_path),
        all_classes=PRONOUN_CLASSES,
        masked_col="masked",
        split_col="split",
        eval_neutral_data=str(neutral_path),
        pair_definitions=pair_definitions,
        **_suite_common_kwargs(
            config,
            "pronoun",
            retain_models_in_memory=retain_models_in_memory,
            add_identity_for_other_classes=False,
            learning_rate=1e-4,
        ),
    )


def build_pronoun_merged_suite(config: ExperimentConfig, *, retain_models_in_memory: bool) -> SymmetricTrainerSuite:
    training_path, neutral_path = _pronoun_data_paths()
    pair_definitions = []
    for run_id_prefix, class_merge_map, _label, transition_group in pronoun_merged_configs:
        merged_keys = list(class_merge_map.keys())
        pair_definitions.append(
            SuitePairDefinition(
                target_classes=(merged_keys[0], merged_keys[1]),
                child_id=run_id_prefix,
                label=f"{merged_keys[0]} <-> {merged_keys[1]}",
                class_merge_map=class_merge_map,
                class_merge_transition_groups=transition_group,
            )
        )
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data=str(training_path),
        masked_col="masked",
        split_col="split",
        eval_neutral_data=str(neutral_path),
        pair_definitions=pair_definitions,
        **_suite_common_kwargs(
            config,
            "pronoun_merged",
            retain_models_in_memory=retain_models_in_memory,
            learning_rate=1e-5,
        ),
    )


def build_sentiment_suite(config: ExperimentConfig, *, retain_models_in_memory: bool) -> SymmetricTrainerSuite:
    training_path, neutral_path = _require_local_prediction_data(
        SENTIMENT_DATA_DIR,
        label="Sentiment",
        creation_module="sentiment",
    )
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data=str(training_path),
        all_classes=SENTIMENT_CLASSES,
        masked_col="masked",
        split_col=None,
        split_group_key=[str.strip, str.casefold],
        split_ratios=(0.6, 0.2, 0.2),
        eval_neutral_data=str(neutral_path),
        pair_definitions=[
            SuitePairDefinition(
                target_classes=("positive", "negative"),
                child_id="sentiment_positive_negative",
                label="positive <-> negative",
            )
        ],
        **_suite_common_kwargs(
            config,
            "sentiment",
            retain_models_in_memory=retain_models_in_memory,
            train_batch_size=8,
            eval_steps=100,
            max_steps=150,
            encoder_eval_train_max_size=None,
            learning_rate=1e-4,
            pre_prune_config=None,
            post_prune_config=PostPruneConfig(topk=0.001, part="decoder-weight"),
        ),
    )


def build_formality_suite(config: ExperimentConfig, *, retain_models_in_memory: bool) -> SymmetricTrainerSuite:
    training_path, neutral_path = _require_local_prediction_data(
        FORMALITY_DATA_DIR,
        label="Formality",
        creation_module="formality",
    )
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data=str(training_path),
        all_classes=FORMALITY_CLASSES,
        masked_col="masked",
        split_col="split",
        eval_neutral_data=str(neutral_path),
        pair_definitions=[
            SuitePairDefinition(
                target_classes=("informal", "formal"),
                child_id="formality_informal_formal",
                label="informal <-> formal",
            )
        ],
        **_suite_common_kwargs(config, "formality", retain_models_in_memory=retain_models_in_memory),
    )


def build_gender_en_trainer(config: ExperimentConfig) -> TextPredictionTrainer:
    from gradiend.examples.train_gender_en import build_gender_trainer

    trainer = build_gender_trainer(
        model=config.model_name,
        names_per_template=4,
        args=problem_args(
            config,
            train_batch_size=32,
            encoder_eval_max_size=10,
            eval_steps=25,
            max_steps=100,
            source="alternative",
            learning_rate=1e-4,
            pre_prune_config=None,
            post_prune_config=None,
        ),
    )
    objective = prediction_objective_for_problem(config, "gender_en")
    if objective is not None:
        trainer.training_args.prediction_objective = objective
    return trainer


def train_gender_en(
    config: ExperimentConfig,
    models_for_heatmap: Dict[str, Any],
    *,
    trainer: TextPredictionTrainer,
) -> TextPredictionTrainer:
    print("=== Gender EN: M vs F ===")
    prepare_decoder_resources(config, problem="gender_en", trainers=[trainer])
    trainer.train()
    stats = trainer.get_training_stats()
    ts = stats.get("training_stats", {}) if stats else {}
    print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
    trainer.cpu()
    models_for_heatmap[trainer.run_id] = trainer.get_model()
    return trainer


def train_all(
    config: ExperimentConfig,
    *,
    pronoun_suite: SymmetricTrainerSuite,
    pronoun_merged_suite: SymmetricTrainerSuite,
    race_suite: SymmetricTrainerSuite,
    religion_suite: SymmetricTrainerSuite,
    gender_de_suite: SymmetricTrainerSuite,
    gender_en_trainer: TextPredictionTrainer,
    sentiment_suite: SymmetricTrainerSuite,
    formality_suite: Optional[SymmetricTrainerSuite] = None,
) -> Dict[str, Any]:
    models_for_heatmap: Dict[str, Any] = {}
    os.makedirs(config.args.experiment_dir, exist_ok=True)

    preflight_suites = [
        pronoun_suite,
        pronoun_merged_suite,
        race_suite,
        religion_suite,
        gender_de_suite,
        sentiment_suite,
    ]
    if formality_suite is not None:
        preflight_suites.append(formality_suite)
    all_preflight_trainers = list(collect_trainers_from_suites(*preflight_suites).values()) + [gender_en_trainer]
    prepare_global_mlm_head(config, all_preflight_trainers)

    print("=== Sentiment (masked emotion words) ===")
    _train_suite(config, problem="sentiment", suite=sentiment_suite, models_for_heatmap=models_for_heatmap)

    if formality_suite is not None:
        print("=== Formality (GYAFC single-word substitutions) ===")
        _train_suite(config, problem="formality", suite=formality_suite, models_for_heatmap=models_for_heatmap)
    else:
        print("=== Formality: skipped (ENABLE_FORMALITY=False) ===")

    print("=== English pronouns ===")
    _train_suite(config, problem="pronoun", suite=pronoun_suite, models_for_heatmap=models_for_heatmap)

    print("=== English pronoun merged groups ===")
    _train_suite(
        config,
        problem="pronoun_merged",
        suite=pronoun_merged_suite,
        models_for_heatmap=models_for_heatmap,
    )

    print("=== Race ===")
    _train_suite(config, problem="train_race_religion", suite=race_suite, models_for_heatmap=models_for_heatmap)
    print("=== Religion ===")
    _train_suite(config, problem="train_race_religion", suite=religion_suite, models_for_heatmap=models_for_heatmap)

    train_gender_en(config, models_for_heatmap, trainer=gender_en_trainer)

    print("=== German gender/case ===")
    _train_suite(config, problem="gender_de", suite=gender_de_suite, models_for_heatmap=models_for_heatmap)

    return models_for_heatmap


def _pretty_label(mid: str) -> str:
    ARR = r"$\longleftrightarrow$"
    CASE_PRETTY = {
        "masc_nom": "Masc.Nom",
        "fem_nom": "Fem.Nom",
        "neut_nom": "Neut.Nom",
        "masc_acc": "Masc.Acc",
        "fem_acc": "Fem.Acc",
        "neut_acc": "Neut.Acc",
        "masc_dat": "Masc.Dat",
        "fem_dat": "Fem.Dat",
        "neut_dat": "Neut.Dat",
        "masc_gen": "Masc.Gen",
        "fem_gen": "Fem.Gen",
        "neut_gen": "Neut.Gen",
    }
    if mid == "gender_en":
        return f"he{ARR}she"
    if mid == "sentiment_positive_negative":
        return f"Pos{ARR}Neg"
    if mid == "formality_informal_formal":
        return f"Inf{ARR}Form"
    if mid.startswith("gender_de_"):
        rest = mid.replace("gender_de_", "")
        parts = rest.split("_")
        if len(parts) >= 4:
            a, b = "_".join(parts[:2]), "_".join(parts[2:])
            return f"{CASE_PRETTY.get(a, a)}{ARR}{CASE_PRETTY.get(b, b)}"
    if mid.startswith("pronoun_number_"):
        return f"SG{ARR}PL"
    if mid.startswith("pronoun_person_"):
        rest = mid.replace("pronoun_person_", "")
        person_pretty = {"1vs2": f"1st{ARR}2nd", "1vs3": f"1st{ARR}3rd", "2vs3": f"2nd{ARR}3rd"}
        return person_pretty.get(rest, rest.replace("vs", ARR))
    if mid.startswith("pronoun_"):
        c1, c2 = mid.replace("pronoun_", "").split("_")
        return f"{c1}{ARR}{c2}"
    if mid.startswith("race_"):
        w1, w2 = mid.replace("race_", "").split("_")
        return f"{w1.capitalize()}{ARR}{w2.capitalize()}"
    if mid.startswith("religion_"):
        w1, w2 = mid.replace("religion_", "").split("_")
        return f"{w1.capitalize()}{ARR}{w2.capitalize()}"
    return mid


def _build_plot_order_and_groups(all_ids: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    gender_de_ids = sorted([m for m in all_ids if m.startswith("gender_de_")])
    pronoun_ids = sorted(
        [
            m
            for m in all_ids
            if m.startswith("pronoun_")
            and not m.startswith("pronoun_number_")
            and not m.startswith("pronoun_person_")
        ]
    )
    pronoun_number_ids = sorted([m for m in all_ids if m.startswith("pronoun_number_")])
    pronoun_person_ids = sorted([m for m in all_ids if m.startswith("pronoun_person_")])
    race_ids = sorted([m for m in all_ids if m.startswith("race_")])
    religion_ids = sorted([m for m in all_ids if m.startswith("religion_")])
    sentiment_ids = sorted([m for m in all_ids if m.startswith("sentiment_")])
    formality_ids = sorted([m for m in all_ids if m.startswith("formality_")])
    gender_en_ids = [m for m in all_ids if m == "gender_en"]
    ordered = (
        gender_de_ids
        + gender_en_ids
        + sentiment_ids
        + formality_ids
        + pronoun_ids
        + pronoun_number_ids
        + pronoun_person_ids
        + race_ids
        + religion_ids
    )

    ARTICLE_MAPPING = {
        "masc_nom": "der",
        "fem_nom": "die",
        "neut_nom": "das",
        "masc_acc": "den",
        "fem_acc": "die",
        "neut_acc": "das",
        "masc_dat": "dem",
        "fem_dat": "der",
        "neut_dat": "dem",
        "masc_gen": "des",
        "fem_gen": "der",
        "neut_gen": "des",
    }
    ARR = r"$\longleftrightarrow$"
    gender_de_ids_to_articles = {
        model_id: ARR.join(
            sorted(
                ARTICLE_MAPPING["_".join(pair)]
                for pair in zip(*[iter(model_id.removeprefix("gender_de_").split("_"))] * 2)
            )
        )
        for model_id in gender_de_ids
    }
    gender_de_transitions_to_ids: Dict[str, List[str]] = defaultdict(list)
    for model_id, transition in gender_de_ids_to_articles.items():
        gender_de_transitions_to_ids[transition].append(model_id)

    pretty_groups = {
        **gender_de_transitions_to_ids,
        "Race": race_ids,
        "Religion": religion_ids,
        "Sentiment": sentiment_ids,
        "Formality": formality_ids,
        "English Gender": gender_en_ids,
        "English Pronouns": pronoun_ids,
        "English Number": pronoun_number_ids,
        "English Person": pronoun_person_ids,
    }
    return ordered, pretty_groups


def _build_feature_plot_groups() -> Dict[str, List[str]]:
    groups = {
        "German Case": GERMAN_CASE_CLASSES,
        "English Gender": ["M", "F"],
        "Sentiment": SENTIMENT_CLASSES,
        "English Pronouns": PRONOUN_CLASSES,
        "Race": ["white", "black", "asian"],
        "Religion": ["christian", "muslim", "jewish"],
    }
    if ENABLE_FORMALITY:
        groups["Formality"] = FORMALITY_CLASSES
    return {
        label: [feature for feature in features if feature in MULTILINGUAL_FEATURE_CLASSES]
        for label, features in groups.items()
        if any(feature in MULTILINGUAL_FEATURE_CLASSES for feature in features)
    }


def plot_results(
    config: ExperimentConfig,
    models_for_heatmap: Dict[str, Any],
    trainers_by_id: Dict[str, TextPredictionTrainer],
) -> None:
    all_ids = list(models_for_heatmap.keys())
    ordered, pretty_groups = _build_plot_order_and_groups(all_ids)
    order = [mid for gids in pretty_groups.values() for mid in gids if mid in models_for_heatmap]
    pretty_labels = {mid: _pretty_label(mid) for mid in order}
    models_display = {pretty_labels[mid]: models_for_heatmap[mid] for mid in order}
    order_display = [pretty_labels[mid] for mid in order]
    pretty_groups_display = {
        group: [pretty_labels[mid] for mid in gids if mid in pretty_labels]
        for group, gids in pretty_groups.items()
        if any(mid in pretty_labels for mid in gids)
    }

    topk = 1000
    output_path = os.path.join(config.args.experiment_dir, f"topk_overlap_heatmap_all_{topk}.pdf")
    plot_topk_overlap_heatmap(
        models_display,
        topk=topk,
        part="decoder-weight",
        value="intersection_frac",
        order=order_display,
        annot="auto",
        output_path=output_path,
        show=True,
        pretty_groups=pretty_groups_display,
        scale="linear",
        group_label_fontsize=16,
        scale_gamma=0.5,
        tick_label_fontsize=11,
        annot_fontsize=9,
        percentages=True,
        cbar_pad=0.1,
        cbar_fontsize=18,
    )
    print(f"Heatmap saved to {output_path}")

    feature_groups = _build_feature_plot_groups()
    grouped_features = [feature for features in feature_groups.values() for feature in features]
    feature_order = grouped_features + [
        feature for feature in MULTILINGUAL_FEATURE_CLASSES if feature not in grouped_features
    ]
    for alignment, column_ids, title, suffix in (
        (
            "factual",
            feature_order,
            "Oriented cross-encoding aligned by factual feature",
            "factual",
        ),
        (
            "counterfactual",
            feature_order,
            "Oriented cross-encoding aligned by counterfactual feature",
            "counterfactual",
        ),
        (
            "transition",
            None,
            "Oriented cross-encoding aligned by directed transition",
            "transition",
        ),
    ):
        cross_encoding_output = os.path.join(
            config.args.experiment_dir,
            f"cross_encoding_oriented_{suffix}_heatmap.pdf",
        )
        plot_anchor_aligned_encoding_heatmap(
            trainers_by_id,
            feature_order,
            alignment=alignment,
            column_ids=column_ids,
            split="test",
            max_size=config.args.encoder_eval_max_size,
            aggregate="mean",
            order=feature_order,
            pretty_groups=feature_groups,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            output_path=cross_encoding_output,
            title=title,
            annot=False,
            tick_label_fontsize=8,
            show=True,
        )
        print(f"Oriented cross-encoding heatmap ({alignment}) saved to {cross_encoding_output}")

    pronoun_ids = [mid for mid in order if mid.startswith("pronoun_") and not mid.startswith("pronoun_number_") and not mid.startswith("pronoun_person_")]
    venn_pronoun_ids = [mid for mid in pronoun_ids if mid in ("pronoun_1SG_3PL", "pronoun_1SG_3SG", "pronoun_3SG_3PL")]
    if len(venn_pronoun_ids) < 3:
        venn_pronoun_ids = pronoun_ids[:3]
    if len(venn_pronoun_ids) >= 3:
        venn_models = {pretty_labels[mid]: models_for_heatmap[mid] for mid in venn_pronoun_ids}
        venn_output = os.path.join(config.args.experiment_dir, "topk_overlap_venn_three_train_english_pronouns.pdf")
        plot_topk_overlap_venn(
            venn_models,
            topk=topk,
            part="decoder-weight",
            output_path=venn_output,
            show=True,
        )
        print(f"Venn plot (3 English pronouns) saved to {venn_output}")
    else:
        print("Skipping Venn plot: fewer than 3 pronoun GRADIENDs available.")


def collect_trainers_from_suites(*suites: SymmetricTrainerSuite) -> Dict[str, TextPredictionTrainer]:
    trainers: Dict[str, TextPredictionTrainer] = {}
    for suite in suites:
        for child_id, trainer in suite.items():
            trainers[child_id] = trainer
    return trainers


def main() -> None:
    cli_args = parse_args()
    config = build_experiment_config(cli_args)
    print(f"Model: {config.model_name}")
    print(f"Decoder eval mode: {config.decoder_eval_mode.value}")
    print(f"MLM head scope: {config.mlm_head_scope.value}")
    print(f"GRADIEND source: {config.args.source}")
    pre_cfg = config.args.pre_prune_config
    if pre_cfg is not None:
        print(f"Pre-prune: n_samples={pre_cfg.n_samples}, topk={pre_cfg.topk}, source={pre_cfg.source}")
    post_cfg = config.args.post_prune_config
    if post_cfg is not None:
        print(f"Post-prune: topk={post_cfg.topk}, part={post_cfg.part}")
    print(f"Experiment dir: {config.args.experiment_dir}")

    retain_models = cli_args.retain_models_in_memory
    pronoun_suite = build_pronoun_suite(config, retain_models_in_memory=retain_models)
    pronoun_merged_suite = build_pronoun_merged_suite(config, retain_models_in_memory=retain_models)
    race_suite, religion_suite = build_train_race_religion_suite(config, retain_models_in_memory=retain_models)
    gender_de_suite = build_gender_de_suite(config, retain_models_in_memory=retain_models)
    gender_en_trainer = build_gender_en_trainer(config)
    sentiment_suite = build_sentiment_suite(config, retain_models_in_memory=retain_models)
    formality_suite = (
        build_formality_suite(config, retain_models_in_memory=retain_models)
        if ENABLE_FORMALITY
        else None
    )

    models_for_heatmap = train_all(
        config,
        pronoun_suite=pronoun_suite,
        pronoun_merged_suite=pronoun_merged_suite,
        race_suite=race_suite,
        religion_suite=religion_suite,
        gender_de_suite=gender_de_suite,
        gender_en_trainer=gender_en_trainer,
        sentiment_suite=sentiment_suite,
        formality_suite=formality_suite,
    )

    trainer_suites = [
        pronoun_suite,
        pronoun_merged_suite,
        race_suite,
        religion_suite,
        gender_de_suite,
        sentiment_suite,
    ]
    if formality_suite is not None:
        trainer_suites.append(formality_suite)
    trainers_by_id = collect_trainers_from_suites(*trainer_suites)
    trainers_by_id[gender_en_trainer.run_id] = gender_en_trainer

    plot_results(config, models_for_heatmap, trainers_by_id)


if __name__ == "__main__":
    main()
