"""
Multilingual GRADIEND comparison for system demo: one large heatmap.

Trains GRADIENDs for:
- German gender/case (train_gender_de_detailed configs: multiple pairs)
- English gender (gender_en: one GRADIEND M vs F)
- Race (3 GRADIENDs: white-black, white-asian, black-asian)
- Religion (3 GRADIENDs: christian-muslim, christian-jewish, muslim-jewish)
- English pronouns (10 GRADIENDs: all pairs from 1SG, 1PL, 2SGPL, 3SG, 3PL)
- English pronoun merged groups (4 GRADIENDs: Number SG vs PL; Person 1vs2, 1vs3, 2vs3)
- English formality (disabled until GYAFC data is available; see ENABLE_FORMALITY)
- English sentiment last (exactly two GRADIENDs: the full NRC positive↔negative
  sentiment task and the single-word good↔bad task)

Uses a single multilingual model and the same TrainingArguments as
train_gender_de_detailed. Encoder mode defaults to multilingual BERT; decoder-only
modes default to XGLM-564M. Pairwise training uses SymmetricTrainerSuite where
possible; heterogeneous groups are combined with TrainerCollection.merge.

At the end, plots in the experiment directory root:
- topk_overlap_heatmap_all_1000.pdf
- cross_encoding_gradiend_by_transition_heatmap.pdf
  (GRADIEND × directed input transition, pre-anchor)
- cross_encoding_oriented_factual_heatmap.pdf
  (dense oriented cross-encoding; columns = factual feature class $s(x)$)
- topk_overlap_venn_three_train_english_pronouns.pdf (when enough pronoun runs exist)

Usage:
  python experiments/multilingual_gradiend_demo.py
  python experiments/multilingual_gradiend_demo.py --plot-only
  python experiments/multilingual_gradiend_demo.py --decoder-eval-mode mlm_head
  python experiments/multilingual_gradiend_demo.py --decoder-eval-mode default \\
      --problem-learning-rate pronoun=5e-5 --problem-learning-rate sentiment=2e-5
  python experiments/multilingual_gradiend_demo.py --problems gender_de
  python experiments/multilingual_gradiend_demo.py --problems gender_de,pronoun
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch

from gradiend.comparison.seed_policy import enter_analysis_mode, enter_analysis_mode_for_trainers
from gradiend import (
    SuitePairDefinition,
    SymmetricTrainerSuite,
    TextPredictionConfig,
    TextPredictionDataCreator,
    TextPredictionTrainer,
    TrainerCollection,
    TrainingArguments,
    plot_cross_encoding_heatmap,
    plot_gradiend_transition_cross_encoding_heatmap,
    plot_topk_overlap_heatmap,
    plot_topk_overlap_venn,
)
from gradiend.comparison.feature_cross_encoding import (
    build_cross_task_encoder_summary,
    collect_unified_test_rows,
    collect_unified_test_transitions,
)
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.examples.create_english_pronoun_data import (
    PRONOUN_CLASSES,
    ensure_english_pronoun_data,
)
from gradiend.trainer.text.prediction.decoder_only_mlm import train_mlm_head
from gradiend.util.paths import has_saved_decoder_mlm_head, resolve_decoder_mlm_head_dir


ENCODER_MODEL = "google-bert/bert-base-multilingual-cased"
DECODER_MODEL = "meta-llama/Llama-3.2-1B"
DECODER_MODEL = "gpt2"
DECODER_MODEL = "Qwen/Qwen2.5-0.5B"
DECODER_MODEL_TORCH_DTYPE = torch.bfloat16


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

# Per-problem GRADIEND training learning rates (encoder / multilingual BERT defaults).
ENCODER_LEARNING_RATE_BY_PROBLEM: Dict[str, float] = {
    "gender_de": 1e-5,
    "gender_en": 1e-4,
    "train_race_religion": 1e-5,
    "pronoun": 1e-4,
    "pronoun_merged": 1e-5,
    "sentiment": 1e-4,
    "formality": 1e-5,
}

# Decoder-only models (XGLM, Llama, …) typically need different rates per task.
DECODER_LEARNING_RATE_BY_PROBLEM: Dict[str, float] = {
    "gender_de": 1e-5,
    "gender_en": 2e-5,
    "train_race_religion": 1e-5,
    "pronoun": 1e-5,
    "pronoun_merged": 1e-5,
    "sentiment": 5e-6,
    "formality": 5e-6,
}

PROBLEM_KEYS_FOR_LEARNING_RATE = tuple(ENCODER_LEARNING_RATE_BY_PROBLEM.keys())

DEMO_PROBLEM_KEYS = (
    "formality",
    "pronoun",
    "pronoun_merged",
    "race",
    "religion",
    "gender_en",
    "gender_de",
    "sentiment",
)

DECODER_MLM_HEAD_ARGS = {
    "batch_size": 4,
    "epochs": 5,
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
SENTIMENT_GOOD_BAD_DATA_DIR = "data/sentiment_nrc_good_bad"
SENTIMENT_GOOD_BAD_PAIR = ("good", "bad")
SENTIMENT_PRE_PRUNE_TOPK = 0.1
SENTIMENT_POST_PRUNE_TOPK = 0.01
FORMALITY_DATA_DIR = "data/formality_gyafc"
# Re-enable when training.csv and neutral.csv exist under FORMALITY_DATA_DIR.
ENABLE_FORMALITY = False
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


SuiteLike = Union[SymmetricTrainerSuite, TrainerCollection]


class ExperimentConfig:
    model_name: str
    decoder_eval_mode: DecoderEvalMode
    mlm_head_scope: MlmHeadScope
    args: TrainingArguments
    mlm_head_args: Dict[str, Any]
    problem_learning_rate_overrides: Dict[str, float]

    def __init__(
        self,
        *,
        model_name: str,
        decoder_eval_mode: DecoderEvalMode,
        mlm_head_scope: MlmHeadScope,
        args: TrainingArguments,
        mlm_head_args: Dict[str, Any],
        problem_learning_rate_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model_name = model_name
        self.decoder_eval_mode = decoder_eval_mode
        self.mlm_head_scope = mlm_head_scope
        self.args = args
        self.mlm_head_args = mlm_head_args
        self.problem_learning_rate_overrides = dict(problem_learning_rate_overrides or {})


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
        default=0.1,
        help="Pre-prune keep fraction (float in (0,1]) or absolute dim count (int). Default: 0.01.",
    )
    parser.add_argument(
        "--post-prune-topk",
        type=float,
        default=0.01,
        help="Post-prune keep fraction after training. Default: 0.05 with pre-prune 0.01, else 0.01.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; plot from cached checkpoints in the experiment dir.",
    )
    parser.add_argument(
        "--problems",
        action="append",
        metavar="PROBLEM",
        default=[],
        help=(
            "Train only these problems (repeatable; comma-separated values ok). "
            f"Choices: {', '.join(DEMO_PROBLEM_KEYS)}. "
            "train_race_religion is accepted as shorthand for race+religion. "
            "Default: train all enabled problems."
        ),
    )
    parser.add_argument(
        "--problem-learning-rate",
        action="append",
        metavar="PROBLEM=LR",
        default=[],
        help=(
            "Override GRADIEND learning rate for one problem (repeatable). "
            f"Problems: {', '.join(PROBLEM_KEYS_FOR_LEARNING_RATE)}. "
            "Example: --problem-learning-rate pronoun=5e-5"
        ),
    )
    parser.add_argument(
        "--three-seed",
        action="store_true",
        help=(
            "Run stability analysis with min_convergent_seeds=3, keep all convergent "
            "checkpoints, and emit additional std-only heatmap variants."
        ),
    )
    return parser.parse_args()


def _parse_problems(specs: Sequence[str]) -> Optional[FrozenSet[str]]:
    if not specs:
        return None
    selected: set[str] = set()
    for raw in specs:
        for part in raw.split(","):
            key = part.strip()
            if not key:
                continue
            if key == "train_race_religion":
                selected.update({"race", "religion"})
                continue
            if key not in DEMO_PROBLEM_KEYS:
                raise ValueError(
                    f"Unknown problem {key!r} in --problems. "
                    f"Expected one of: {', '.join(DEMO_PROBLEM_KEYS)} "
                    "(or train_race_religion for race+religion)."
                )
            selected.add(key)
    if not selected:
        return None
    return frozenset(selected)


def _problem_selected(
    selected_problems: Optional[FrozenSet[str]],
    problem: str,
) -> bool:
    if selected_problems is None:
        return True
    return problem in selected_problems


def _learning_rate_problem_keys(
    selected_problems: Optional[FrozenSet[str]],
) -> Sequence[str]:
    if selected_problems is None:
        return PROBLEM_KEYS_FOR_LEARNING_RATE
    keys: List[str] = []
    for problem in sorted(selected_problems):
        if problem in ("race", "religion"):
            lr_key = "train_race_religion"
        else:
            lr_key = problem
        if lr_key in PROBLEM_KEYS_FOR_LEARNING_RATE and lr_key not in keys:
            keys.append(lr_key)
    return keys


def _assert_selected_problems_runnable(
    selected_problems: Optional[FrozenSet[str]],
    *,
    pronoun_suite: Optional[SymmetricTrainerSuite],
    pronoun_merged_suite: Optional[SymmetricTrainerSuite],
    race_suite: Optional[SymmetricTrainerSuite],
    religion_suite: Optional[SymmetricTrainerSuite],
    gender_de_suite: Optional[SymmetricTrainerSuite],
    gender_en_trainer: Optional[TextPredictionTrainer],
    formality_suite: Optional[SymmetricTrainerSuite],
) -> None:
    if selected_problems is None:
        return
    if "formality" in selected_problems and not ENABLE_FORMALITY:
        raise ValueError(
            "--problems formality was requested, but formality is disabled "
            "(ENABLE_FORMALITY=False in multilingual_gradiend_demo.py)."
        )
    has_work = any(
        suite is not None
        for suite in (
            pronoun_suite,
            pronoun_merged_suite,
            race_suite,
            religion_suite,
            gender_de_suite,
            formality_suite,
        )
    ) or gender_en_trainer is not None or "sentiment" in selected_problems
    if not has_work:
        raise ValueError(
            f"No runnable problems in --problems {sorted(selected_problems)!r}. "
            f"Expected one of: {', '.join(DEMO_PROBLEM_KEYS)}."
        )


def _parse_problem_learning_rate_overrides(specs: Sequence[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --problem-learning-rate {raw!r}; expected PROBLEM=LR "
                f"(e.g. pronoun=5e-5). Known problems: {', '.join(PROBLEM_KEYS_FOR_LEARNING_RATE)}"
            )
        problem, lr_text = raw.split("=", 1)
        problem = problem.strip()
        if problem not in PROBLEM_KEYS_FOR_LEARNING_RATE:
            raise ValueError(
                f"Unknown problem {problem!r} in --problem-learning-rate. "
                f"Choose from: {', '.join(PROBLEM_KEYS_FOR_LEARNING_RATE)}"
            )
        overrides[problem] = float(lr_text.strip())
    return overrides


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
    three_seed = bool(getattr(cli_args, "three_seed", False))
    seed_suffix = "_3seed" if three_seed else ""
    experiment_dir = (
        f"runs/multilingual_gradiend_demo_{model_name.split('/')[-1]}"
        f"{mode_suffix}{pre_suffix}{seed_suffix}_v7"
    )

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
        max_seeds=20,
        min_convergent_seeds=3 if three_seed else 1,
        analyze_seed_stability=three_seed,
        saved_seed_runs="all_convergent",
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        torch_dtype=(
            DECODER_MODEL_TORCH_DTYPE
            if decoder_eval_mode is not DecoderEvalMode.NONE
            else None
        ),
        pre_prune_config=PrePruneConfig(n_samples=16, topk=pre_prune_topk, source="diff"),
        post_prune_config=PostPruneConfig(topk=post_prune_topk, part="decoder-weight"),
        add_identity_for_other_classes=True,
        use_cache="only_convergent",
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
        problem_learning_rate_overrides=_parse_problem_learning_rate_overrides(
            cli_args.problem_learning_rate
        ),
    )


def learning_rate_for_problem(config: ExperimentConfig, problem: str) -> float:
    """Return the GRADIEND training LR for a demo problem (encoder vs decoder table + CLI overrides)."""
    if problem in config.problem_learning_rate_overrides:
        return config.problem_learning_rate_overrides[problem]
    if config.decoder_eval_mode is not DecoderEvalMode.NONE:
        table = DECODER_LEARNING_RATE_BY_PROBLEM
    else:
        table = ENCODER_LEARNING_RATE_BY_PROBLEM
    return table.get(problem, config.args.learning_rate)


def problem_args(config: ExperimentConfig, **overrides: Any) -> TrainingArguments:
    if "train_batch_size" in overrides and "base_gradient_batch_size" not in overrides:
        overrides = dict(overrides)
        overrides["base_gradient_batch_size"] = overrides["train_batch_size"]
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
    if not has_saved_decoder_mlm_head(shared_path):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.lexists(dest):
        if os.path.islink(dest):
            os.unlink(dest)
        elif os.path.isdir(dest) and not os.listdir(dest):
            os.rmdir(dest)
        elif os.path.isdir(dest):
            return
        else:
            os.remove(dest)
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


def _assert_suite_has_expected_trainers(
    suite: SymmetricTrainerSuite,
    *,
    expected_count: int,
    suite_label: str,
) -> None:
    actual = len(suite.trainers)
    if actual == expected_count:
        return
    training_path, _ = _pronoun_data_paths()
    sidecar = training_path.with_name(f"{training_path.stem}_incomplete_classes{training_path.suffix}")
    sidecar_hint = (
        f" Found incomplete-class sidecar at {sidecar}."
        if sidecar.is_file()
        else ""
    )
    raise ValueError(
        f"{suite_label} expected {expected_count} trainers but built {actual}. "
        "TrainerSuite skips pair definitions when pronoun classes are listed in "
        f"{sidecar.name} or missing from training data.{sidecar_hint} "
        "Regenerate with ensure_english_pronoun_data(force=True) or delete the stale sidecar."
    )


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


def _sentiment_data_paths() -> Tuple[Path, Path]:
    base = Path(SENTIMENT_DATA_DIR)
    training_path = base / "training.csv"
    neutral_path = base / "neutral.csv"
    if training_path.is_file() and neutral_path.is_file():
        from gradiend.examples.train_sentiment import _normalize_training_labels

        cached = _normalize_training_labels(pd.read_csv(training_path))
        if len(cached):
            cached.to_csv(training_path, index=False)
        return training_path, neutral_path

    from gradiend.examples.train_sentiment import (
        DEFAULT_LEXICON_WORDS_PER_CLASS,
        DEFAULT_MAX_SIZE_PER_CLASS,
        DEFAULT_MIN_OCCURRENCES_PER_TARGET,
        generate_data,
    )

    print(f"=== Sentiment data: generating missing CSVs in {base} ===")
    return generate_data(
        output_dir=base,
        max_size_per_class=DEFAULT_MAX_SIZE_PER_CLASS,
        neutral_max_size=1000,
        lexicon_words_per_class=DEFAULT_LEXICON_WORDS_PER_CLASS,
        min_occurrences_per_target=DEFAULT_MIN_OCCURRENCES_PER_TARGET,
        seed=42,
    )


def _sentiment_single_pair_child_id(positive_word: str, negative_word: str) -> str:
    return f"sentiment_{positive_word}_{negative_word}"


def _unique_sentiment_target_words(words: Iterable[str]) -> List[str]:
    from gradiend.examples.train_sentiment import _canonical_target_word

    seen: set[str] = set()
    ordered: List[str] = []
    for word in words:
        key = _canonical_target_word(word)
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _subset_sentiment_training_for_pair(
    training_df: pd.DataFrame,
    positive_word: str,
    negative_word: str,
) -> pd.DataFrame:
    from gradiend.examples.train_sentiment import _canonical_target_word

    pos_key = _canonical_target_word(positive_word)
    neg_key = _canonical_target_word(negative_word)
    labels = training_df["label"].map(_canonical_target_word)
    mask = (
        (training_df["label_class"] == "positive") & (labels == pos_key)
    ) | (
        (training_df["label_class"] == "negative") & (labels == neg_key)
    )
    out = training_df.loc[mask].copy()
    if out.empty:
        raise ValueError(
            f"No training rows for sentiment pair {positive_word!r}/{negative_word!r} "
            "in the shared single-pair dataset."
        )
    return out


def _generate_sentiment_single_pairs_data(
    pairs: Sequence[Tuple[str, str]],
    *,
    output_dir: Path,
    max_size_per_class: int,
    neutral_max_size: int,
    min_occurrences_per_target: int,
    seed: int,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Shared tweet_eval pass for all single-word sentiment pairs (row-level splits)."""
    from gradiend.examples.nrc_sentiment_lexicon import NRC_CITATION, load_nrc_sentiment_words
    from gradiend.examples.train_sentiment import (
        _feature_targets,
        _load_tweet_eval_texts,
        _normalize_training_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training.csv"
    neutral_path = output_dir / "neutral.csv"
    pairs_path = output_dir / "pairs.json"

    positive_words = _unique_sentiment_target_words(pos for pos, _ in pairs)
    negative_words = _unique_sentiment_target_words(neg for _, neg in pairs)
    base_texts = _load_tweet_eval_texts()

    creator = TextPredictionDataCreator(
        base_data=base_texts,
        spacy_model="en_core_web_sm",
        feature_targets=_feature_targets(
            positive_words=positive_words,
            negative_words=negative_words,
        ),
        seed=seed,
        output_dir=str(output_dir),
        use_cache=False,
        download_if_missing=True,
    )

    print("=== Decoder-only sentiment single-pair data (tweet_eval + NRC ADJ) ===")
    print(f"  lexicon: {NRC_CITATION}")
    print(f"  pairs: {len(pairs)}")
    print(f"  positive words: {positive_words}")
    print(f"  negative words: {negative_words}")

    training_df = creator.generate_training_data(
        max_size_per_class=max_size_per_class,
        format="unified",
        balance="strict",
        min_rows_per_class_for_split=min(100, max_size_per_class // 2),
        min_rows_per_target_for_balance=min_occurrences_per_target,
        output=str(training_path),
    )
    training_df = _normalize_training_labels(training_df)
    if len(training_df):
        training_df.to_csv(training_path, index=False)

    nrc_positive, nrc_negative = load_nrc_sentiment_words()
    creator.generate_neutral_data(
        additional_excluded_words=nrc_positive + nrc_negative,
        max_size=neutral_max_size,
        output=str(neutral_path),
    )
    pairs_path.write_text(
        json.dumps([{"positive": pos, "negative": neg} for pos, neg in pairs], indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {training_path} ({len(training_df)} rows)")
    print(f"  wrote {neutral_path}")
    print(f"  wrote {pairs_path}")
    return training_df, training_path, neutral_path


def _sentiment_good_bad_training_frame(
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Path]:
    from gradiend.examples.train_sentiment import (
        DEFAULT_MIN_OCCURRENCES_PER_TARGET,
        _canonical_target_word,
        load_sentiment_training_data,
    )

    base = Path(SENTIMENT_GOOD_BAD_DATA_DIR)
    training_path = base / "training.csv"
    neutral_path = base / "neutral.csv"
    pairs_path = base / "pairs.json"
    pairs = [SENTIMENT_GOOD_BAD_PAIR]

    if not training_path.is_file() or not neutral_path.is_file() or not pairs_path.is_file():
        print(f"=== Sentiment good-vs-bad data: generating missing CSVs in {base} ===")
        training_df, training_path, neutral_path = _generate_sentiment_single_pairs_data(
            pairs,
            output_dir=base,
            max_size_per_class=500,
            neutral_max_size=500,
            min_occurrences_per_target=DEFAULT_MIN_OCCURRENCES_PER_TARGET,
            seed=int(config.args.seed or 0),
        )
        return training_df, neutral_path

    training_df = load_sentiment_training_data(training_path)
    recorded = json.loads(pairs_path.read_text(encoding="utf-8"))
    recorded_pairs = [(item["positive"], item["negative"]) for item in recorded]
    needed_words = {_canonical_target_word(word) for pair in pairs for word in pair}
    present_words = set(training_df["label"].map(_canonical_target_word))
    if recorded_pairs != pairs or not needed_words.issubset(present_words):
        print(f"=== Sentiment good-vs-bad data: refreshing stale cache in {base} ===")
        training_df, training_path, neutral_path = _generate_sentiment_single_pairs_data(
            pairs,
            output_dir=base,
            max_size_per_class=500,
            neutral_max_size=500,
            min_occurrences_per_target=DEFAULT_MIN_OCCURRENCES_PER_TARGET,
            seed=int(config.args.seed or 0),
        )
    return training_df, neutral_path


def _suite_common_kwargs(
    config: ExperimentConfig,
    problem: str,
    *,
    retain_models_in_memory: bool,
    **overrides: Any,
) -> Dict[str, Any]:
    objective = prediction_objective_for_problem(config, problem)
    if "learning_rate" not in overrides:
        overrides = dict(overrides)
        overrides["learning_rate"] = learning_rate_for_problem(config, problem)
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
    suite: SuiteLike,
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
        analysis_trainer = enter_analysis_mode(trainer)
        models_for_heatmap[child_id] = analysis_trainer.get_model(gradiend_only=True)


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
        split_col="split",
        eval_neutral_data="aieng-lab/biasneutral",
        pair_definitions=pair_definitions[:3],
        **_suite_common_kwargs(config, "train_race_religion", retain_models_in_memory=retain_models_in_memory),
    ), SymmetricTrainerSuite(
        TextPredictionTrainer,
        data="aieng-lab/gradiend_religion_data",
        masked_col="masked",
        split_col="split",
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
    suite = SymmetricTrainerSuite(
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
        ),
    )
    _assert_suite_has_expected_trainers(
        suite,
        expected_count=len(pronoun_pairs),
        suite_label="English pronoun binary suite",
    )
    return suite


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
    suite = SymmetricTrainerSuite(
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
        ),
    )
    _assert_suite_has_expected_trainers(
        suite,
        expected_count=len(pronoun_merged_configs),
        suite_label="English pronoun merged suite",
    )
    return suite


def build_sentiment_suite(
    config: ExperimentConfig,
    *,
    retain_models_in_memory: bool,
) -> TrainerCollection:
    full_suite = _build_sentiment_full_lexicon_suite(
        config,
        retain_models_in_memory=retain_models_in_memory,
    )
    good_bad_trainer = _build_sentiment_good_bad_trainer(config)
    return TrainerCollection.merge(
        full_suite,
        good_bad_trainer,
        retain_models_in_memory=retain_models_in_memory,
    )


def _build_sentiment_good_bad_trainer(config: ExperimentConfig) -> TextPredictionTrainer:
    training_df, neutral_path = _sentiment_good_bad_training_frame(config)
    objective = prediction_objective_for_problem(config, "sentiment")
    positive_word, negative_word = SENTIMENT_GOOD_BAD_PAIR
    child_id = _sentiment_single_pair_child_id(positive_word, negative_word)
    pair_df = _subset_sentiment_training_for_pair(training_df, positive_word, negative_word)
    args = problem_args(
        config,
        fail_on_non_convergence=False,
        learning_rate=learning_rate_for_problem(config, "sentiment"),
        pre_prune_config=PrePruneConfig(
            n_samples=16,
            topk=SENTIMENT_PRE_PRUNE_TOPK,
            source="diff",
        ),
        post_prune_config=PostPruneConfig(
            topk=SENTIMENT_POST_PRUNE_TOPK,
            part="decoder-weight",
        ),
    )
    if objective is not None:
        args = replace(args, prediction_objective=objective)
    return TextPredictionTrainer(
        model=config.model_name,
        config=TextPredictionConfig(
            run_id=child_id,
            data=pair_df,
            target_classes=list(SENTIMENT_CLASSES),
            eval_neutral_data=str(neutral_path),
            split_col="split",
        ),
        args=args,
    )


def _build_sentiment_full_lexicon_suite(
    config: ExperimentConfig,
    *,
    retain_models_in_memory: bool,
) -> SymmetricTrainerSuite:
    from gradiend.examples.train_sentiment import (
        load_and_split_sentiment_training_data,
        sentiment_training_arguments,
    )

    training_path, neutral_path = _sentiment_data_paths()
    sentiment_args = sentiment_training_arguments(
        experiment_dir=config.args.experiment_dir,
        use_cache=False,
        max_steps=15000,
        num_train_epochs=10,
        max_seeds=10,
        min_convergent_seeds=1,
        fail_on_non_convergence=False,
        torch_dtype=config.args.torch_dtype,
        learning_rate=learning_rate_for_problem(config, "sentiment"),
    )
    sentiment_args = replace(
        sentiment_args,
        pre_prune_config=PrePruneConfig(
            n_samples=16,
            topk=SENTIMENT_PRE_PRUNE_TOPK,
            source="diff",
        ),
        post_prune_config=PostPruneConfig(
            topk=SENTIMENT_POST_PRUNE_TOPK,
            part="decoder-weight",
        ),
    )
    objective = prediction_objective_for_problem(config, "sentiment")
    if objective is not None:
        sentiment_args = replace(sentiment_args, prediction_objective=objective)
    training_df = load_and_split_sentiment_training_data(
        training_path,
        seed=int(sentiment_args.seed or 0),
    )
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        data=training_df,
        all_classes=SENTIMENT_CLASSES,
        masked_col="masked",
        split_col="split",
        eval_neutral_data=str(neutral_path),
        pair_definitions=[
            SuitePairDefinition(
                target_classes=("positive", "negative"),
                child_id="sentiment_positive_negative",
                label="positive <-> negative",
            )
        ],
        model=config.model_name,
        args=sentiment_args,
        retain_models_in_memory=retain_models_in_memory,
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
            learning_rate=learning_rate_for_problem(config, "gender_en"),
            pre_prune_config=PrePruneConfig(topk=0.1, n_samples=4),
            post_prune_config=PostPruneConfig(topk=0.01),
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
    analysis_trainer = enter_analysis_mode(trainer)
    models_for_heatmap[trainer.run_id] = analysis_trainer.get_model(gradiend_only=True)
    return analysis_trainer


def train_all(
    config: ExperimentConfig,
    *,
    selected_problems: Optional[FrozenSet[str]] = None,
    pronoun_suite: Optional[SymmetricTrainerSuite] = None,
    pronoun_merged_suite: Optional[SymmetricTrainerSuite] = None,
    race_suite: Optional[SymmetricTrainerSuite] = None,
    religion_suite: Optional[SymmetricTrainerSuite] = None,
    gender_de_suite: Optional[SymmetricTrainerSuite] = None,
    gender_en_trainer: Optional[TextPredictionTrainer] = None,
    sentiment_suite: Optional[SuiteLike] = None,
    formality_suite: Optional[SymmetricTrainerSuite] = None,
    retain_models_in_memory: bool = False,
) -> Tuple[Dict[str, Any], Optional[SuiteLike]]:
    if config.decoder_eval_mode is not DecoderEvalMode.NONE:
        print("=== Decoder-only learning rates ===")
        for problem in _learning_rate_problem_keys(selected_problems):
            print(f"  {problem}: {learning_rate_for_problem(config, problem):g}")
    models_for_heatmap: Dict[str, Any] = {}
    os.makedirs(config.args.experiment_dir, exist_ok=True)

    preflight_suites = [
        suite
        for suite in (
            pronoun_suite,
            pronoun_merged_suite,
            race_suite,
            religion_suite,
            gender_de_suite,
            formality_suite,
        )
        if suite is not None
    ]
    preflight_parts: List[Any] = list(preflight_suites)
    if gender_en_trainer is not None:
        preflight_parts.append(gender_en_trainer)
    if preflight_parts:
        all_preflight_trainers = list(TrainerCollection.merge(*preflight_parts).values())
        prepare_global_mlm_head(config, all_preflight_trainers)

    if _problem_selected(selected_problems, "formality"):
        if formality_suite is not None:
            print("=== Formality (GYAFC single-word substitutions) ===")
            _train_suite(
                config,
                problem="formality",
                suite=formality_suite,
                models_for_heatmap=models_for_heatmap,
            )
        elif selected_problems is None:
            print("=== Formality: skipped (ENABLE_FORMALITY=False) ===")

    if pronoun_suite is not None:
        print("=== English pronouns ===")
        _train_suite(config, problem="pronoun", suite=pronoun_suite, models_for_heatmap=models_for_heatmap)

    if pronoun_merged_suite is not None:
        print("=== English pronoun merged groups ===")
        _train_suite(
            config,
            problem="pronoun_merged",
            suite=pronoun_merged_suite,
            models_for_heatmap=models_for_heatmap,
        )

    if race_suite is not None:
        print("=== Race ===")
        _train_suite(
            config,
            problem="train_race_religion",
            suite=race_suite,
            models_for_heatmap=models_for_heatmap,
        )
    if religion_suite is not None:
        print("=== Religion ===")
        _train_suite(
            config,
            problem="train_race_religion",
            suite=religion_suite,
            models_for_heatmap=models_for_heatmap,
        )

    if gender_en_trainer is not None:
        train_gender_en(config, models_for_heatmap, trainer=gender_en_trainer)

    if gender_de_suite is not None:
        print("=== German gender/case ===")
        _train_suite(
            config,
            problem="gender_de",
            suite=gender_de_suite,
            models_for_heatmap=models_for_heatmap,
        )

    if _problem_selected(selected_problems, "sentiment"):
        if sentiment_suite is None:
            sentiment_suite = build_sentiment_suite(
                config,
                retain_models_in_memory=retain_models_in_memory,
            )
        print("=== Sentiment (full positive<->negative + good<->bad only) ===")
        _train_suite(
            config,
            problem="sentiment",
            suite=sentiment_suite,
            models_for_heatmap=models_for_heatmap,
        )

    return models_for_heatmap, sentiment_suite


def load_models_for_heatmap_from_cache(
    config: ExperimentConfig,
    trainers_by_id: Dict[str, TextPredictionTrainer],
) -> Dict[str, Any]:
    """Resolve trained models for top-k plots without re-running training loops."""
    os.makedirs(config.args.experiment_dir, exist_ok=True)
    prepare_global_mlm_head(config, list(trainers_by_id.values()))

    trainers_by_problem: Dict[str, List[TextPredictionTrainer]] = defaultdict(list)
    for trainer in trainers_by_id.values():
        trainers_by_problem[_problem_for_trainer(trainer)].append(trainer)
    for problem, trainers in trainers_by_problem.items():
        prepare_decoder_resources(config, problem=problem, trainers=trainers)

    models_for_heatmap: Dict[str, Any] = {}
    for child_id, trainer in trainers_by_id.items():
        trainer.train(use_cache=config.args.use_cache)
        trainer.cpu()
        analysis_trainer = enter_analysis_mode(trainer)
        models_for_heatmap[child_id] = analysis_trainer.get_model(gradiend_only=True)
    return models_for_heatmap


def _pretty_label(mid: str) -> str:
    from gradiend.visualizer.multilingual_demo_labels import pretty_demo_trainer_id

    return pretty_demo_trainer_id(mid)


def _cross_task_probe_trainers(config: ExperimentConfig) -> Dict[str, TextPredictionTrainer]:
    """Untrained trainers used only to materialize full-domain test transition pools."""
    probes: Dict[str, TextPredictionTrainer] = {}
    training_path, neutral_path = _pronoun_data_paths()
    probe_args = problem_args(config, experiment_dir=None, use_cache=False)
    objective = prediction_objective_for_problem(config, "pronoun")
    if objective is not None:
        probe_args = replace(probe_args, prediction_objective=objective)
    probes["pronoun_probe_pool"] = TextPredictionTrainer(
        model=config.model_name,
        run_id="pronoun_probe_pool",
        data=str(training_path),
        all_classes=PRONOUN_CLASSES,
        target_classes=("1SG", "1PL"),
        masked_col="masked",
        split_col="split",
        eval_neutral_data=str(neutral_path),
        args=probe_args,
    )
    return probes


def _build_plot_order_and_groups(all_ids: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    from gradiend.visualizer.multilingual_demo_labels import build_demo_trainer_order_and_groups

    return build_demo_trainer_order_and_groups(all_ids)


def _build_feature_plot_groups() -> Dict[str, List[str]]:
    from gradiend.visualizer.multilingual_demo_labels import build_demo_feature_plot_groups

    return build_demo_feature_plot_groups(
        MULTILINGUAL_FEATURE_CLASSES,
        include_formality=ENABLE_FORMALITY,
    )


def _plot_std_heatmap_from_cell_stats(
    comparison_data: Dict[str, Any],
    *,
    output_path: str,
    order: Sequence[str],
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    **plot_kwargs: Any,
) -> None:
    """Plot a heatmap whose cells are per-cell seed std (requires ``cell_stats``)."""
    from gradiend.comparison.common import comparison_matrix_from_cell_stat
    from gradiend.visualizer.heatmaps import plot_comparison_heatmap
    from gradiend.visualizer.heatmaps.base import filter_comparison_heatmap_plot_kwargs

    if not comparison_data.get("cell_stats"):
        return
    std_data = comparison_matrix_from_cell_stat(comparison_data, field="std")
    plot_comparison_heatmap(
        std_data,
        order=list(order),
        output_path=output_path,
        show=True,
        pretty_groups=pretty_groups,
        **filter_comparison_heatmap_plot_kwargs(plot_kwargs),
    )
    print(f"Std heatmap saved to {output_path}")


def plot_cross_encoding(
    config: ExperimentConfig,
    trainers_by_id: Dict[str, TextPredictionTrainer],
    *,
    trainer_order: Sequence[str],
    trainer_pretty_groups: Dict[str, List[str]],
    feature_order: Sequence[str],
    feature_pretty_groups: Dict[str, List[str]],
) -> None:
    """Plot pre-anchor and anchor-aligned cross-encoding from one cross-task encoder pass."""
    from gradiend.comparison import (
        compute_anchor_aligned_encoding_std_matrix,
        compute_gradiend_transition_cross_encoding_matrix,
        pair_by_id_from_trainers,
        source_by_id_from_trainers,
    )
    from gradiend.visualizer.multilingual_demo_labels import (
        build_demo_feature_label_mapping,
        build_demo_trainer_label_mapping,
        build_demo_transition_label_mapping,
        demo_encoding_heatmap_normalized_style_kwargs,
        demo_encoding_heatmap_style_kwargs,
    )

    eval_rows = collect_unified_test_rows(
        trainers_by_id,
        split="test",
        probe_trainers=_cross_task_probe_trainers(config),
        required_factual_classes=feature_order,
    )
    encoder_summary = build_cross_task_encoder_summary(
        trainers_by_id,
        feature_order,
        eval_rows=eval_rows,
        split="test",
        max_size=config.args.encoder_eval_max_size,
    )
    transition_order = collect_unified_test_transitions(
        trainers_by_id,
        split="test",
        eval_rows=eval_rows,
    )
    style = demo_encoding_heatmap_style_kwargs()
    trainer_labels = build_demo_trainer_label_mapping(trainer_order)
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
        pretty_groups=trainer_pretty_groups,
        row_label_mapping=trainer_labels,
        column_label_mapping=transition_labels,
        output_path=gradiend_transition_path,
        title="GRADIEND × input transition (pre-anchor)",
        show=True,
        **style,
    )
    print(f"GRADIEND × transition cross-encoding heatmap saved to {gradiend_transition_path}")

    if config.args.analyze_seed_stability:
        transition_comparison = compute_gradiend_transition_cross_encoding_matrix(
            trainers_by_id,
            trainer_order=trainer_order,
            transition_order=transition_order,
            encoder_summary=encoder_summary,
            split="test",
            max_size=config.args.encoder_eval_max_size,
        )
        transition_std_path = os.path.join(
            config.args.experiment_dir,
            "cross_encoding_gradiend_by_transition_std_heatmap.pdf",
        )
        _plot_std_heatmap_from_cell_stats(
            transition_comparison,
            output_path=transition_std_path,
            order=trainer_order,
            pretty_groups=trainer_pretty_groups,
            row_label_mapping=trainer_labels,
            column_label_mapping=transition_labels,
            title="GRADIEND × input transition (pre-anchor, seed std)",
            **style,
        )

    feature_labels = build_demo_feature_label_mapping(feature_order)
    normalized_style = demo_encoding_heatmap_normalized_style_kwargs()
    for s in ['factual', 'counterfactual']:
        cross_encoding_output = os.path.join(
            config.args.experiment_dir,
            f"cross_encoding_oriented_{s}_heatmap.pdf",
        )
        plot_cross_encoding_heatmap(
            trainers_by_id,
            feature_order,
            alignment=s,
            column_ids=feature_order,
            encoder_summary=encoder_summary,
            split="test",
            max_size=config.args.encoder_eval_max_size,
            cross_task_eval=False,
            aggregate="mean",
            order=feature_order,
            pretty_groups=feature_pretty_groups,
            row_label_mapping=feature_labels,
            column_label_mapping=feature_labels,
            output_path=cross_encoding_output,
            title=False,
            show=True,
            **style,
        )
        print(f"Oriented cross-encoding heatmap saved to {cross_encoding_output}")

        if config.args.analyze_seed_stability:
            oriented_std = compute_anchor_aligned_encoding_std_matrix(
                pair_by_id=pair_by_id_from_trainers(trainers_by_id),
                encoder_summary=encoder_summary,
                feature_classes=feature_order,
                alignment=s,
                column_ids=feature_order,
                source_by_id=source_by_id_from_trainers(trainers_by_id),
            )
            from gradiend.visualizer.heatmaps import plot_comparison_heatmap
            from gradiend.visualizer.heatmaps.base import filter_comparison_heatmap_plot_kwargs

            std_output = os.path.join(
                config.args.experiment_dir,
                f"cross_encoding_oriented_{s}_std_heatmap.pdf",
            )
            std_style = dict(style)
            std_style["cbar_label"] = "Encoding std"
            plot_comparison_heatmap(
                oriented_std,
                order=feature_order,
                pretty_groups=feature_pretty_groups,
                row_label_mapping=feature_labels,
                column_label_mapping=feature_labels,
                output_path=std_output,
                title=False,
                show=True,
                models=trainers_by_id,
                **filter_comparison_heatmap_plot_kwargs(std_style),
            )
            print(f"Oriented cross-encoding std heatmap saved to {std_output}")

        normalized_output = os.path.join(
            config.args.experiment_dir,
            f"cross_encoding_oriented_{s}_row_normalized_heatmap.pdf",
        )
        plot_cross_encoding_heatmap(
            trainers_by_id,
            feature_order,
            alignment=s,
            column_ids=feature_order,
            encoder_summary=encoder_summary,
            split="test",
            max_size=config.args.encoder_eval_max_size,
            cross_task_eval=False,
            aggregate="mean",
            normalize=True,
            order=feature_order,
            pretty_groups=feature_pretty_groups,
            row_label_mapping=feature_labels,
            column_label_mapping=feature_labels,
            output_path=normalized_output,
            title=False,
            show=True,
            **normalized_style,
        )
        print(f"Row-normalized oriented cross-encoding heatmap saved to {normalized_output}")


def plot_results(
    config: ExperimentConfig,
    models_for_heatmap: Dict[str, Any],
    trainers_by_id: Dict[str, TextPredictionTrainer],
) -> None:
    from gradiend.visualizer.multilingual_demo_labels import demo_topk_overlap_style_kwargs

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
        output_path=output_path,
        show=True,
        pretty_groups=pretty_groups_display,
        **demo_topk_overlap_style_kwargs(),
    )
    print(f"Heatmap saved to {output_path}")

    if config.args.analyze_seed_stability:
        from gradiend.comparison import compute_similarity_matrix

        topk_comparison = compute_similarity_matrix(
            models_display,
            measure="topk_overlap",
            part="decoder-weight",
            topk=topk,
            value="intersection_frac",
            dispersion="std",
        )
        std_output_path = os.path.join(
            config.args.experiment_dir,
            f"topk_overlap_heatmap_all_{topk}_std.pdf",
        )
        std_style = dict(demo_topk_overlap_style_kwargs())
        std_style["cbar_label"] = "Overlap std"
        _plot_std_heatmap_from_cell_stats(
            topk_comparison,
            output_path=std_output_path,
            order=order_display,
            pretty_groups=pretty_groups_display,
            percentages=True,
            **std_style,
        )

    feature_groups = _build_feature_plot_groups()
    grouped_features = [feature for features in feature_groups.values() for feature in features]
    feature_order = grouped_features + [
        feature for feature in MULTILINGUAL_FEATURE_CLASSES if feature not in grouped_features
    ]
    trainer_order = [mid for mid in ordered if mid in trainers_by_id]
    plot_cross_encoding(
        config,
        trainers_by_id,
        trainer_order=trainer_order,
        trainer_pretty_groups=pretty_groups,
        feature_order=feature_order,
        feature_pretty_groups=feature_groups,
    )

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
    if config.args.analyze_seed_stability:
        print(
            f"Multi-seed stability: min_convergent_seeds={config.args.min_convergent_seeds}, "
            f"saved_seed_runs={config.args.saved_seed_runs}"
        )

    retain_models = cli_args.retain_models_in_memory
    selected_problems = _parse_problems(cli_args.problems)
    if selected_problems is not None:
        print(f"Problems (--problems): {', '.join(sorted(selected_problems))}")

    pronoun_suite = (
        build_pronoun_suite(config, retain_models_in_memory=retain_models)
        if _problem_selected(selected_problems, "pronoun")
        else None
    )
    pronoun_merged_suite = (
        build_pronoun_merged_suite(config, retain_models_in_memory=retain_models)
        if _problem_selected(selected_problems, "pronoun_merged")
        else None
    )
    race_suite = religion_suite = None
    if _problem_selected(selected_problems, "race") or _problem_selected(selected_problems, "religion"):
        built_race, built_religion = build_train_race_religion_suite(
            config,
            retain_models_in_memory=retain_models,
        )
        race_suite = built_race if _problem_selected(selected_problems, "race") else None
        religion_suite = built_religion if _problem_selected(selected_problems, "religion") else None
    gender_de_suite = (
        build_gender_de_suite(config, retain_models_in_memory=retain_models)
        if _problem_selected(selected_problems, "gender_de")
        else None
    )
    gender_en_trainer = (
        build_gender_en_trainer(config)
        if _problem_selected(selected_problems, "gender_en")
        else None
    )
    formality_suite = None
    if _problem_selected(selected_problems, "formality") and ENABLE_FORMALITY:
        formality_suite = build_formality_suite(config, retain_models_in_memory=retain_models)

    _assert_selected_problems_runnable(
        selected_problems,
        pronoun_suite=pronoun_suite,
        pronoun_merged_suite=pronoun_merged_suite,
        race_suite=race_suite,
        religion_suite=religion_suite,
        gender_de_suite=gender_de_suite,
        gender_en_trainer=gender_en_trainer,
        formality_suite=formality_suite,
    )

    trainer_suites = [
        suite
        for suite in (
            pronoun_suite,
            pronoun_merged_suite,
            race_suite,
            religion_suite,
            gender_de_suite,
            formality_suite,
        )
        if suite is not None
    ]

    if cli_args.plot_only:
        sentiment_suite = None
        if _problem_selected(selected_problems, "sentiment"):
            sentiment_suite = build_sentiment_suite(config, retain_models_in_memory=retain_models)
            trainer_suites.append(sentiment_suite)
        merge_parts: List[Any] = list(trainer_suites)
        if gender_en_trainer is not None:
            merge_parts.append(gender_en_trainer)
        trainers_by_id = (
            TrainerCollection.merge(*merge_parts).trainers if merge_parts else {}
        )
        print("--plot-only: skipping training (using cached checkpoints where available)")
        models_for_heatmap = load_models_for_heatmap_from_cache(config, trainers_by_id)
    else:
        models_for_heatmap, sentiment_suite = train_all(
            config,
            selected_problems=selected_problems,
            pronoun_suite=pronoun_suite,
            pronoun_merged_suite=pronoun_merged_suite,
            race_suite=race_suite,
            religion_suite=religion_suite,
            gender_de_suite=gender_de_suite,
            gender_en_trainer=gender_en_trainer,
            formality_suite=formality_suite,
            retain_models_in_memory=retain_models,
        )
        if sentiment_suite is not None:
            trainer_suites.append(sentiment_suite)

    merge_parts = list(trainer_suites)
    if gender_en_trainer is not None:
        merge_parts.append(gender_en_trainer)
    trainers_by_id = TrainerCollection.merge(*merge_parts).trainers if merge_parts else {}
    trainers_by_id = enter_analysis_mode_for_trainers(trainers_by_id)

    plot_results(config, models_for_heatmap, trainers_by_id)


if __name__ == "__main__":
    main()
