import importlib.util
import sys
from pathlib import Path


def _load_experiment_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gender_de_run = _load_experiment_module("gender_de_run_under_test", "experiments/gender_de/run.py")
_pruning_ablation = _load_experiment_module(
    "gender_de_pruning_parameter_ablation_under_test",
    "experiments/gender_de_pruning_parameter_ablation.py",
)

build_arg_parser = _gender_de_run.build_arg_parser
build_training_arguments = _gender_de_run.build_training_arguments
infer_needs_decoder_only_mlm_head = _gender_de_run.infer_needs_decoder_only_mlm_head
make_trainer = _gender_de_run.make_trainer
transition_from_spec = _gender_de_run.transition_from_spec
CLASS_IDS = _gender_de_run.CLASS_IDS
PairSpec = _gender_de_run.PairSpec

AblationResult = _pruning_ablation.AblationResult
POST_PARTS = _pruning_ablation.POST_PARTS
POST_TOPK_VALUES = _pruning_ablation.POST_TOPK_VALUES
PRE_N_SAMPLES = _pruning_ablation.PRE_N_SAMPLES
PRE_SOURCES = _pruning_ablation.PRE_SOURCES
PRE_TOPK_VALUES = _pruning_ablation.PRE_TOPK_VALUES
build_pruning_ablation_arg_parser = _pruning_ablation.build_arg_parser
plot_pre_metric = _pruning_ablation.plot_pre_metric


def test_decoder_only_mlm_head_inference_for_common_model_ids():
    assert infer_needs_decoder_only_mlm_head("bert-base-german-cased")[0] is False
    assert infer_needs_decoder_only_mlm_head("FacebookAI/roberta-base")[0] is False
    assert infer_needs_decoder_only_mlm_head("dbmdz/german-gpt2")[0] is True
    assert infer_needs_decoder_only_mlm_head("meta-llama/Llama-3.2-3B")[0] is True
    assert infer_needs_decoder_only_mlm_head("Qwen/Qwen2.5-1.5B")[0] is True


def test_decoder_only_mlm_head_cli_defaults_to_auto_with_alias():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.decoder_only_mlm_head == "auto"
    assert args.train_decoder_only_mlm_head is None
    assert args.skip_article_heatmap is False
    assert args.large_model is False

    args = parser.parse_args(["--large-model"])
    assert args.large_model is True

    args = parser.parse_args(["--base-model-device-map", "false"])
    assert args.base_model_device_map is False

    args = parser.parse_args(["--train-decoder-only-mlm-head"])
    assert args.train_decoder_only_mlm_head is True


def test_article_heatmap_direction_selects_target_class():
    spec = PairSpec("masc_nom", "fem_nom", "custom")
    transition = transition_from_spec(spec, source_article="der", target_article="die")
    assert transition.source_class == "masc_nom"
    assert transition.target_class == "fem_nom"

    reverse = transition_from_spec(spec, source_article="die", target_article="der")
    assert reverse.source_class == "fem_nom"
    assert reverse.target_class == "masc_nom"


def test_gender_de_uses_conservative_decoder_lr_grid_by_default():
    args = build_training_arguments(experiment_dir="runs/_tmp")
    assert args.decoder_eval_lrs == [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

    parser = build_arg_parser()
    parsed = parser.parse_args(["--decoder-eval-lrs", "0.3,0.1,0.03"])
    assert parsed.decoder_eval_lrs == [0.3, 0.1, 0.03]


def test_gender_de_trainer_uses_explicit_all_classes_to_avoid_hf_config_autodiscovery():
    args = build_training_arguments(experiment_dir="runs/_tmp")
    trainer = make_trainer(
        model="bert-base-german-cased",
        spec=PairSpec("masc_nom", "fem_nom", "custom"),
        args=args,
    )
    assert trainer.config.all_classes == list(CLASS_IDS)


def test_gender_de_pruning_parameter_ablation_defaults():
    assert PRE_N_SAMPLES == [1, 2, 4, 8, 16, 32, 64]
    assert PRE_SOURCES == ["alternative", "factual", "diff"]
    assert PRE_TOPK_VALUES == [1.0, 0.1, 0.01, 0.001]
    assert POST_PARTS == ["decoder-weight", "decoder-bias", "decoder-sum", "encoder-weight"]
    assert POST_TOPK_VALUES == [1.0, 0.1, 0.01, 0.001, 0.0001]

    parser = build_pruning_ablation_arg_parser()
    args = parser.parse_args(
        [
            "--mode",
            "post-part",
            "--post-topk-values",
            "1,0.1,0.01",
            "--post-parts",
            "decoder-weight,decoder-bias",
        ]
    )
    assert args.mode == "post-part"
    assert args.post_topk_values == [1.0, 0.1, 0.01]
    assert args.post_parts == ["decoder-weight", "decoder-bias"]

    args = parser.parse_args(["--mode", "plot"])
    assert args.mode == "plot"


def test_gender_de_pruning_parameter_ablation_pre_plot(tmp_path):
    results = [
        AblationResult(
            ablation="pre",
            run_id="pre_src_alternative_n_1_topk_0_010000",
            pre_topk=0.01,
            pre_source="alternative",
            pre_n_samples=1,
            decoder_prob_delta=0.1,
        ),
        AblationResult(
            ablation="pre",
            run_id="pre_src_alternative_n_2_topk_0_010000",
            pre_topk=0.01,
            pre_source="alternative",
            pre_n_samples=2,
            decoder_prob_delta=0.2,
        ),
    ]
    output_path = tmp_path / "pre_decoder_prob_delta.pdf"
    plot_pre_metric(results, metric="decoder_prob_delta", output_path=str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0
