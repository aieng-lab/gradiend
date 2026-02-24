"""
Race/religion benchmark for Llama models (1B to 70B).

Based on the race_religion example. Uses:
- torch.bfloat16 for memory efficiency (Ampere+ GPUs)
- Minimal batch size (4) for large models
- Scalable options: train_max_size, pruning, reduced eval sizes
- trust_remote_code for Llama

Requires: huggingface_hub login; meta-llama/* needs license. Qwen and codellama are open.
Run: python -m experiments.race_llama32_benchmark
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig

# Models 1B → 70B: Llama + Qwen + CodeLlama (mix of families for intermediate steps)
# meta-llama/* requires HF auth + license; Qwen/codellama are open
BENCHMARK_MODELS: List[str] = [
    #"meta-llama/Llama-3.2-1B",       # 1B
    #"meta-llama/Llama-3.2-3B",       # 3B
    #"Qwen/Qwen2.5-7B",               # 7B
    "meta-llama/Llama-3.1-8B",       # 8B
    #"Qwen/Qwen2.5-14B",              # 14B
    #"codellama/CodeLlama-13b-hf",    # 13B
    #"Qwen/Qwen2.5-32B",              # 32B
    "codellama/CodeLlama-34b-hf",    # 34B
    "meta-llama/Llama-3.1-70B",      # 70B
]

# Race/religion configs (bias_type, (pair), other_classes)
RACE_CONFIG = ("race", ("white", "black"), ["asian"])
RELIGION_CONFIG = ("religion", ("christian", "muslim"), ["jewish"])
BIAS_CONFIGS: List[Tuple[str, Tuple[str, ...], List[str]]] = [RACE_CONFIG, RELIGION_CONFIG]


def _make_args(
    experiment_dir: str,
    use_cache: bool = True,
) -> TrainingArguments:
    """Scalable training config: bfloat16, small batch, capped data, pruning."""
    return TrainingArguments(
        experiment_dir=experiment_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        train_batch_size=1,
        train_max_size=500,  # Cap training data for scalability
        encoder_eval_max_size=100,
        encoder_eval_train_max_size=10,  # Fast in-training eval
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=200,
        eval_steps=25,
        max_steps=500,
        eval_batch_size=4,
        learning_rate=1e-6,
        source="alternative",
        target="diff",
        use_cache=use_cache,
        add_identity_for_other_classes=False,
        model_use_cache=False,
        pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01, source="diff", batch_size=1),
        post_prune_config=PostPruneConfig(topk=0.1, part="decoder-weight"),
    )


def run_benchmark(
    models: List[str] | None = None,
    bias_configs: List[Tuple[str, Tuple[str, ...], List[str]]] | None = None,
    output_dir: str = "runs/race_llama32_benchmark",
    use_cache: bool = True,
) -> None:
    models = models or BENCHMARK_MODELS
    bias_configs = bias_configs or BIAS_CONFIGS
    os.makedirs(output_dir, exist_ok=True)

    for model_id in models:
        model_slug = model_id.replace("/", "_")
        experiment_dir = os.path.join(output_dir, model_slug)
        for bias_type, pair, _other in bias_configs:
            run_id = f"{bias_type}_{pair[0]}_{pair[1]}"
            args = _make_args(experiment_dir, use_cache=use_cache)

            trainer = TextPredictionTrainer(
                model=model_id,
                run_id=run_id,
                data=f"aieng-lab/gradiend_{bias_type}_data",
                target_classes=list(pair),
                masked_col="masked",
                eval_neutral_data="aieng-lab/biasneutral",
                args=args,
            )

            print(f"\n=== {model_id} | {bias_type}: {pair[0]} vs {pair[1]} ===")
            try:
                trainer.train()
                stats = trainer.get_training_stats()
                if stats:
                    ts = stats.get("training_stats", {})
                    corr = ts.get("correlation")
                    print(f"  correlation={corr}, mean_by_class={ts.get('mean_by_class')}")
                else:
                    print("  (no training stats)")
            except NotImplementedError as e:
                print(f"  FAILED: {e}")


if __name__ == "__main__":
    run_benchmark(use_cache=True)
