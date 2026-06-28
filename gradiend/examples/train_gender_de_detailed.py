"""
German gender/case – detailed workflow using more API features.

Demonstrates:

- Pre-pruning (1% top weights) and post-pruning (5% top weights) via TrainingArguments
- Encoder plot with renamed class labels (legend_name_mapping)
- Pairwise top-k overlap heatmap and training convergence (live + static)

See also (separate examples): supervised training, interactive scatter/beeswarm plot.
"""

import os

from gradiend import (
    TextPredictionTrainer,
    TrainingArguments,
    compute_grouped_similarity_matrices,
    plot_comparison_heatmap,
)
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend import plot_topk_overlap_heatmap, plot_topk_overlap_venn

CONFIGS = {
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

ENCODER_LABEL_NAMES = {
    "masc_nom": "Masc. Nom.",
    "fem_nom": "Fem. Nom.",
    "neut_nom": "Neut. Nom.",
    "masc_acc": "Masc. Acc.",
    "fem_acc": "Fem. Acc.",
    "neut_acc": "Neut. Acc.",
    "masc_dat": "Masc. Dat.",
    "fem_dat": "Fem. Dat.",
    "neut_dat": "Neut. Dat.",
    "masc_gen": "Masc. Gen.",
    "fem_gen": "Fem. Gen.",
    "neut_gen": "Neut. Gen.",
}


def _mean_off_diagonal(matrix):
    values = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i != j:
                values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def _layer_sort_key(layer_name):
    if layer_name.startswith("layer_"):
        try:
            return int(layer_name.split("_", 1)[1])
        except ValueError:
            pass
    return layer_name


def _plot_layerwise_similarity(models_for_heatmap, output_path):
    grouped = compute_grouped_similarity_matrices(
        models_for_heatmap,
        measure="cosine",
        part="decoder-weight",
        topk=1000,
        group_by="layer",
    )
    layer_scores = []
    for group_name, matrix_data in sorted(grouped.items()):
        if not group_name.startswith("layer_"):
            continue
        layer_scores.append((group_name, _mean_off_diagonal(matrix_data["matrix"])))
    layer_scores = sorted(layer_scores, key=lambda item: _layer_sort_key(item[0]))
    if not layer_scores:
        print("No layer-specific parameter groups were found for the layer-wise similarity plot.")
        return None

    comparison_data = {
        "measure": "layerwise_similarity",
        "part": "decoder-weight",
        "model_ids": ["mean"],
        "column_ids": [name.split("_", 1)[1] for name, _ in layer_scores],
        "matrix": [[value for _, value in layer_scores]],
        "row_labels": {"mean": "mean pairwise cosine"},
    }
    return plot_comparison_heatmap(
        comparison_data,
        output_path=output_path,
        title="Layer-wise decoder similarity",
        show=False,
        vmin=0.0,
        vmax=1.0,
    )


if __name__ == "__main__":
    model_name = "bert-base-german-cased"
    args = TrainingArguments(
        experiment_dir=f"runs/german_de_detailed_{model_name.split('/')[-1]}",
        trust_remote_code=True,  # required for EuroBERT
        train_batch_size=8,
        train_max_size=500,
        encoder_eval_max_size=50,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=500,
        eval_steps=100,
        num_train_epochs=1,
        max_seeds=3,
        min_convergent_seeds=1,
        max_steps=100,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=5e-5,
        pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01, source="diff"),
        post_prune_config=PostPruneConfig(topk=0.05, part="decoder-weight"),
        use_cache=True,
        add_identity_for_other_classes=True,
        fail_on_non_convergence=True,
    )

    models_for_heatmap = {}
    models_by_transition = {}
    for transition, pairs in CONFIGS.items():
        for pair in pairs:
            trainer = TextPredictionTrainer(
                model=model_name,
                run_id=f"gender_de_{pair[0]}_{pair[1]}",
                data="aieng-lab/de-gender-case-articles",
                target_classes=list(pair),
                masked_col="masked",
                split_col="split",
                eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
                args=args,
            )

            print(f"=== German: {pair[0]} vs {pair[1]} (detailed workflow) ===")

            trainer.train()
            print(f"  Model available at {trainer.model_path}")

            stats = trainer.get_training_stats()
            ts = stats.get("training_stats", {}) if stats else {}
            print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")

            trainer.plot_training_convergence(label_name_mapping=ENCODER_LABEL_NAMES)

            max_size = 100
            base_legend_group_mapping = {
                'die': ['fem_acc', 'fem_nom'],
                'der': ['fem_dat', 'fem_gen', 'masc_nom'],
                'den': ['masc_acc'],
                'dem': ['masc_dat', 'neut_dat'],
                'des': ['masc_gen', 'neut_gen'],
                'das': ['neut_acc', 'neut_nom'],
            }

            legend_group_mapping = {}
            for group_label, classes in base_legend_group_mapping.items():
                kept = []
                for cls in classes:
                    if cls in pair:
                        continue
                    display = ENCODER_LABEL_NAMES.get(cls, cls)
                    kept.append(f"{display} -> {display}")
                if kept:
                    legend_group_mapping[group_label] = kept

            plot_kwargs = {
                'class_label_mapping': ENCODER_LABEL_NAMES,
                'legend_group_mapping': legend_group_mapping,
                'target_and_neutral_only': False,
            }
            enc_eval = trainer.evaluate_encoder(max_size=max_size, return_df=True, plot=True, plot_kwargs=plot_kwargs)
            print(f"  encoder metrics: {enc_eval}")

            dec_results = trainer.evaluate_decoder()
            print(f"  decoder summary: {[k for k in dec_results if k not in ('grid', 'plot_path', 'plot_paths')]}")

            #trainer.rewrite_base_model(decoder_results=dec_stats, target_class=pair[0], output_dir="./output")
            trainer.cpu()  # move model to CPU to free GPU for next config
            models_for_heatmap[trainer.run_id] = trainer.get_model()
            models_by_transition.setdefault(transition, {})[" - ".join(pair)] = trainer.get_model()

    plot_topk_overlap_heatmap(
        models_for_heatmap,
        topk=1000,
        part="decoder-weight",
        value="intersection_frac",
        output_path=os.path.join(args.experiment_dir, "topk_overlap_heatmap.png"),
    )
    _plot_layerwise_similarity(
        models_for_heatmap,
        output_path=os.path.join(args.experiment_dir, "layerwise_decoder_similarity.png"),
    )
    for transition, models in sorted(models_by_transition.items()):
        if not (2 <= len(models) <= 6):
            continue
        topk_result = plot_topk_overlap_venn(
            models,
            topk=1000,
            part="decoder-weight",
            output_path=os.path.join(args.experiment_dir, f"venn_{transition.replace(' ', '').replace('<->', '_')}.png"),
        )
        print(f"[{transition}] Top-k intersection size =", len(topk_result["intersection"]))
        print(f"[{transition}] Top-k union size =", len(topk_result["union"]))
