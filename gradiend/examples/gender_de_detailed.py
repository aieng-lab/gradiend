"""
German gender/case – detailed workflow using more API features.

Demonstrates:
- Pre-pruning (1% top weights) and post-pruning (5% top weights) via TrainingArguments
- Encoder plot with renamed class labels (legend_name_mapping)
- Pairwise top-k overlap heatmap and training convergence (live + static)

See also (separate examples): supervised training, interactive scatter/beeswarm plot.
"""

import os

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.visualizer.topk.pairwise_heatmap import plot_topk_overlap_heatmap
from gradiend.visualizer.topk.venn_ import plot_topk_overlap_venn

model_name = "EuroBERT/EuroBERT-210m"
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
)

configs = {
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

models_for_heatmap = {}
models_by_transition = {}
for transition, pairs in configs.items():
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
            'die': ['fem_acc -> fem_acc', 'fem_nom -> fem_nom'],
            'der': ['fem_dat -> fem_dat', 'fem_gen -> fem_gen', 'masc_nom -> masc_nom'],
            'den': ['masc_acc -> masc_acc'],
            'dem': ['masc_dat -> masc_dat', 'neut_dat -> neut_dat'],
            'des': ['masc_gen -> masc_gen', 'neut_gen -> neut_gen'],
            'das': ['neut_acc -> neut_acc', 'neut_nom -> neut_nom'],
        }

        legend_group_mapping = {}
        for group_label, labels in base_legend_group_mapping.items():
            kept = []
            for label in labels:
                cls = label.split(" -> ", 1)[0].strip() if isinstance(label, str) else None
                if cls in pair:
                    continue
                kept.append(label)
            if kept:
                legend_group_mapping[group_label] = kept

        plot_kwargs = {
            'legend_group_mapping': legend_group_mapping
        }
        enc_eval = trainer.evaluate_encoder(max_size=max_size, return_df=True, plot=True, plot_kwargs=plot_kwargs)
        print(f"  encoder metrics: {enc_eval}")

        dec_results = trainer.evaluate_decoder()
        print(f"  decoder summary: {list(dec_results['summary'])}")

        #trainer.select_and_save_changed_model(decoder_results=dec_stats, metric_key=pair[0])
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
