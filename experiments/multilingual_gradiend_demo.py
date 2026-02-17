"""
Multilingual GRADIEND comparison for system demo: one large heatmap.

Derived from gender_de_detailed.py. Trains GRADIENDs for:
- German gender/case (gender_de_detailed configs: multiple pairs)
- English gender (gender_en: one GRADIEND M vs F)
- Race (3 GRADIENDs: white-black, white-asian, black-asian)
- Religion (3 GRADIENDs: christian-muslim, christian-jewish, muslim-jewish)
- English pronouns (10 GRADIENDs: all pairs from 1SG, 1PL, 2, 3SG, 3PL)

Uses a single multilingual model (EuroBERT-210m) and the same TrainingArguments
as gender_de_detailed. At the end, plots one large top-k overlap heatmap over
all trained models.
"""

import os
from collections import defaultdict
from pathlib import Path

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.visualizer.topk.pairwise_heatmap import plot_topk_overlap_heatmap


model_name = "google-bert/bert-base-multilingual-cased"
args = TrainingArguments(
    experiment_dir=f"runs/multilingual_gradiend_demo_{model_name.split('/')[-1]}",
    train_batch_size=16,
    train_max_size=20000,
    eval_steps=250,
    max_steps=1000,
    encoder_eval_max_size=50,
    decoder_eval_max_size_training_like=100,
    decoder_eval_max_size_neutral=500,
    num_train_epochs=1,
    max_seeds=5,
    min_convergent_seeds=1,
    source="alternative",
    target="diff",
    eval_batch_size=8,
    learning_rate=1e-5,
    pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01, source="diff"),
    post_prune_config=PostPruneConfig(topk=0.05, part="decoder-weight"),
    add_identity_for_other_classes=True,
    use_cache=True,
)

# German gender/case configs (from gender_de_detailed)
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

# Race: 3 binary GRADIENDs (white-black, white-asian, black-asian)
race_configs = [
    ("race", ("white", "black"), ["asian"]),
    ("race", ("white", "asian"), ["black"]),
    ("race", ("black", "asian"), ["white"]),
]

# Religion: 3 binary GRADIENDs (christian-muslim, christian-jewish, muslim-jewish)
religion_configs = [
    ("religion", ("christian", "muslim"), ["jewish"]),
    ("religion", ("christian", "jewish"), ["muslim"]),
    ("religion", ("muslim", "jewish"), ["christian"]),
]

# Pronoun pairs (from pronoun_workflow / data_creation_pronouns: 1SG, 1PL, 2, 3SG, 3PL)
# Data assumed at data/english_pronouns/ (training.csv, neutral.csv)
PRONOUN_DATA_DIR = "data/english_pronouns"
PRONOUN_CLASSES = ["1SG", "1PL", "2", "3SG", "3PL"]
pronoun_pairs = [
    (PRONOUN_CLASSES[i], PRONOUN_CLASSES[j])
    for i in range(len(PRONOUN_CLASSES))
    for j in range(i + 1, len(PRONOUN_CLASSES))
]

models_for_heatmap = {}


def train_gender_de():
    """Train all German gender/case GRADIENDs."""
    for transition, pairs in configs_de.items():
        for pair in pairs:
            trainer = TextPredictionTrainer(
                model=model_name,
                run_id=f"gender_de_{pair[0]}_{pair[1]}",
                data="aieng-lab/de-gender-case-articles",
                target_classes=pair,
                masked_col="masked",
                split_col="split",
                eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
                args=args,
            )
            print(f"=== German: {pair[0]} vs {pair[1]} ===")
            trainer.train()
            stats = trainer.get_training_stats()
            ts = stats.get("training_stats", {}) if stats else {}
            print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
            trainer.cpu()
            models_for_heatmap[trainer.run_id] = trainer.get_model()


def train_gender_en():
    """Train English gender GRADIEND (M vs F)."""
    from gradiend.examples.gender_en import build_gender_trainer

    trainer = build_gender_trainer(model=model_name, names_per_template=10, args=args)
    print("=== Gender EN: M vs F ===")
    trainer.train()
    stats = trainer.get_training_stats()
    ts = stats.get("training_stats", {}) if stats else {}
    print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
    trainer.cpu()
    models_for_heatmap[trainer.run_id] = trainer.get_model()


def train_race_religion():
    """Train 3 race + 3 religion GRADIENDs."""
    for bias_type, pair, other_classes in race_configs + religion_configs:
        trainer = TextPredictionTrainer(
            model=model_name,
            run_id=f"{bias_type}_{pair[0]}_{pair[1]}",
            data=f"aieng-lab/gradiend_{bias_type}_data",
            target_classes=pair,
            masked_col="masked",
            eval_neutral_data="aieng-lab/biasneutral",
            args=args,
        )
        print(f"=== {bias_type}: {pair[0]} vs {pair[1]} ===")
        trainer.train()
        stats = trainer.get_training_stats()
        ts = stats.get("training_stats", {}) if stats else {}
        print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
        trainer.cpu()
        models_for_heatmap[trainer.run_id] = trainer.get_model()


def train_pronoun():
    """Train all pronoun pair GRADIENDs (data from data_creation_pronouns)."""
    base = Path(PRONOUN_DATA_DIR)
    training_path = base / "training.csv"
    neutral_path = base / "neutral.csv"
    if not training_path.is_file() or not neutral_path.is_file():
        raise FileNotFoundError(
            f"Pronoun data not found at {PRONOUN_DATA_DIR}. "
            "Run gradiend.examples.data_creation_pronouns to generate training.csv and neutral.csv."
        )
    for c1, c2 in pronoun_pairs:
        trainer = TextPredictionTrainer(
            model=model_name,
            run_id=f"pronoun_{c1}_{c2}",
            data=str(training_path),
            target_classes=[c1, c2],
            masked_col="masked",
            split_col="split",
            eval_neutral_data=str(neutral_path),
            args=args,
        )
        print(f"=== Pronoun: {c1} vs {c2} ===")
        trainer.train()
        stats = trainer.get_training_stats()
        ts = stats.get("training_stats", {}) if stats else {}
        print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
        trainer.cpu()
        models_for_heatmap[trainer.run_id] = trainer.get_model()


def main():
    os.makedirs(args.experiment_dir, exist_ok=True)

    train_pronoun()
    train_race_religion()
    train_gender_en()
    train_gender_de()

    all_ids = list(models_for_heatmap.keys())
    gender_de_ids = sorted([m for m in all_ids if m.startswith("gender_de_")])
    pronoun_ids = sorted([m for m in all_ids if m.startswith("pronoun_")])
    race_ids = sorted([m for m in all_ids if m.startswith("race_")])
    religion_ids = sorted([m for m in all_ids if m.startswith("religion_")])
    gender_en_ids = [m for m in all_ids if m == "gender_en"]
    ordered = gender_de_ids + gender_en_ids + pronoun_ids + race_ids + religion_ids

    # Pretty labels: feature_class_id -> display name
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
    gender_de_ids_to_articles = {
        id: r"$\longleftrightarrow$".join(
            sorted(
                ARTICLE_MAPPING["_".join(pair)]
                for pair in zip(*[iter(id.removeprefix("gender_de_").split("_"))] * 2)
            )
        )
        for id in gender_de_ids
    }

    gender_de_transitions_to_ids = defaultdict(list)
    for id, transition in gender_de_ids_to_articles.items():
        gender_de_transitions_to_ids[transition].append(id)

    def _pretty_label(mid: str) -> str:
        if mid == "gender_en":
            return "Gender (M$\longleftrightarrow$F)"
        if mid.startswith("gender_de_"):
            rest = mid.replace("gender_de_", "")  # e.g. fem_acc_neut_acc
            parts = rest.split("_")
            if len(parts) >= 4:  # two case names (gender_case)
                a, b = "_".join(parts[:2]), "_".join(parts[2:])
                return f"{CASE_PRETTY.get(a, a)}$\longleftrightarrow${CASE_PRETTY.get(b, b)}"
        if mid.startswith("pronoun_"):
            c1, c2 = mid.replace("pronoun_", "").split("_")
            return f"{c1}$\longleftrightarrow${c2}"
        if mid.startswith("race_"):
            w1, w2 = mid.replace("race_", "").split("_")
            return f"{w1.capitalize()}$\longleftrightarrow${w2.capitalize()}"
        if mid.startswith("religion_"):
            w1, w2 = mid.replace("religion_", "").split("_")
            return f"{w1.capitalize()}$\longleftrightarrow${w2.capitalize()}"
        return mid

    # Pretty groups: partition of all model ids (disjoint, cover all)
    remaining = [m for m in all_ids if m not in ordered]
    pretty_groups = {
        **gender_de_transitions_to_ids,
        "English Gender": gender_en_ids,
        "English Pronouns": pronoun_ids,
        "Race": race_ids,
        "Religion": religion_ids,
    }
    if remaining:
        pretty_groups["Other"] = remaining

    # order for pretty_labels: match heatmap's order (from pretty_groups)
    order = [mid for gids in pretty_groups.values() for mid in gids]
    pretty_labels = {mid: _pretty_label(mid) for mid in order}

    topk = 1000
    plot_topk_overlap_heatmap(
        models_for_heatmap,
        topk=topk,
        part="decoder-weight",
        value="intersection_frac",
        annot="auto",
        output_path=os.path.join(args.experiment_dir, f"topk_overlap_heatmap_all_{topk}.pdf"),
        show=True,
        pretty_labels=pretty_labels,
        pretty_groups=pretty_groups,
        show_values=True,
        scale="linear",
        group_label_fontsize=16,
        scale_gamma=0.5,
        tick_label_fontsize=8,
        annot_fontsize=6,
        cbar_pad=0.1,
    )
    print(f"Heatmap saved to {os.path.join(args.experiment_dir, 'topk_overlap_heatmap_all.pdf')}")


if __name__ == "__main__":
    main()
