"""
Multilingual GRADIEND comparison for system demo: one large heatmap.

Derived from gender_de_detailed.py. Trains GRADIENDs for:
- German gender/case (gender_de_detailed configs: multiple pairs)
- English gender (gender_en: one GRADIEND M vs F)
- Race (3 GRADIENDs: white-black, white-asian, black-asian)
- Religion (3 GRADIENDs: christian-muslim, christian-jewish, muslim-jewish)
- English pronouns (10 GRADIENDs: all pairs from 1SG, 1PL, 2, 3SG, 3PL)
- English pronoun merged groups (4 GRADIENDs: Number SG vs PL; Person 1vs2, 1vs3, 2vs3)

Uses a single multilingual model (EuroBERT-210m) and the same TrainingArguments
as gender_de_detailed. At the end, plots one large top-k overlap heatmap over
all trained models.
"""

import os
from collections import defaultdict
from pathlib import Path

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend import plot_topk_overlap_heatmap, plot_topk_overlap_venn


model_name = "google-bert/bert-base-multilingual-cased"
args = TrainingArguments(
    experiment_dir=f"runs/multilingual_gradiend_demo_{model_name.split('/')[-1]}_v2",
    train_batch_size=16,
    train_max_size=20000,
    eval_steps=250,
    max_steps=1000,
    encoder_eval_max_size=50,
    decoder_eval_max_size_training_like=100,
    decoder_eval_max_size_neutral=500,
    num_train_epochs=3,
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

# Pronoun pairs (from english_pronoun_singular_plural / data_creation_pronouns: 1SG, 1PL, 2, 3SG, 3PL)
# Data assumed at data/english_pronouns/ (training.csv, neutral.csv)
PRONOUN_DATA_DIR = "data/english_pronouns"
PRONOUN_CLASSES = ["1SG", "1PL", "2", "3SG", "3PL"]
pronoun_pairs = [
    (PRONOUN_CLASSES[i], PRONOUN_CLASSES[j])
    for i in range(len(PRONOUN_CLASSES))
    for j in range(i + 1, len(PRONOUN_CLASSES))
]

# Merged pronoun groups (class_merge_map): Number (SG vs PL), Person (1vs2, 1vs3, 2vs3)
pronoun_merged_configs = [
    # Number: singular vs plural
    ("pronoun_number_singular_plural", {"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]}, "English Number SG↔PL", None),
    # Person: 1vs2, 1vs3, 2vs3
    ("pronoun_person_1vs2", {"1st": ["1SG", "1PL"], "2nd": ["2"]}, "English Person 1vs2", None),
    ("pronoun_person_1vs3", {"1st": ["1SG", "1PL"], "3rd": ["3SG", "3PL"]}, "English Person 1vs3", [["1SG", "3SG"], ["1PL", "3PL"]]),
    ("pronoun_person_2vs3", {"2nd": ["2"], "3rd": ["3SG", "3PL"]}, "English Person 2vs3", None),
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
        args.add_identity_for_other_classes = False
        trainer = TextPredictionTrainer(
            model=model_name,
            run_id=f"pronoun_{c1}_{c2}",
            data=str(training_path),
            target_classes=[c1, c2],
            all_classes=[c1, c2],
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


def train_pronoun_merged():
    """Train merged pronoun GRADIENDs (English Number: SG vs PL; English Person: 1st vs 2nd, 2nd vs 3rd, 1st vs 3rd)."""
    base = Path(PRONOUN_DATA_DIR)
    training_path = base / "training.csv"
    neutral_path = base / "neutral.csv"
    if not training_path.is_file() or not neutral_path.is_file():
        raise FileNotFoundError(
            f"Pronoun data not found at {PRONOUN_DATA_DIR}. "
            "Run gradiend.examples.data_creation_pronouns to generate training.csv and neutral.csv."
        )
    for run_id_prefix, class_merge_map, label, transition_group in pronoun_merged_configs:
        merged_keys = list(class_merge_map.keys())
        args.learning_rate = 1e-6
        trainer = TextPredictionTrainer(
            model=model_name,
            run_id=run_id_prefix,
            data=str(training_path),
            class_merge_map=class_merge_map,
            class_merge_transition_groups=transition_group,
            masked_col="masked",
            split_col="split",
            eval_neutral_data=str(neutral_path),
            args=args,
        )
        print(f"=== {label}: {merged_keys[0]} vs {merged_keys[1]} ===")
        trainer.train()
        stats = trainer.get_training_stats()
        ts = stats.get("training_stats", {}) if stats else {}
        print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")
        trainer.cpu()
        models_for_heatmap[trainer.run_id] = trainer.get_model()


def main():
    os.makedirs(args.experiment_dir, exist_ok=True)

    train_pronoun()
    train_pronoun_merged()
    train_race_religion()
    train_gender_en()
    train_gender_de()

    all_ids = list(models_for_heatmap.keys())
    gender_de_ids = sorted([m for m in all_ids if m.startswith("gender_de_")])
    pronoun_ids = sorted([m for m in all_ids if m.startswith("pronoun_") and not m.startswith("pronoun_number_") and not m.startswith("pronoun_person_")])
    pronoun_number_ids = sorted([m for m in all_ids if m.startswith("pronoun_number_")])
    pronoun_person_ids = sorted([m for m in all_ids if m.startswith("pronoun_person_")])
    race_ids = sorted([m for m in all_ids if m.startswith("race_")])
    religion_ids = sorted([m for m in all_ids if m.startswith("religion_")])
    gender_en_ids = [m for m in all_ids if m == "gender_en"]
    ordered = gender_de_ids + gender_en_ids + pronoun_ids + pronoun_number_ids + pronoun_person_ids + race_ids + religion_ids

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
    ARR = r"$\longleftrightarrow$"
    gender_de_ids_to_articles = {
        id: ARR.join(
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
            return f"he{ARR}she"
        if mid.startswith("gender_de_"):
            rest = mid.replace("gender_de_", "")  # e.g. fem_acc_neut_acc
            parts = rest.split("_")
            if len(parts) >= 4:  # two case names (gender_case)
                a, b = "_".join(parts[:2]), "_".join(parts[2:])
                return f"{CASE_PRETTY.get(a, a)}{ARR}{CASE_PRETTY.get(b, b)}"
        if mid.startswith("pronoun_number_"):
            return f"SG{ARR}PL"
        if mid.startswith("pronoun_person_"):
            rest = mid.replace("pronoun_person_", "")
            person_pretty = {"1vs2": f"1st{ARR}2nd", "1vs3": f"1st{ARR}3rd", "2vs3": f"2nd{ARR}3rd"}
            return person_pretty.get(rest, rest.replace("vs", ARR))
        if mid.startswith("pronoun_"):
            # Use class IDs like 1SG, 1PL, 2, 3SG, 3PL for heatmap consistency
            c1, c2 = mid.replace("pronoun_", "").split("_")
            return f"{c1}{ARR}{c2}"
        if mid.startswith("race_"):
            w1, w2 = mid.replace("race_", "").split("_")
            return f"{w1.capitalize()}{ARR}{w2.capitalize()}"
        if mid.startswith("religion_"):
            w1, w2 = mid.replace("religion_", "").split("_")
            return f"{w1.capitalize()}{ARR}{w2.capitalize()}"
        return mid

    # Pretty groups: partition of all model ids (disjoint, cover all)
    remaining = [m for m in all_ids if m not in ordered]
    pretty_groups = {
        **gender_de_transitions_to_ids,
        "Race": race_ids,
        "Religion": religion_ids,
        "English Gender": gender_en_ids,
        "English Pronouns": pronoun_ids,
        "English Number": pronoun_number_ids,
        "English Person": pronoun_person_ids,
    }

    # order and labels: use pretty labels as dict keys so heatmap and Venn show them consistently
    order = [mid for gids in pretty_groups.values() for mid in gids]
    pretty_labels = {mid: _pretty_label(mid) for mid in order}
    models_display = {pretty_labels[mid]: models_for_heatmap[mid] for mid in order}
    order_display = [pretty_labels[mid] for mid in order]
    pretty_groups_display = {k: [pretty_labels[mid] for mid in gids] for k, gids in pretty_groups.items()}

    topk = 1000
    output_path = os.path.join(args.experiment_dir, f"topk_overlap_heatmap_all_{topk}.pdf")
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

    # Venn diagram: three English pronoun GRADIENDs (same labels as heatmap)
    venn_pronoun_ids = [mid for mid in pronoun_ids if mid in ("pronoun_1SG_3PL", "pronoun_1SG_3SG", "pronoun_3SG_3PL")]
    if len(venn_pronoun_ids) < 3:
        venn_pronoun_ids = pronoun_ids[:3]
    if len(venn_pronoun_ids) >= 3:
        venn_models = {pretty_labels[mid]: models_for_heatmap[mid] for mid in venn_pronoun_ids}
        venn_output = os.path.join(args.experiment_dir, "topk_overlap_venn_three_english_pronouns.pdf")
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


if __name__ == "__main__":
    main()
