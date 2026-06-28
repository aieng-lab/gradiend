"""Label and grouping helpers shared with multilingual_gradiend_demo top-k overlap plots."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

LATEX_BIDI_ARROW = r"$\longleftrightarrow$"
LATEX_ARROW = r"$\rightarrow$"

CASE_PRETTY: Dict[str, str] = {
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

# English gender feature ids (M/F) use pronoun tokens on encoding axes to match overlap labels.
ENGLISH_GENDER_FEATURE_PRETTY: Dict[str, str] = {"M": "he", "F": "she"}

# Named domain buckets in display order (shared by top-k overlap + cross-encoding heatmaps).
# German case uses article subgroups (der, die, das, …) — same vocabulary as trainer
# article-pair brackets on the weight overlap heatmap.
# Rationale: morphology by language → English pronoun family together → lexical sentiment
# → register → social attributes.
DEMO_NAMED_DOMAIN_GROUP_ORDER: Tuple[str, ...] = (
    "Gender",
    "Pronouns",
    "Number",
    "Person",
    "Sentiment",
    "Formality",
    "Race",
    "Religion",
)

ARTICLE_MAPPING: Dict[str, str] = {
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

# Canonical case-cell order when flattening German feature subgroups.
GERMAN_CASE_FEATURE_ORDER: Tuple[str, ...] = (
    "masc_nom",
    "fem_nom",
    "neut_nom",
    "masc_acc",
    "fem_acc",
    "neut_acc",
    "masc_dat",
    "fem_dat",
    "neut_dat",
    "masc_gen",
    "fem_gen",
    "neut_gen",
)

GERMAN_ARTICLE_FEATURE_GROUP_ORDER: Tuple[str, ...] = (
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
)


def pretty_demo_trainer_id(trainer_id: str) -> str:
    """Row/column tick label for a trained GRADIEND (same as demo top-k overlap heatmap)."""
    mid = str(trainer_id)
    if mid == "gender_en":
        return f"he{LATEX_BIDI_ARROW}she"
    if mid == "sentiment_positive_negative":
        return f"Pos{LATEX_BIDI_ARROW}Neg"
    if mid.startswith("sentiment_"):
        rest = mid.removeprefix("sentiment_")
        if "_" in rest:
            positive, negative = rest.split("_", 1)
            return (
                f"{positive.capitalize()}"
                f"{LATEX_BIDI_ARROW}"
                f"{negative.capitalize()}"
            )
    if mid == "formality_informal_formal":
        return f"Inf{LATEX_BIDI_ARROW}Form"
    if mid.startswith("gender_de_"):
        rest = mid.removeprefix("gender_de_")
        parts = rest.split("_")
        if len(parts) >= 4:
            left = "_".join(parts[:2])
            right = "_".join(parts[2:])
            return (
                f"{CASE_PRETTY.get(left, left)}"
                f"{LATEX_BIDI_ARROW}"
                f"{CASE_PRETTY.get(right, right)}"
            )
    if mid.startswith("pronoun_number_"):
        return f"SG{LATEX_BIDI_ARROW}PL"
    if mid.startswith("pronoun_person_"):
        rest = mid.removeprefix("pronoun_person_")
        person_pretty = {
            "1vs2": f"1st{LATEX_BIDI_ARROW}2nd",
            "1vs3": f"1st{LATEX_BIDI_ARROW}3rd",
            "2vs3": f"2nd{LATEX_BIDI_ARROW}3rd",
        }
        return person_pretty.get(rest, rest.replace("vs", LATEX_BIDI_ARROW))
    if mid.startswith("pronoun_"):
        c1, c2 = mid.removeprefix("pronoun_").split("_", 1)
        return f"{c1}{LATEX_BIDI_ARROW}{c2}"
    if mid.startswith("race_"):
        w1, w2 = mid.removeprefix("race_").split("_", 1)
        return f"{w1.capitalize()}{LATEX_BIDI_ARROW}{w2.capitalize()}"
    if mid.startswith("religion_"):
        w1, w2 = mid.removeprefix("religion_").split("_", 1)
        return f"{w1.capitalize()}{LATEX_BIDI_ARROW}{w2.capitalize()}"
    return mid


def _gender_de_article_group_key(trainer_id: str) -> str:
    rest = str(trainer_id).removeprefix("gender_de_")
    parts = rest.split("_")
    if len(parts) < 4:
        return str(trainer_id)
    articles = sorted(
        ARTICLE_MAPPING["_".join(pair)]
        for pair in zip(*[iter(parts)] * 2)
    )
    return LATEX_BIDI_ARROW.join(articles)


def build_demo_trainer_order_and_groups(
    trainer_ids: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Trainer order and bracket groups (same as ``_build_plot_order_and_groups``)."""
    all_ids = list(trainer_ids)
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
    gender_de_transitions_to_ids: Dict[str, List[str]] = defaultdict(list)
    for model_id in gender_de_ids:
        gender_de_transitions_to_ids[_gender_de_article_group_key(model_id)].append(model_id)

    named_groups: Dict[str, List[str]] = {
        "Gender": gender_en_ids,
        "Sentiment": sentiment_ids,
        "Formality": formality_ids,
        "Pronouns": pronoun_ids,
        "Number": pronoun_number_ids,
        "Person": pronoun_person_ids,
        "Race": race_ids,
        "Religion": religion_ids,
    }

    pretty_groups: Dict[str, List[str]] = {}
    for article_group, ids in sorted(gender_de_transitions_to_ids.items()):
        if ids:
            pretty_groups[str(article_group)] = ids
    for group_name in DEMO_NAMED_DOMAIN_GROUP_ORDER:
        ids = named_groups.get(group_name, [])
        if ids:
            pretty_groups[group_name] = ids

    ordered = [mid for ids in pretty_groups.values() for mid in ids]
    return ordered, pretty_groups


def pretty_demo_feature_id(feature_id: str) -> str:
    """Tick label for a single feature-class id (German uses ``CASE_PRETTY``)."""
    fid = str(feature_id)
    if fid in CASE_PRETTY:
        return CASE_PRETTY[fid]
    if fid in ENGLISH_GENDER_FEATURE_PRETTY:
        return ENGLISH_GENDER_FEATURE_PRETTY[fid]
    if fid in {"white", "black", "asian", "christian", "muslim", "jewish"}:
        return fid.capitalize()
    if fid == "positive":
        return "Pos"
    if fid == "negative":
        return "Neg"
    return fid


def pretty_demo_transition_id(transition_id: str) -> str:
    """Directed transition column label using the same case/feature naming."""
    tid = str(transition_id)
    if "->" not in tid:
        return pretty_demo_feature_id(tid)
    src, tgt = tid.split("->", 1)
    return f"{pretty_demo_feature_id(src)}{LATEX_ARROW}{pretty_demo_feature_id(tgt)}"


def build_german_article_feature_subgroups(
    feature_ids: Sequence[str],
) -> Dict[str, List[str]]:
    """Bracket groups for German case cells keyed by surface article (der, die, …)."""
    present = {str(fid) for fid in feature_ids}
    groups: Dict[str, List[str]] = {}
    for article in GERMAN_ARTICLE_FEATURE_GROUP_ORDER:
        ids = [
            fid
            for fid in GERMAN_CASE_FEATURE_ORDER
            if fid in present and ARTICLE_MAPPING.get(fid) == article
        ]
        if ids:
            groups[article] = ids
    return groups


def build_demo_feature_plot_groups(
    feature_ids: Sequence[str],
    *,
    include_formality: bool = False,
) -> Dict[str, List[str]]:
    """Feature-class bracket groups (German article subgroups + domain buckets)."""
    present = {str(fid) for fid in feature_ids}
    pretty_groups: Dict[str, List[str]] = dict(build_german_article_feature_subgroups(feature_ids))
    buckets: Dict[str, List[str]] = {
        "Gender": [fid for fid in ("M", "F") if fid in present],
        "Sentiment": [fid for fid in ("positive", "negative") if fid in present],
        "Pronouns": [
            fid for fid in ("1SG", "1PL", "2SGPL", "3SG", "3PL") if fid in present
        ],
        "Race": [fid for fid in ("white", "black", "asian") if fid in present],
        "Religion": [fid for fid in ("christian", "muslim", "jewish") if fid in present],
    }
    if include_formality:
        buckets["Formality"] = [fid for fid in ("informal", "formal") if fid in present]

    for label in DEMO_NAMED_DOMAIN_GROUP_ORDER:
        ids = buckets.get(label, [])
        if ids:
            pretty_groups[label] = ids
    return pretty_groups


def build_demo_trainer_label_mapping(trainer_ids: Sequence[str]) -> Dict[str, str]:
    return {str(tid): pretty_demo_trainer_id(tid) for tid in trainer_ids}


def build_demo_feature_label_mapping(feature_ids: Sequence[str]) -> Dict[str, str]:
    return {str(fid): pretty_demo_feature_id(fid) for fid in feature_ids}


def build_demo_transition_label_mapping(transition_ids: Sequence[str]) -> Dict[str, str]:
    return {str(tid): pretty_demo_transition_id(tid) for tid in transition_ids}


def demo_topk_overlap_style_kwargs(**overrides: object) -> Dict[str, object]:
    """Matplotlib kwargs for ``plot_topk_overlap_heatmap`` in the multilingual demo."""
    style: Dict[str, object] = {
        "scale": "linear",
        "scale_gamma": 0.5,
        "group_label_fontsize": 16,
        "tick_label_fontsize": 14,
        "axis_label_fontsize": 16,
        "annot": True,
        "annot_fontsize": 9,
        "cbar_pad": 0.15,
        "cbar_fontsize": 18,
        "cbar_shrink": 0.75,
        "percentages": True,
    }
    style.update(overrides)
    tick = style.get("tick_label_fontsize")
    axis = style.get("axis_label_fontsize")
    if isinstance(tick, (int, float)) and isinstance(axis, (int, float)):
        style["axis_label_fontsize"] = max(float(axis), float(tick) + 2)
    return style


def demo_encoding_heatmap_style_kwargs(**overrides: object) -> Dict[str, object]:
    """Cross-encoding heatmaps: same typography as overlap; diverging signed scale."""
    style = demo_topk_overlap_style_kwargs(
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        annot=False,
        percentages=False,
        scale_gamma=None,
        cbar_label="Encoding",
    )
    style.update(overrides)
    return style


def demo_encoding_heatmap_normalized_style_kwargs(**overrides: object) -> Dict[str, object]:
    """Row-normalized cross-encoding heatmaps: auto color scale (diagonal fixed at 1)."""
    style = demo_encoding_heatmap_style_kwargs(
        annot=True,
        annot_fmt=".2f",
        vmin=None,
        vmax=None,
        cbar_label="Relative encoding",
    )
    style.update(overrides)
    return style
