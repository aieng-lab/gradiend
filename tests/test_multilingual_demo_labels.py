"""Tests for multilingual demo heatmap label helpers (overlap + encoding parity)."""

from gradiend.visualizer.multilingual_demo_labels import (
    build_demo_feature_label_mapping,
    build_demo_feature_plot_groups,
    build_demo_trainer_label_mapping,
    build_demo_trainer_order_and_groups,
    build_demo_transition_label_mapping,
    build_german_article_feature_subgroups,
    pretty_demo_feature_id,
    pretty_demo_trainer_id,
    pretty_demo_transition_id,
)


def test_pretty_demo_feature_id_english_and_german():
    assert pretty_demo_feature_id("M") == "he"
    assert pretty_demo_feature_id("F") == "she"
    assert pretty_demo_feature_id("1SG") == "1SG"
    assert pretty_demo_feature_id("masc_nom") == "Masc.Nom"
    assert pretty_demo_feature_id("white") == "White"


def test_pretty_demo_transition_id_uses_arrows():
    label = pretty_demo_transition_id("M->F")
    assert "he" in label
    assert "she" in label
    assert "rightarrow" in label or r"\rightarrow" in label


def test_pretty_demo_trainer_id_uses_case_pretty_not_articles():
    label = pretty_demo_trainer_id("gender_de_masc_nom_masc_dat")
    assert "Masc.Nom" in label
    assert "Masc.Dat" in label
    assert "der" not in label
    assert "dem" not in label
    assert pretty_demo_trainer_id("gender_en") == r"he$\longleftrightarrow$she"


def test_build_demo_feature_plot_groups_uses_article_subgroups():
    groups = build_demo_feature_plot_groups(["masc_nom", "fem_nom", "M", "F", "white"])
    assert groups["der"] == ["masc_nom"]
    assert groups["die"] == ["fem_nom"]
    assert groups["Gender"] == ["M", "F"]
    assert groups["Race"] == ["white"]


def test_build_german_article_feature_subgroups_orders_by_case_grid():
    groups = build_german_article_feature_subgroups(
        ["fem_acc", "masc_nom", "neut_dat", "masc_acc"],
    )
    assert groups["der"] == ["masc_nom"]
    assert groups["die"] == ["fem_acc"]
    assert groups["den"] == ["masc_acc"]
    assert groups["dem"] == ["neut_dat"]


def test_pretty_demo_trainer_id_sentiment_word_pairs():
    label = pretty_demo_trainer_id("sentiment_good_bad")
    assert "Good" in label
    assert "Bad" in label
    assert pretty_demo_trainer_id("sentiment_positive_negative") == r"Pos$\longleftrightarrow$Neg"


def test_build_demo_trainer_order_and_groups():
    ordered, groups = build_demo_trainer_order_and_groups(
        [
            "race_white_black",
            "gender_de_masc_nom_masc_dat",
            "gender_en",
            "sentiment_positive_negative",
        ]
    )
    assert ordered[0] == "gender_de_masc_nom_masc_dat"
    assert ordered[1] == "gender_en"
    assert ordered[2] == "sentiment_positive_negative"
    assert ordered[3] == "race_white_black"
    assert list(groups.keys())[-1] == "Race"
    assert groups["Race"] == ["race_white_black"]
    assert groups[r"dem$\longleftrightarrow$der"] == ["gender_de_masc_nom_masc_dat"]


def test_build_demo_feature_plot_groups_follows_domain_order():
    groups = build_demo_feature_plot_groups(
        ["white", "M", "masc_nom", "positive", "1SG"],
    )
    assert list(groups.keys()) == [
        "der",
        "Gender",
        "Pronouns",
        "Sentiment",
        "Race",
    ]


def test_label_mappings_cover_all_ids():
    features = ["M", "F", "masc_nom"]
    trainers = ["gender_en", "race_white_black"]
    feature_labels = build_demo_feature_label_mapping(features)
    assert set(feature_labels) == {"M", "F", "masc_nom"}
    assert feature_labels["masc_nom"] == "Masc.Nom"
    assert feature_labels["M"] == "he"
    assert set(build_demo_trainer_label_mapping(trainers)) == set(trainers)
