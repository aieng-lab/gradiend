"""
Top-k selection and Venn plotting utilities for GRADIEND models.

Uses matplotlib_venn for 2–3 sets and the venn package for 4–6 sets.
Neither package is a hard requirement; if missing, an informative error
tells the user how to install (e.g. pip install matplotlib-venn).
matplotlib is also optional; if missing, plotting raises ImportError with install instructions.
"""

from typing import Dict, List, Optional, Tuple, Union

from gradiend.visualizer.plot_optional import _require_matplotlib


def _require_matplotlib_venn() -> None:
    """Import matplotlib_venn; on failure raise an error with install instructions."""
    try:
        from matplotlib_venn import venn2, venn3  # noqa: F401
        return
    except ImportError as e:
        raise ImportError(
            "Venn diagram plotting for 2–3 models requires the matplotlib-venn package. "
            "Install it with: pip install matplotlib-venn"
        ) from e


def _require_venn() -> None:
    """Import venn (for 4+ sets); on failure raise an error with install instructions."""
    try:
        from venn import venn  # noqa: F401
        return
    except ImportError as e:
        raise ImportError(
            "Venn diagram plotting for 4+ models requires the venn package. "
            "Install it with: pip install venn"
        ) from e


def compute_topk_sets(
    models: Dict[str, object],
    topk: int = 100,
    part: str = "decoder-weight",
) -> Tuple[Dict[str, List[int]], List[int], List[int]]:
    """
    Compute top-k weight index sets for multiple models and their intersection/union.

    Uses the weight view: one importance per base-model weight (GRADIEND input dimension).
    Supports any number of models (e.g. 2–6 GRADIENDs for comparison).

    Args:
        models: Mapping from model identifier to a model with ``get_topk_weights(part, topk)``
            (e.g. ``ModelWithGradiend``).
        topk: Number of top weights to select per model.
        part: ``'encoder-weight'`` or ``'decoder-weight'``.

    Returns:
        per_model: dict mapping model_id -> list of weight indices (flattened base-model weights)
        intersection: weight indices that appear in all models' top-k sets
        union: weight indices that appear in at least one top-k set
    """
    per_model: Dict[str, List[int]] = {}

    for model_id, model in models.items():
        per_model[model_id] = model.get_topk_weights(part=part, topk=topk)

    sets = {k: set(v) for k, v in per_model.items()}
    all_sets = list(sets.values())
    if not all_sets:
        return per_model, [], []

    intersection = list(set.intersection(*all_sets))
    union = list(set.union(*all_sets))
    return per_model, sorted(intersection), sorted(union)


def plot_topk_venn(
    models: Dict[str, object],
    topk: int = 100,
    part: str = "decoder-weight",
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    circle_names_fontsize: Optional[Union[int, float]] = None,
    region_counts_fontsize: Optional[Union[int, float]] = None,
    patch_linewidth: Optional[float] = None,
    alpha: float = 0.5,
    title: Optional[str] = None,
) -> None:
    """
    Plot a Venn diagram for top-k weight index sets (2–6 models).

    - 2–3 models: uses matplotlib_venn (venn2/venn3). Requires ``pip install matplotlib-venn``.
    - 4–6 models: uses the venn package. Requires ``pip install venn``.

    If the required package is missing, raises ImportError with install instructions.
    No fallback to other plot types.

    Args:
        models: Mapping from model identifier (or display label) to ModelWithGradiend instance.
            Dict keys are used as set labels; use pretty labels as keys for consistent display.
        topk: Number of top weights per model (base-model weight indices).
        part: ``'encoder-weight'`` or ``'decoder-weight'``.
        output_path: Optional path to save the figure.
        show: If True, call ``plt.show()`` so the plot is displayed.
        figsize: (width, height) in inches. Default (6, 6) for 2–3 models, (8, 8) for 4–6.
        circle_names_fontsize: Font size for the name of each circle (e.g. model label). Default 12 (2–3 sets) or 13 (4–6 sets).
        region_counts_fontsize: Font size for the numbers inside each region (overlap counts). Default 10 (2–3 sets) or 11 (4–6 sets).
        patch_linewidth: Line width of Venn patches. Default 3.0.
        alpha: Patch transparency in [0, 1]. Default 0.5.
        title: Optional figure title.
    """
    model_ids = list(models.keys())
    n_models = len(model_ids)

    if n_models < 2:
        raise ValueError("At least 2 models are required for a Venn diagram.")
    if n_models > 6:
        raise ValueError("Venn diagrams support at most 6 models.")

    plt = _require_matplotlib()
    per_model, _, _ = compute_topk_sets(models, topk=topk, part=part)
    sets_dict = {mid: set(per_model[mid]) for mid in model_ids}

    if figsize is None:
        figsize = (8, 8) if n_models >= 4 else (6, 6)
    # Default font sizes scaled to figsize so circles stay visually dominant (not huge text)
    if circle_names_fontsize is None:
        circle_names_fontsize = 20 if figsize[0] >= 8 else 18
    if region_counts_fontsize is None:
        region_counts_fontsize = 18 if figsize[0] >= 8 else 16

    if n_models == 2:
        _require_matplotlib_venn()
        from matplotlib_venn import venn2

        A, B = model_ids
        set_a, set_b = sets_dict[A], sets_dict[B]
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        ab = len(set_a & set_b)

        plt.figure(figsize=figsize)
        v = venn2(
            subsets=(only_a, only_b, ab),
            set_labels=(model_ids[0], model_ids[1]),
            alpha=alpha,
        )
        _style_venn2_venn3(
            v, topk,
            circle_names_fontsize=circle_names_fontsize,
            region_counts_fontsize=region_counts_fontsize,
            patch_linewidth=patch_linewidth,
        )

    elif n_models == 3:
        _require_matplotlib_venn()
        from matplotlib_venn import venn3

        A, B, C = model_ids
        set_a, set_b, set_c = sets_dict[A], sets_dict[B], sets_dict[C]
        only_a = len(set_a - set_b - set_c)
        only_b = len(set_b - set_a - set_c)
        only_c = len(set_c - set_a - set_b)
        ab = len((set_a & set_b) - set_c)
        ac = len((set_a & set_c) - set_b)
        bc = len((set_b & set_c) - set_a)
        abc = len(set_a & set_b & set_c)

        plt.figure(figsize=figsize)
        v = venn3(
            subsets=(only_a, only_b, ab, only_c, ac, bc, abc),
            set_labels=(model_ids[0], model_ids[1], model_ids[2]),
            alpha=alpha,
        )
        _style_venn2_venn3(
            v, topk,
            circle_names_fontsize=circle_names_fontsize,
            region_counts_fontsize=region_counts_fontsize,
            patch_linewidth=patch_linewidth,
        )

    else:
        _require_venn()
        from venn import venn

        plt.figure(figsize=figsize)
        ax = plt.gca()
        venn(sets_dict, ax=ax, legend_loc="upper center")
        for txt in ax.texts:
            if txt:
                txt.set_color("black")
        if circle_names_fontsize is not None:
            for txt in ax.texts:
                if txt:
                    txt.set_fontsize(circle_names_fontsize)

    if title:
        plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()


def _style_venn2_venn3(
    v: object,
    topk: int,
    circle_names_fontsize: Optional[Union[int, float]] = None,
    region_counts_fontsize: Optional[Union[int, float]] = None,
    patch_linewidth: Optional[float] = None,
) -> None:
    """Apply circle names and region counts font sizes and patch linewidth."""
    if region_counts_fontsize is None:
        region_counts_fontsize = 10
    if circle_names_fontsize is None:
        circle_names_fontsize = 12
    if patch_linewidth is None:
        patch_linewidth = 3.0
    if hasattr(v, "subset_labels"):
        for txt in v.subset_labels:
            if txt:
                txt.set_fontsize(region_counts_fontsize)
    if hasattr(v, "set_labels"):
        for txt in v.set_labels:
            if txt:
                txt.set_fontsize(circle_names_fontsize)
    if hasattr(v, "patches"):
        for patch in v.patches:
            if patch:
                patch.set_linewidth(patch_linewidth)


def plot_topk_overlap_venn(
    models: Dict[str, object],
    topk: int = 1000,
    part: str = "decoder-weight",
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    circle_names_fontsize: Optional[Union[int, float]] = None,
    region_counts_fontsize: Optional[Union[int, float]] = None,
    patch_linewidth: Optional[float] = None,
    alpha: float = 0.5,
    title: Optional[str] = None,
) -> Dict[str, object]:
    """
    Plot top-k weight index set intersection across multiple GRADIEND models and return overlap stats.

    This is a standalone function: pass a dict of label -> ModelWithGradiend. Dict keys are used
    as set labels; use pretty labels (e.g. ``"3SG $\\longleftrightarrow$ 3PL"``) as keys for
    consistent display with the heatmap. Uses Venn diagrams: matplotlib_venn for 2–3 models, venn
    package for 4–6 models. Missing packages raise ImportError with install instructions.

    Args:
        models: Mapping from display label (or model id) to ModelWithGradiend instance.
        topk: Number of top weights to consider per model (base-model weight indices).
        part: ``'encoder-weight'`` or ``'decoder-weight'``.
        output_path: Optional path to save the figure.
        show: If True, display the plot (default True so something is shown when running).
        figsize: (width, height) in inches. Default (6, 6) for 2–3 models, (8, 8) for 4–6.
        circle_names_fontsize: Font size for the name of each circle (e.g. model label). Default scales with figsize.
        region_counts_fontsize: Font size for the numbers inside each region (overlap counts). Default scales with figsize.
        patch_linewidth: Line width of Venn patch borders. Default 3.0.
        alpha: Patch transparency in [0, 1].
        title: Optional figure title.

    Returns:
        Dict with keys: ``per_model`` (model_id -> list of weight indices), ``intersection``,
        ``union``, ``topk``, ``part``.
    """
    if not models:
        return {"per_model": {}, "intersection": [], "union": [], "topk": topk, "part": part}
    if not isinstance(models, dict):
        models = {"model": models}

    per_model, intersection, union = compute_topk_sets(models, topk=topk, part=part)
    plot_topk_venn(
        models,
        topk=topk,
        part=part,
        output_path=output_path,
        show=show,
        figsize=figsize,
        circle_names_fontsize=circle_names_fontsize,
        region_counts_fontsize=region_counts_fontsize,
        patch_linewidth=patch_linewidth,
        alpha=alpha,
        title=title,
    )

    return {
        "per_model": per_model,
        "intersection": intersection,
        "union": union,
        "topk": topk,
        "part": part,
    }
