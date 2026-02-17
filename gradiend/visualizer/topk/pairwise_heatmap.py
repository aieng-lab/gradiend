"""
Pairwise overlap heatmap for top-k weight sets.

Requires matplotlib and seaborn. If missing, raises ImportError with install instructions.
"""

from typing import Dict, Union, List, Optional, Tuple

from gradiend.visualizer.plot_optional import _require_matplotlib, _require_seaborn
from gradiend.visualizer.topk.venn_ import compute_topk_sets


def _validate_and_complete_pretty_groups(
    pretty_groups: Dict[str, List[str]], model_ids: List[str]
) -> None:
    """Validate pretty_groups (disjoint) and auto-add 'Other' for any uncovered ids."""
    covered = set()
    for group_name, ids in pretty_groups.items():
        for mid in ids:
            if mid not in model_ids:
                raise ValueError(
                    f"pretty_groups: id {mid!r} in group {group_name!r} is not in model_ids"
                )
            if mid in covered:
                raise ValueError(
                    f"pretty_groups: id {mid!r} appears in multiple groups (not disjoint)"
                )
            covered.add(mid)
    missing = set(model_ids) - covered
    if missing:
        pretty_groups["Other"] = sorted(missing)


def plot_topk_overlap_heatmap(
    models: Dict[str, object],
    topk: int = 1000,
    part: str = "decoder-weight",
    value: str = "intersection",
    order: Union[str, List[str]] = "input",
    cluster: bool = False,
    annot: Union[bool, str] = "auto",
    fmt: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[Union[str, bool]] = False,
    output_path: Optional[str] = None,
    show: bool = True,
    return_data: bool = True,
    pretty_labels: Optional[Dict[str, str]] = None,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    scale: str = "linear",
    scale_gamma: Optional[float] = None,
    show_values: Optional[bool] = None,
    annot_fontsize: Optional[Union[int, float]] = None,
    tick_label_fontsize: Optional[Union[int, float]] = None,
    group_label_fontsize: Optional[Union[int, float]] = None,
    cbar_pad: Optional[float] = None,
) -> Dict[str, object]:
    """
    Plot a heatmap visualizing pairwise overlap between top-k weight index sets
    of multiple GRADIEND models.

    This function computes, for each model, the top-k most important base-model
    weight indices (via ``get_topk_weights``) and visualizes their *pairwise
    intersections* in a square heatmap of shape (N, N), where N is the number
    of models.

    A key assumption of this plot is that **all models use the same ``topk``**,
    i.e. each set has identical cardinality |A| = |B| = k. Under this assumption,
    the raw intersection size |A ∩ B| is a meaningful and monotonic similarity
    measure. Optionally, it can be normalized by k to obtain a scale-invariant
    similarity in [0, 1].

    The diagonal entries always correspond to self-overlap and therefore equal
    ``topk`` (or 1.0 if normalized).

    Typical use cases:
      - Comparing how similarly different GRADIEND models select important
        base-model weights.
      - Quantifying stability of top-k selections across seeds, datasets,
        or debiasing variants.
      - Identifying clusters of models with highly overlapping explanations.

    Args:
        models:
            Mapping from model identifier (used as axis labels) to a model
            instance implementing
            ``get_topk_weights(part: str, topk: int) -> List[int]``.

        topk:
            Number of top-ranked base-model weights selected per model.
            Must be identical across all models for meaningful comparison.

        part:
            Which part of the base model the weights belong to.
            Typically ``"encoder-weight"`` or ``"decoder-weight"``.

        value:
            Quantity visualized in the heatmap.
            - ``"intersection"``: raw overlap size |A ∩ B| (integer, range [0, k]).
            - ``"intersection_frac"``: normalized overlap |A ∩ B| / k (float, range [0, 1]).
            The normalized version is recommended when comparing across different
            ``topk`` values or across experiments.

        order:
            Ordering of models on both axes. Ignored if ``pretty_groups`` is provided.
            - ``"input"``: preserve the insertion order of the ``models`` dict.
            - ``"name"``: sort model identifiers alphabetically.
            - explicit ``List[str]``: user-defined order (must match keys in ``models``).

        cluster:
            If True, reorder models using a lightweight greedy similarity heuristic
            (based on normalized intersection) so that similar models appear close
            to each other. This improves visual interpretability without introducing
            additional dependencies (e.g. SciPy).

        annot:
            Whether to annotate each heatmap cell with its numeric value.
            - ``True``: always annotate.
            - ``False``: never annotate.
            - ``"auto"``: annotate only if the number of models is small (≤ 25),
              to avoid clutter.

        fmt:
            String format used for annotations (e.g. ``"d"``, ``".2f"``).
            If None, a sensible default is chosen based on ``value``.

        figsize:
            Size of the matplotlib figure in inches (width, height).
            If None, uses ``(max(14, n*0.4), max(14, n*0.4))``.

        cmap:
            Matplotlib/Seaborn colormap used for the heatmap.

        vmin, vmax:
            Lower and upper bounds for the colormap.
            Defaults are chosen automatically:
              - [0, k] for ``"intersection"``
              - [0, 1] for ``"intersection_frac"``

        title:
            Optional plot title. If None, a descriptive default title is generated.

        output_path:
            Optional file path. If provided, the figure is saved to disk
            (with ``bbox_inches="tight"``).

        show:
            If True, call ``plt.show()`` to display the figure.

        return_data:
            If True, return the computed overlap matrix and auxiliary data
            for downstream analysis.

        pretty_labels:
            Optional mapping from model id (feature_class_id) to display label.
            If provided, tick labels use these instead of raw ids.

        pretty_groups:
            Optional mapping from group name to list of model ids. Must be disjoint.
            If not all model ids are covered, ``"Other"`` is auto-added for the rest.
            When provided, ``order`` is derived from the group order (ids concatenated
            in iteration order). Grouping is shown on the top and right sides with
            group names.

        scale:
            Color scale for the heatmap: ``"linear"``, ``"log"``, ``"sqrt"``
            (gamma=0.5), or ``"power"`` (uses ``scale_gamma``).

        scale_gamma:
            Gamma for power-law scale when ``scale="power"``. E.g. 0.5 = sqrt.

        show_values:
            Whether to annotate cells with numeric values. If None, uses ``annot``.

        annot_fontsize:
            Font size for cell value annotations. If None, uses default.

        tick_label_fontsize:
            Font size for axis tick labels. If None, uses default.

        group_label_fontsize:
            Font size for group labels (top/right). If None, uses default.

        cbar_pad:
            Padding between heatmap and colorbar (fraction of axes width).
            Larger values shift the colorbar further right. If None, uses default.

    Returns:
        If return_data is True, a dictionary with the following keys:
            - ``per_model``: model_id -> list of top-k weight indices
            - ``model_ids``: list of model identifiers in plotted order
            - ``matrix``: NxN nested list containing the overlap values
            - ``value``: the selected overlap measure
            - ``topk``: the value of ``topk``
    """

    if not models or len(models) < 2:
        raise ValueError("At least 2 models are required.")
    if not isinstance(models, dict):
        models = {"model": models}

    plt = _require_matplotlib()
    sns = _require_seaborn()
    per_model, _, _ = compute_topk_sets(models, topk=topk, part=part)
    sets = {mid: set(per_model[mid]) for mid in per_model.keys()}

    # ordering
    if pretty_groups is not None:
        _validate_and_complete_pretty_groups(pretty_groups, list(sets.keys()))
        model_ids = []
        for ids in pretty_groups.values():
            model_ids.extend(ids)
    elif isinstance(order, list):
        model_ids = order
    elif order == "name":
        model_ids = sorted(sets.keys())
    else:
        model_ids = list(sets.keys())

    # optional lightweight clustering (greedy path by intersection_frac)
    def score(a: str, b: str) -> float:
        inter = len(sets[a] & sets[b])
        return inter / topk if topk else 0.0

    if cluster:
        remaining = model_ids[:]
        path = [remaining.pop(0)]
        while remaining:
            last = path[-1]
            best = max(remaining, key=lambda m: score(last, m))
            path.append(best)
            remaining.remove(best)
        model_ids = path

    n = len(model_ids)

    # matrix
    mat = [[0.0] * n for _ in range(n)]
    for i, mi in enumerate(model_ids):
        for j, mj in enumerate(model_ids):
            inter = len(sets[mi] & sets[mj])
            mat[i][j] = float(inter) if value == "intersection" else (inter / topk if topk else 0.0)

    # defaults
    if fmt is None:
        fmt = "d" if value == "intersection" else ".2f"
    if figsize is None:
        s = max(14.0, n * 0.4)
        figsize = (s, s)
    _annot = annot if show_values is None else show_values
    if _annot == "auto":
        _annot = n <= 25

    if vmin is None and value == "intersection_frac":
        vmin = 0.0
    if vmax is None and value == "intersection_frac":
        vmax = 1.0
    if vmin is None and value == "intersection":
        vmin = 0.0
    if vmax is None and value == "intersection":
        vmax = float(topk)

    if title in {None, True}:
        y = "|A ∩ B|" if value == "intersection" else "|A ∩ B| / k"
        title = f"Top-{topk} pairwise overlap ({y})"

    # tick labels
    xticklabels = [pretty_labels.get(mid, mid) if pretty_labels else mid for mid in model_ids]
    yticklabels = xticklabels[:]

    linecolor = "white"
    boundary_lw = 2.5

    # scale: linear, log, sqrt, or power (color scale only; data values unchanged)
    import numpy as np
    from matplotlib.colors import LogNorm, PowerNorm

    mat_arr = np.array(mat)
    norm = None
    _vmin = float(vmin) if vmin is not None else (0.0 if value == "intersection_frac" else 0.0)
    _vmax = float(vmax) if vmax is not None else (1.0 if value == "intersection_frac" else float(topk))
    eps = max(1e-10, np.finfo(float).tiny)
    if scale == "log":
        norm = LogNorm(vmin=max(eps, _vmin), vmax=_vmax)
    elif scale == "sqrt":
        norm = PowerNorm(gamma=0.5, vmin=_vmin, vmax=_vmax)
    elif scale == "power" and scale_gamma is not None:
        norm = PowerNorm(gamma=scale_gamma, vmin=_vmin, vmax=_vmax)

    annot_kws = {}
    if annot_fontsize is not None:
        annot_kws["fontsize"] = annot_fontsize

    cbar_kws = {"shrink": 0.75}
    if cbar_pad is not None:
        cbar_kws["pad"] = cbar_pad

    # plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        mat_arr,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        annot=_annot,
        fmt=fmt,
        annot_kws=annot_kws,
        square=True,
        cbar=True,
        cbar_kws=cbar_kws,
        linewidths=0.5,
        linecolor=linecolor,
    )

    # tick label font size
    if tick_label_fontsize is not None:
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_fontsize)
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_fontsize)

    if isinstance(title, str):
        ax.set_title(title)

    # add group indicators on top and right when pretty_groups provided
    if pretty_groups is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        id_to_group = {mid: gname for gname, ids in pretty_groups.items() for mid in ids}
        group_col_spans = {}
        group_row_spans = {}
        for gname, ids in pretty_groups.items():
            indices = [model_ids.index(mid) for mid in ids]
            if indices:
                group_col_spans[gname] = (min(indices), max(indices))
                group_row_spans[gname] = (min(indices), max(indices))

        # Thick white lines at group boundaries (same color as cell borders)
        for j in range(1, n):
            if id_to_group[model_ids[j]] != id_to_group[model_ids[j - 1]]:
                ax.axvline(x=j, color=linecolor, linewidth=boundary_lw, zorder=5)
        for i in range(1, n):
            if id_to_group[model_ids[i]] != id_to_group[model_ids[i - 1]]:
                ax.axhline(y=i, color=linecolor, linewidth=boundary_lw, zorder=5)

        # Inset so group lines have margin; gap between two group lines matches matrix boundary width
        line_margin = 0.1  # in data coords (gap of 2*line_margin between adjacent groups)
        group_fontsize = (
            group_label_fontsize
            if group_label_fontsize is not None
            else (tick_label_fontsize + 2 if tick_label_fontsize is not None else max(11, min(14, 280 / n)))
        )

        # Top strip: single line per group (inset) + text labels close to line
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="8%", pad=0.02)
        ax_top.set_xlim(0, n)
        ax_top.set_ylim(0, 1)
        ax_top.set_aspect("auto")
        ax_top.axis("off")
        if title:
            ax_top.set_title(title, fontsize=plt.rcParams["axes.titlesize"])
        ax.set_title("")

        line_y = 0.12
        text_y = 0.28
        for gname, (start, end) in group_col_spans.items():
            x1, x2 = start, end + 1
            x1_inset = x1 + line_margin
            x2_inset = x2 - line_margin
            x_center = (x1 + x2 - 1) / 2 + 0.5
            ax_top.hlines(line_y, x1_inset, x2_inset, colors="gray", linewidth=3)
            ax_top.text(
                x_center, text_y, gname,
                rotation=90, ha="center", va="bottom",
                fontsize=group_fontsize, transform=ax_top.transData,
            )

        # Right strip: single line per group (inset) + horizontal text (left to right)
        ax_right = divider.append_axes("right", size="8%", pad=0.02)
        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(n, 0)
        ax_right.set_aspect("auto")
        ax_right.axis("off")

        line_x = 0.12
        text_x = 0.28
        for gname, (start, end) in group_row_spans.items():
            y1, y2 = start, end + 1
            y1_inset = y1 + line_margin
            y2_inset = y2 - line_margin
            y_center = (y1 + y2 - 1) / 2 + 0.5
            ax_right.vlines(line_x, y1_inset, y2_inset, colors="gray", linewidth=3)
            ax_right.text(
                text_x, y_center, gname,
                rotation=0, ha="left", va="center",
                fontsize=group_fontsize, transform=ax_right.transData,
            )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    if return_data:
        return {
            "per_model": per_model,
            "model_ids": model_ids,
            "matrix": mat,
            "value": value,
            "topk": topk,
            "part": part,
        }