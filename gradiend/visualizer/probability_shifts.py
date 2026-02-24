"""
Probability shifts plot: target token probabilities vs learning rate,
grouped by dataset class with selection metrics highlighted.

Requires matplotlib. If missing, raises ImportError with install instructions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def plot_probability_shifts(
    trainer: Optional[Any] = None,
    decoder_results: Optional[Dict[str, Any]] = None,
    plotting_data: Optional[Dict[str, Any]] = None,
    class_ids: Optional[List[str]] = None,
    target_class: Optional[str] = None,
    increase_target_probabilities: bool = True,
    output: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> str:
    """
    Plot decoder probability shifts vs learning rate for a single target (target_class).

    All plots use the same feature factor (the one that pushes toward target_class, as in rewrite_base_model).
    Base model (lr=0) is prepended to each line; x-tick at lr=0 is labeled "base".

    Subplots:
    - LMS: Language Modeling Score (one line)
    - One per dataset: P(class) on that dataset. The metric used to select the learning rate is
      highlighted (thicker curve, labeled "selection") and the selected (lr, value) is marked with a star.

    Args:
        trainer: Trainer instance (used to get model/data if needed)
        decoder_results: Decoder evaluation result dict (summary entries at top level, e.g. result['3SG'], plus 'grid')
        plotting_data: Extended grid data from analyze_decoder_for_plotting (with probs_by_dataset)
        class_ids: Classes to plot (defaults to all_classes or target_classes from trainer)
        target_class: Target class to focus on (same as rewrite_base_model target_class). Default: first target class.
        increase_target_probabilities: If True, plot for strengthen (default). If False, use weaken summary (target_class_weaken).
        output: Path to save plot. If None and trainer.experiment_dir is set, saves there.
        show: Whether to display the plot
        figsize: Figure size in inches. If None, auto-calculated.
        **kwargs: Additional arguments passed to matplotlib

    Returns:
        Path to saved plot file (or empty string if not saved)
    """
    plt = _require_matplotlib()
    
    # Get decoder_results and plotting_data
    if decoder_results is None and trainer is not None:
        decoder_results = trainer.evaluate_decoder(use_cache=kwargs.get("use_cache"))
    
    if plotting_data is None and trainer is not None:
        plotting_data = trainer.analyze_decoder_for_plotting(
            decoder_results=decoder_results,
            class_ids=class_ids,
            use_cache=kwargs.get("use_cache"),
        )
    
    if decoder_results is None or plotting_data is None:
        raise ValueError("Must provide decoder_results and plotting_data, or trainer to generate them")
    
    grid = plotting_data.get("plotting_data", plotting_data.get("grid", {}))
    _reserved = {"grid", "plot_path", "plot_paths"}
    summary = {k: v for k, v in decoder_results.items() if k not in _reserved}
    
    # Determine classes to plot
    if class_ids is None:
        if trainer is not None:
            class_ids = trainer.all_classes if trainer.all_classes else trainer.target_classes
        else:
            # Infer from grid data
            class_ids = set()
            for entry in grid.values():
                probs_by_dataset = entry.get("probs_by_dataset", {})
                for dataset_probs in probs_by_dataset.values():
                    if isinstance(dataset_probs, dict):
                        class_ids.update(dataset_probs.keys())
            class_ids = sorted(list(class_ids))
    
    if not class_ids:
        raise ValueError("No classes to plot. Provide class_ids or ensure classes are available.")
    
    # Resolve target_class and summary key (strengthen vs weaken)
    if target_class is None:
        if trainer is not None and getattr(trainer, "target_classes", None):
            target_class = trainer.target_classes[0]
        elif summary:
            candidates = [k for k in summary.keys() if k != "lms" and not k.endswith("_weaken")]
            target_class = candidates[0] if candidates else None
        if target_class is None and class_ids:
            target_class = list(class_ids)[0]
    if not increase_target_probabilities and target_class and not target_class.endswith("_weaken"):
        summary_key = f"{target_class}_weaken"
    else:
        summary_key = target_class
    base_metric = (target_class[:-7] if target_class and target_class.endswith("_weaken") else target_class)
    if target_class is not None and class_ids and base_metric not in class_ids and summary_key not in (summary or {}):
        target_class = list(class_ids)[0]
        base_metric = target_class
        summary_key = target_class if increase_target_probabilities else f"{target_class}_weaken"
    if target_class is None:
        raise ValueError("No target_class available. Pass target_class (e.g. '3SG') or ensure class_ids/summary are set.")
    
    # Extract learning rates and probabilities from grid
    lr_data = {}  # {feature_factor: {lr: {dataset_class: {class_name: prob, ...}, ...}, ...}, ...}
    metrics_data = {}  # {feature_factor: {lr: {metric: value, ...}, ...}, ...}
    
    for candidate_id, entry in grid.items():
        if candidate_id == "base":
            continue
        
        # Extract feature_factor and lr
        if isinstance(candidate_id, tuple) and len(candidate_id) == 2:
            feature_factor, lr = candidate_id
        elif isinstance(candidate_id, dict):
            feature_factor = candidate_id.get("feature_factor")
            lr = candidate_id.get("learning_rate")
        else:
            continue
        
        if feature_factor is None or lr is None:
            continue
        
        # Extract probs_by_dataset
        probs_by_dataset = entry.get("probs_by_dataset", {})
        if not probs_by_dataset:
            continue
        
        # Store probabilities by feature_factor -> lr -> dataset_class -> class_name
        if feature_factor not in lr_data:
            lr_data[feature_factor] = {}
        if lr not in lr_data[feature_factor]:
            lr_data[feature_factor][lr] = {}
        
        lr_data[feature_factor][lr] = probs_by_dataset
        
        # Extract metrics (LMS, etc.); be strict about expected keys
        if feature_factor not in metrics_data:
            metrics_data[feature_factor] = {}
        if lr not in metrics_data[feature_factor]:
            metrics_data[feature_factor][lr] = {}
        
        if "lms" in entry:
            lms_val = entry["lms"]
            if isinstance(lms_val, dict):
                if "lms" in lms_val:
                    metrics_data[feature_factor][lr]["lms"] = float(lms_val["lms"])
                elif "perplexity" in lms_val:
                    metrics_data[feature_factor][lr]["lms"] = float(lms_val["perplexity"])
                else:
                    raise KeyError("Decoder grid entry 'lms' dict missing both 'lms' and 'perplexity' keys.")
            else:
                metrics_data[feature_factor][lr]["lms"] = float(lms_val)
    
    if not lr_data:
        raise ValueError("No valid grid data found for plotting. Ensure grid contains entries with probs_by_dataset.")
    
    # Determine dataset classes from grid data
    dataset_classes = set()
    for feature_factor_data in lr_data.values():
        for lr_data_entry in feature_factor_data.values():
            dataset_classes.update(lr_data_entry.keys())
    dataset_classes = sorted(list(dataset_classes))
    
    if not dataset_classes:
        raise ValueError("No dataset classes found in grid data.")
    
    feature_factors = sorted(set(lr_data.keys()))
    
    # Determine which feature_factor to use for each target class (class -> ff that pushes toward that class)
    class_to_feature_factor = {}
    if trainer is not None and hasattr(trainer, "get_model"):
        try:
            model = trainer.get_model()
            if model and hasattr(model, "feature_class_encoding_direction"):
                direction = model.feature_class_encoding_direction
                if direction:
                    for class_name in class_ids:
                        if class_name in direction:
                            class_to_feature_factor[class_name] = -direction[class_name]
        except Exception:
            pass
    if not class_to_feature_factor and feature_factors:
        default_ff = feature_factors[0]
        class_to_feature_factor = {class_name: default_ff for class_name in class_ids}
    
    # Single feature factor for chosen target (consistent across all plots)
    # For _weaken keys, use ff from summary (opposite direction)
    ff = None
    if summary_key and summary_key in (summary or {}):
        ff = summary[summary_key].get("feature_factor")
    if ff is None:
        ff = class_to_feature_factor.get(base_metric, feature_factors[0] if feature_factors else None)
    if ff is None or ff not in lr_data:
        raise ValueError(f"No data for target_class={target_class}. Ensure class_to_feature_factor maps it to an evaluated feature factor.")
    
    # lr=0 (base) x-position — one step left of min lr on log scale
    lrs_ff = sorted(lr_data[ff].keys())
    min_lr = min(lrs_ff) if lrs_ff else 1e-5
    lr0_x = min_lr / 10
    
    # Base model data
    base_entry = grid.get("base", {})
    if not base_entry or "lms" not in base_entry:
        raise KeyError("Decoder grid is missing base entry with 'lms'; cannot plot LMS panel reliably.")
    base_probs_by_dataset = base_entry.get("probs_by_dataset", {})
    lm = base_entry["lms"]
    if isinstance(lm, dict):
        if "lms" in lm:
            base_lms_val = float(lm["lms"])
        elif "perplexity" in lm:
            base_lms_val = float(lm["perplexity"])
        else:
            raise KeyError("Base entry 'lms' dict missing both 'lms' and 'perplexity' keys.")
    else:
        base_lms_val = float(lm)
    
    def _prepend_base(xs, ys, x0, y0):
        """Prepend base point so line connects from base."""
        return [x0] + list(xs), [y0] + list(ys)
    
    # Subplots: 1) LMS, 2+) Dataset probability shifts (selection star on counterfactual or factual line)
    lrs = sorted(lr_data[ff].keys())
    other_classes = [c for c in class_ids if c != base_metric]
    is_weaken = summary_key and summary_key.endswith("_weaken")
    n_subplots = 1 + len(dataset_classes)
    if figsize is None:
        figsize = (8, 2 * n_subplots)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]

    # Plot 1: LMS (one line) — use green to distinguish from probability lines (blue, orange)
    ax_lms = axes[0]
    lms_vals = [metrics_data[ff][lr].get("lms", 0.0) for lr in lrs]
    base_lms = base_lms_val if base_lms_val is not None else (lms_vals[0] if lms_vals else 0.0)
    x_lms, y_lms = _prepend_base(lrs, lms_vals, lr0_x, base_lms)
    ax_lms.plot(x_lms, y_lms, marker="o", label="LMS", alpha=0.7, color="#2ca02c")
    ax_lms.set_ylabel("LMS")
    ax_lms.set_title("LMS (Language Modeling Score)")
    ax_lms.legend(loc="best", fontsize=8)
    ax_lms.set_xscale("log")
    ax_lms.grid(True, alpha=0.3)
    
    # Selection metric:
    # - strengthen (increase_target_probabilities=True):
    #       P(target_class) on the *other* class dataset (e.g. P(3SG) on 3PL)
    # - weaken (increase_target_probabilities=False):
    #       P(target_class) on its own dataset (e.g. P(3SG) on 3SG)
    selection_metric_class = base_metric
    if is_weaken:
        selection_dataset_class = base_metric
    else:
        selection_dataset_class = other_classes[0] if other_classes else base_metric
    selection_metric_label = f"P({selection_metric_class})"

    # Plot 2+: Dataset probability shifts — P(3PL) and P(3SG) on each dataset; highlight selection metric
    for dataset_idx, dataset_class in enumerate(dataset_classes):
        ax = axes[1 + dataset_idx]
        is_selection_dataset = dataset_class == selection_dataset_class
        for class_name in class_ids:
            probs_c = []
            for lr in lrs:
                d = lr_data[ff][lr].get(dataset_class, {})
                prob = d.get(class_name, 0.0) if isinstance(d, dict) else 0.0
                probs_c.append(prob)
            d_base = base_probs_by_dataset.get(dataset_class, {})
            base_p = d_base.get(class_name, 0.0) if isinstance(d_base, dict) else 0.0
            x_p, y_p = _prepend_base(lrs, probs_c, lr0_x, base_p)
            # Emphasize the curve that is the selection metric (used to choose learning rate)
            is_selection_curve = is_selection_dataset and class_name == selection_metric_class
            label = f"{class_name} (target)" if is_selection_curve else class_name
            ax.plot(x_p, y_p, marker="o", label=label, alpha=0.7, linewidth=2.5 if is_selection_curve else 1.5)
        if summary_key in (summary or {}) and is_selection_dataset:
            selected_lr = summary[summary_key].get("learning_rate")
            if selected_lr is not None:
                # Use same ff as plotted curves so star lies exactly on the selection-metric curve
                entry = lr_data[ff].get(selected_lr, {}).get(dataset_class, {})
                sp = entry.get(selection_metric_class, 0.0) if isinstance(entry, dict) else 0.0
                ax.scatter([selected_lr], [sp], marker="*", s=280, zorder=5, alpha=0.95, color="red", label=f"selected (lr={selected_lr:g})")
        ax.set_ylabel("Probability")
        title = f"Dataset: {dataset_class} — P(class)"
        if is_selection_dataset:
            title += f"  [selection metric: {selection_metric_label}]"
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if dataset_idx == len(dataset_classes) - 1:
            ax.set_xlabel("Learning Rate")
    
    # X-axis: clamp to actual data range, use data lrs as ticks
    max_lr = max(lrs) if lrs else lr0_x
    x_min, x_max = lr0_x * 0.5, max_lr * 1.5
    x_ticks = [lr0_x] + list(lrs)
    x_labels = ["base"] + [f"{lr:g}" for lr in lrs]
    for ax in axes:
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    # Vertical line at selected learning rate (all subplots)
    selected_lr = None
    if summary_key in (summary or {}):
        selected_lr = summary[summary_key].get("learning_rate")
    if selected_lr is not None:
        for ax in axes:
            ax.axvline(x=selected_lr, color="gray", linestyle="--", alpha=0.7, zorder=1)
    
    # Shared legend for dataset probability plots
    if len(dataset_classes) > 0:
        handles, labels = axes[-1].get_legend_handles_labels()
        seen = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            ncol=min(len(unique_labels), 5),
            fontsize=8,
        )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for top legend
    
    # Save plot
    output_path = None
    if output is not None:
        output_path = output
    elif trainer is not None and hasattr(trainer, "experiment_dir") and trainer.experiment_dir:
        # Construct output path
        experiment_dir = trainer.experiment_dir
        run_id = getattr(trainer, "run_id", None)
        img_format = kwargs.get("img_format", "pdf")
        
        if run_id:
            output_dir = os.path.join(experiment_dir, run_id)
        else:
            output_dir = experiment_dir
        
        filename = f"decoder_probability_shifts.{img_format}"
        output_path = os.path.join(output_dir, filename)
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150), bbox_inches="tight")
        logger.info(f"Saved probability shifts plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path or ""
