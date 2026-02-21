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
    metrics_to_plot: Optional[List[str]] = None,
    output: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> str:
    """
    Plot decoder probability shifts vs learning rate.

    Creates vertically stacked subplots:
    - Top subplot: Selection metrics (e.g., LMS) vs learning rate with highlighted selected points
    - Bottom subplots: One per dataset class, showing probabilities for all classes vs learning rate

    Args:
        trainer: Trainer instance (used to get model/data if needed)
        decoder_results: Standard decoder evaluation results (with 'summary' and 'grid')
        plotting_data: Extended grid data from analyze_decoder_for_plotting (with probs_by_dataset)
        class_ids: Classes to plot (defaults to all_classes or target_classes from trainer)
        metrics_to_plot: Metrics to show in top subplot (defaults to LMS + target classes)
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
    summary = decoder_results.get("summary", {})
    
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
    
    # Determine metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = ["lms"]
        if trainer is not None:
            target_classes = trainer.target_classes or []
            metrics_to_plot.extend(target_classes)
        else:
            # Infer from summary
            metrics_to_plot.extend([k for k in summary.keys() if k != "lms"])
    
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
        
        # Extract metrics (LMS, etc.)
        if feature_factor not in metrics_data:
            metrics_data[feature_factor] = {}
        if lr not in metrics_data[feature_factor]:
            metrics_data[feature_factor][lr] = {}
        
        if "lms" in entry:
            lms_val = entry["lms"]
            if isinstance(lms_val, dict):
                metrics_data[feature_factor][lr]["lms"] = lms_val.get("lms", lms_val.get("perplexity", 0.0))
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
    
    # Determine number of subplots: 1 for metrics + N for datasets
    n_subplots = 1 + len(dataset_classes)
    
    # Create figure
    if figsize is None:
        figsize = (10, 3 * n_subplots)
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    # Get feature factors (for grouping lines)
    feature_factors = sorted(set(lr_data.keys()))
    
    # Plot 1: Selection metrics
    ax_metrics = axes[0]
    for metric in metrics_to_plot:
        if metric == "lms":
            # Plot LMS for each feature_factor
            for ff_idx, feature_factor in enumerate(feature_factors):
                lrs = sorted(metrics_data[feature_factor].keys())
                lms_values = [metrics_data[feature_factor][lr].get("lms", 0.0) for lr in lrs]
                ax_metrics.plot(lrs, lms_values, marker="o", label=f"LMS (ff={feature_factor:.3f})", alpha=0.7)
        else:
            # Plot probability metric from summary
            if metric in summary:
                selected_lr = summary[metric].get("learning_rate")
                selected_value = summary[metric].get("value")
                if selected_lr is not None and selected_value is not None:
                    # Find corresponding feature_factor
                    for feature_factor in feature_factors:
                        if selected_lr in metrics_data.get(feature_factor, {}):
                            ax_metrics.scatter([selected_lr], [selected_value], 
                                             marker="*", s=200, zorder=5, 
                                             label=f"{metric} (selected)", alpha=0.8)
                            break
    
    ax_metrics.set_ylabel("Metric Value")
    ax_metrics.set_title("Selection Metrics")
    ax_metrics.legend(loc="best", fontsize=8)
    ax_metrics.grid(True, alpha=0.3)
    
    # Plot 2-N: Probability shifts per dataset class
    # Determine which feature_factor to use for each class
    # If trainer has feature_class_encoding_direction, use it to map classes to feature_factors
    class_to_feature_factor = {}
    if trainer is not None and hasattr(trainer, "get_model"):
        model = trainer.get_model()
        if model and hasattr(model, "feature_class_encoding_direction"):
            direction = model.feature_class_encoding_direction
            if direction:
                # Map class to feature_factor: feature_factor = -direction[class]
                for class_name in class_ids:
                    if class_name in direction:
                        class_to_feature_factor[class_name] = -direction[class_name]
    
    # If we couldn't map classes to feature_factors, use first feature_factor for all
    if not class_to_feature_factor and feature_factors:
        default_ff = feature_factors[0]
        class_to_feature_factor = {class_name: default_ff for class_name in class_ids}
    
    for dataset_idx, dataset_class in enumerate(dataset_classes):
        ax = axes[1 + dataset_idx]
        
        # Plot one line per class (token class)
        for class_idx, class_name in enumerate(class_ids):
            # Get feature_factor for this class
            feature_factor = class_to_feature_factor.get(class_name)
            if feature_factor is None:
                # Fallback: use first feature_factor
                feature_factor = feature_factors[0] if feature_factors else None
            
            if feature_factor is None or feature_factor not in lr_data:
                continue
            
            # Collect data for this class across lrs
            lrs = sorted(lr_data[feature_factor].keys())
            probs = []
            for lr in lrs:
                if dataset_class in lr_data[feature_factor][lr]:
                    prob = lr_data[feature_factor][lr][dataset_class].get(class_name, 0.0)
                    probs.append(prob)
                else:
                    probs.append(0.0)
            
            if probs:
                ax.plot(lrs, probs, marker="o", label=class_name, alpha=0.7)
        
        # Highlight selected points from summary
        for metric in metrics_to_plot:
            if metric != "lms" and metric in summary:
                selected_lr = summary[metric].get("learning_rate")
                selected_ff = summary[metric].get("feature_factor")
                if selected_lr is not None:
                    # Find probability for this metric's class on this dataset
                    metric_class = metric  # Assume metric name matches class name
                    # Use selected feature_factor if available, else use class mapping
                    ff_to_use = selected_ff if selected_ff is not None else class_to_feature_factor.get(metric_class)
                    if ff_to_use is None and feature_factors:
                        ff_to_use = feature_factors[0]
                    
                    if (metric_class in class_ids and ff_to_use is not None and 
                        ff_to_use in lr_data and selected_lr in lr_data[ff_to_use] and
                        dataset_class in lr_data[ff_to_use][selected_lr]):
                        selected_prob = lr_data[ff_to_use][selected_lr][dataset_class].get(metric_class, 0.0)
                        ax.scatter([selected_lr], [selected_prob], marker="*", s=200, 
                                 zorder=5, alpha=0.8, color="red", label=f"{metric_class} (selected)" if dataset_idx == 0 else "")
        
        ax.set_ylabel("Probability")
        ax.set_title(f"Dataset: {dataset_class}")
        ax.grid(True, alpha=0.3)
        if dataset_idx == len(dataset_classes) - 1:
            ax.set_xlabel("Learning Rate")
    
    # Shared legend for probability plots (positioned at top)
    if len(dataset_classes) > 0:
        # Get handles and labels from last subplot
        handles, labels = axes[-1].get_legend_handles_labels()
        # Remove duplicates while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(unique_handles, unique_labels, loc="upper center", ncol=min(len(unique_labels), 5), fontsize=8)
    
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
