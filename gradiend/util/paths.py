import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple


def is_under_temp_dir(path: Optional[str]) -> bool:
    """Return True if path is under the system temp directory (do not log such paths)."""
    if not path:
        return False
    try:
        real = os.path.realpath(path)
        tmp = os.path.realpath(tempfile.gettempdir())
        return real == tmp or real.startswith(tmp + os.sep)
    except Exception:
        return False

# Artifact types for default subpaths under experiment_dir
ARTIFACT_MODEL = "model"
ARTIFACT_ENCODED_VALUES = "encoded_values"
ARTIFACT_ENCODER_PLOT = "encoder_plot"
ARTIFACT_CONVERGENCE_PLOT = "convergence_plot"
ARTIFACT_DECODER_ANALYSIS_DIR = "decoder_analysis_dir"
ARTIFACT_DECODER_ANALYSIS_SUMMARY = "decoder_analysis_summary"
ARTIFACT_MODEL_CHANGED = "model_changed"
ARTIFACT_CACHE_GRADIENTS = "cache_gradients"

_DEFAULT_SUBPATHS = {
    ARTIFACT_MODEL: "model",
    ARTIFACT_ENCODED_VALUES: "encoded_values.csv",
    ARTIFACT_ENCODER_PLOT: "encoder_analysis.pdf",
    ARTIFACT_CONVERGENCE_PLOT: "training_convergence.pdf",
    ARTIFACT_DECODER_ANALYSIS_DIR: "decoder_analysis",
    ARTIFACT_DECODER_ANALYSIS_SUMMARY: "decoder_analysis.json",
    ARTIFACT_MODEL_CHANGED: None,  # requires target_class
    ARTIFACT_CACHE_GRADIENTS: "cache/gradients",
}


def _create_readable_key(components: Dict[str, Any]) -> str:
    """Readable cache key from key-value components (for keyed filenames)."""
    parts = []
    for key, value in sorted(components.items()):
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value_str = "_".join(str(v) for v in value)
        elif isinstance(value, dict):
            value_str = hashlib.md5(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]
        else:
            value_str = str(value)
        value_str = value_str.replace("/", "_").replace("\\", "_").replace(":", "_")
        parts.append(f"{key}_{value_str}")
    return "_".join(parts)


def create_readable_key(components: Dict[str, Any]) -> str:
    """Public wrapper for readable cache keys."""
    return _create_readable_key(components)


def _resolve_keyed_artifact_path(
    experiment_dir: Optional[str],
    explicit_path: Optional[str],
    *,
    base_name: str,
    suffix: str,
    **key_kwargs: Any,
) -> Optional[str]:
    if explicit_path is not None and str(explicit_path).strip():
        return os.path.normpath(str(explicit_path).strip())
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    base = os.path.normpath(str(experiment_dir).strip())
    if key_kwargs:
        key = _create_readable_key(key_kwargs)
        return os.path.join(base, f"{base_name}_{key}{suffix}")
    return os.path.join(base, f"{base_name}{suffix}")



def resolve_encoder_analysis_path(
    experiment_dir: Optional[str],
    explicit_path: Optional[str] = None,
    **key_kwargs: Any,
) -> Optional[str]:
    """Path for encoder analysis CSV: experiment_dir/encoded_values.csv or encoded_values_{key}.csv. None if no experiment_dir."""
    return _resolve_keyed_artifact_path(
        experiment_dir,
        explicit_path,
        base_name="encoded_values",
        suffix=".csv",
        **key_kwargs,
    )


def resolve_encoder_plot_path(
    experiment_dir: Optional[str],
    explicit_path: Optional[str] = None,
    **key_kwargs: Any,
) -> Optional[str]:
    """Path for encoder analysis PDF: experiment_dir/encoder_analysis.pdf or encoder_analysis_{key}.pdf. None if no experiment_dir."""
    return _resolve_keyed_artifact_path(
        experiment_dir,
        explicit_path,
        base_name="encoder_analysis",
        suffix=".pdf",
        **key_kwargs,
    )


def resolve_encoder_eval_result_path(
    experiment_dir: Optional[str],
    **key_kwargs: Any,
) -> Optional[str]:
    """Path for encoder evaluation result JSON (correlation, mean_by_class, etc.). None if no experiment_dir."""
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    base = os.path.normpath(str(experiment_dir).strip())
    if key_kwargs:
        key = _create_readable_key(key_kwargs)
        return os.path.join(base, f"encoder_eval_result_{key}.json")
    return os.path.join(base, "encoder_eval_result.json")


def resolve_decoder_stats_path(
    experiment_dir: Optional[str],
    *,
    feature_factors: Any = None,
    lrs: Any = None,
    topk: Any = None,
    part: Any = None,
    topk_part: Any = None,
    metric_name: Any = None,
) -> Optional[str]:
    """Path for decoder stats JSON: experiment_dir/decoder_stats_{key}.json. None if no experiment_dir."""
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    base = os.path.normpath(str(experiment_dir).strip())
    components = {
        "topk": topk,
        "part": part,
        "topk_part": topk_part,
        "feature_factors": list(feature_factors) if feature_factors is not None else None,
        "lrs": list(lrs) if lrs is not None else None,
        "metric_name": metric_name,
    }
    key = _create_readable_key(components)
    return os.path.join(base, f"decoder_stats_{key}.json")


def resolve_decoder_per_model_cache_path(
    experiment_dir: Optional[str],
    cache_folder: str = "",
) -> Optional[str]:
    """Path for decoder eval cache JSON: experiment_dir/decoder/{cache_folder}.json. None if no experiment_dir."""
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    base = os.path.normpath(str(experiment_dir).strip())
    decoder_dir = os.path.join(base, "decoder")
    folder = (cache_folder or "base").strip("/\\").replace("/", "_").replace("\\", "_")
    return os.path.join(decoder_dir, f"{folder}.json")



def resolve_decoder_grid_cache_path(
    experiment_dir: Optional[str],
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """
    Single decoder grid cache file path (decoder_grid_cache.json).

    Stores the full grid plus the requested feature_factors/lrs so cache
    validity can be checked from file content rather than filename.
    """
    if explicit_path is not None and str(explicit_path).strip():
        return os.path.normpath(str(explicit_path).strip())
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    return os.path.join(os.path.normpath(str(experiment_dir).strip()), "decoder_grid_cache.json")


def resolve_decoder_mlm_head_dir(
    experiment_dir: Optional[str],
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """Directory for decoder-only MLM head: experiment_dir/decoder_mlm_head. None if no experiment_dir."""
    if explicit_path is not None and str(explicit_path).strip():
        return os.path.normpath(str(explicit_path).strip())
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    return os.path.join(os.path.normpath(str(experiment_dir).strip()), "decoder_mlm_head")


def resolve_classification_head_dir(
    experiment_dir: Optional[str],
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """Directory for classification head: experiment_dir/classification_head. None if no experiment_dir."""
    if explicit_path is not None and str(explicit_path).strip():
        return os.path.normpath(str(explicit_path).strip())
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    return os.path.join(os.path.normpath(str(experiment_dir).strip()), "classification_head")


def resolve_custom_prediction_head_dir(
    experiment_dir: Optional[str],
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """Directory for custom prediction head: experiment_dir/custom_prediction_head. None if no experiment_dir."""
    if explicit_path is not None and str(explicit_path).strip():
        return os.path.normpath(str(explicit_path).strip())
    if experiment_dir is None or not str(experiment_dir).strip():
        return None
    return os.path.join(os.path.normpath(str(experiment_dir).strip()), "custom_prediction_head")


def invalidate_experiment_caches(experiment_dir: Optional[str]) -> None:
    """Remove known cache files/dirs under experiment_dir. No-op if experiment_dir is None."""
    if experiment_dir is None or not str(experiment_dir).strip():
        return
    base = os.path.normpath(str(experiment_dir).strip())
    if not os.path.isdir(base):
        return
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if name.startswith("encoded_values") and (name.endswith(".csv") or name.endswith(".json")):
            try:
                os.remove(path)
            except OSError:
                pass
        elif name.startswith("decoder_grid_") and name.endswith(".json"):
            try:
                os.remove(path)
            except OSError:
                pass
        elif name == "decoder_grid_cache.json":
            try:
                os.remove(path)
            except OSError:
                pass
        elif name.startswith("decoder_stats_") and name.endswith(".json"):
            try:
                os.remove(path)
            except OSError:
                pass
        elif name == "decoder" and os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except OSError:
                pass


def resolve_output_path(
    experiment_dir: Optional[str],
    explicit_path: Optional[str],
    artifact_type: str,
    **extra: Any,
) -> Optional[str]:
    """
    Resolve the path to use for saving/loading an artifact.

    If explicit_path is provided, it overrides the experiment_dir default.
    Otherwise, if experiment_dir is set, returns the default subpath for that artifact.
    Otherwise returns None (caller must require explicit path or not save).

    Args:
        experiment_dir: Root directory for the experiment (e.g. from TrainingArguments.experiment_dir).
        explicit_path: User-provided path for this artifact (overrides experiment_dir).
        artifact_type: One of ARTIFACT_* constants.
        **extra: For ARTIFACT_MODEL_CHANGED pass target_class=str. For ARTIFACT_MODEL the default
            is experiment_dir/model (one model per experiment dir). For ARTIFACT_ENCODER_PLOT
            pass run_id and model_name for {run_id}_{name}_encoder_distributions.pdf.

    Returns:
        Path to use, or None if neither experiment_dir nor explicit_path is set.
    """
    if explicit_path is not None and str(explicit_path).strip():
        return os.path.normpath(str(explicit_path).strip())

    if experiment_dir is None or not str(experiment_dir).strip():
        return None

    base = os.path.normpath(str(experiment_dir).strip())
    if artifact_type == ARTIFACT_MODEL_CHANGED:
        target_class = extra.get("target_class")
        if target_class is None:
            return None
        return os.path.join(base, f"{target_class}")
    # One experiment dir = one model; model lives at experiment_dir/model (no models/id/name)
    if artifact_type == ARTIFACT_ENCODER_PLOT and "run_id" in extra and "model_name" in extra:
        rid = extra.get("run_id") or "run"
        filename = f"{rid}_{extra['model_name']}_encoder_distributions.pdf"
        return os.path.join(base, filename)
    subpath = _DEFAULT_SUBPATHS.get(artifact_type)
    if subpath is None:
        return None
    return os.path.join(base, subpath)


def has_saved_model(output_dir: str) -> bool:
    """
    Return True if output_dir contains a complete saved GRADIEND model (skip training when use_cache).

    Requires:
    - Weights: model.safetensors or pytorch_model.bin (as in BaseModel.save_pretrained)
    - config.json (always written by save_pretrained; ensures it's our format)

    Note: training_args.json is not used; the codebase writes training.json (optional) or embeds
    run info in the save_pretrained(training=...) path.
    """
    if not output_dir or not os.path.exists(output_dir):
        return False
    has_weights = (
        os.path.exists(os.path.join(output_dir, "model.safetensors"))
        or os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
    )
    has_config = os.path.exists(os.path.join(output_dir, "config.json"))
    return has_weights and has_config


def should_use_cached(path: Optional[str], use_cache: bool) -> bool:
    """
    Return whether we can skip computation and use cached output.

    True only when path is set, use_cache is True, and path exists (file or dir).

    Args:
        path: Resolved output path (from resolve_output_path).
        use_cache: Whether to use cache when path exists.

    Returns:
        True if path exists and use_cache is True; False otherwise.
    """
    if path is None or not use_cache:
        return False
    return os.path.exists(path)


def require_output_path(
    experiment_dir: Optional[str],
    explicit_path: Optional[str],
    artifact_type: str,
    **extra: Any,
) -> str:
    """
    Like resolve_output_path but raises ValueError if result would be None.

    Use for operations that must have a path (e.g. training).
    """
    path = resolve_output_path(experiment_dir, explicit_path, artifact_type, **extra)
    if path is None:
        raise ValueError(
            f"Output path required for {artifact_type}. "
            "Set experiment_dir on TrainingArguments or pass output_dir= explicitly."
        )
    return path