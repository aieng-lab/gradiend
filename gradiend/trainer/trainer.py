"""
Trainer: HF-like API with model at creation time and lazy Evaluator.

Trainer subclasses FeatureLearningDefinition and adds: model at construction, training_arguments (optional),
get_model(), evaluator_class with lazy Evaluator. Training logic lives in _train();
override _train() in subclasses to customize.
"""

import gc
import os
import tempfile
import shutil
import random
import time
import numpy as np
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union, Tuple

import pandas as pd
import torch

from torch.utils.data import DataLoader

from gradiend import ModelWithGradiend
from gradiend.model._source_target import sync_model_source_target_from_training_args
from gradiend.util.paths import resolve_encoder_plot_path
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition, _resolve_encoder_df
from gradiend.util.paths import (
    resolve_output_path,
    require_output_path,
    invalidate_experiment_caches,
    has_saved_model,
    is_under_temp_dir,
    ARTIFACT_MODEL, ARTIFACT_MODEL_CHANGED,
    resolve_decoder_stats_path,
    resolve_pre_prune_cache_dir,
    remove_pre_prune_cache,
)
from gradiend.evaluator.decoder_eval_utils import read_decoder_stats_file
from gradiend.util.logging import get_logger
from gradiend.trainer.core.dataset import PreComputedTrainingDataset
from gradiend.trainer.core.annotation import TrainerAnnotationMixin
from gradiend.trainer.core.training import format_non_convergence_error, train as core_train
from gradiend.trainer.factory import create_model_with_gradiend
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.config import validate_source_target
from gradiend.trainer.core.cache_policy import (
    should_reuse_seed_training_cache,
    should_reuse_training_cache,
)
from gradiend.trainer.core.multi_seed import MultiSeedTrainerView, resolve_seed_run_entries
from gradiend.evaluator import Evaluator
from gradiend.visualizer.plot_delegation import see_implementation
from gradiend.trainer.core.pruning import post_prune as _post_prune, pre_prune_with_cache
import gradiend.trainer.core.stats as trainer_stats
from gradiend.util.encoder_splits import EncoderSplit, encoder_split_cache_key
from gradiend.util.runtime_monitor import RuntimeMonitor, is_cuda_oom_error, write_cuda_oom_log

logger = get_logger(__name__)

_ANALYZE_SEED_STABILITY_MIN_ONE_WARNING = (
    "analyze_seed_stability=True with min_convergent_seeds=1: only one convergent seed will be "
    "available for stability analysis, so variance statistics (std, dispersion) are not meaningful. "
    "Set min_convergent_seeds>=2 to study seed stability across multiple convergent runs."
)


def set_seed(seed: int) -> None:
    """
    Set all RNG seeds and env vars for reproducible runs. Call at the very start of your
    script (before creating the trainer or loading data) for best reproducibility.
    The Trainer also calls this when TrainingArguments.seed is set.

    Sets: Python random, numpy, PyTorch (CPU + CUDA), cuDNN deterministic, and
    CUBLAS_WORKSPACE_CONFIG / OMP_NUM_THREADS / MKL_NUM_THREADS.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    _apply_seed(seed)


def _apply_seed(seed_value: int) -> None:
    """Set global RNG seeds and CUDA determinism for reproducible runs. Used by Trainer when TrainingArguments.seed is set."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("Failed to remove temporary JSON file %s", tmp_path, exc_info=True)


def _monitor_path(monitor: Any) -> Optional[str]:
    return getattr(monitor, "path", None) if monitor is not None else None


class Trainer(TrainerAnnotationMixin, FeatureLearningDefinition):
    """
    Abstract base trainer for GRADIEND models with HuggingFace-like API.
    
    The Trainer class is an abstract base class that provides the main interface for training,
    evaluating, and working with GRADIEND models. It cannot be instantiated directly; you must
    use a concrete subclass such as `TextPredictionTrainer` that implements the required
    abstract methods from `FeatureLearningDefinition`.
    
    **Abstract Methods:**
    
    Subclasses must implement the following abstract methods from `FeatureLearningDefinition`:

    - `create_training_data()`: Create training dataset without gradient computation
    - `create_gradient_training_dataset()`: Create training dataset with gradient computation
    - `_get_decoder_eval_dataframe()`: Get DataFrames for decoder evaluation
    - `_get_decoder_eval_targets()`: Get target tokens for decoder evaluation
    - `evaluate_base_model()`: Evaluate a single model for decoder evaluation
    - `_analyze_encoder()`: Analyze encoder by encoding gradients from training data
    
    **Class Management:**
    
    The trainer uses `target_classes` to specify which classes are used for training.
    The `pair` property is automatically inferred from `target_classes` when exactly two
    target classes are specified (i.e., `pair = (target_classes[0], target_classes[1])`).
    The `all_classes` property includes all classes in the dataset (including non-target
    classes) and can be inferred from data if not explicitly set.
    
    - **target_classes**: Classes used for training (required)
    - **all_classes**: All classes in dataset (optional, inferred from data if not set)
    - **pair**: Automatically computed from target_classes when len(target_classes) == 2
    
    **Key Features:**
    
    - **Model Management**: Stores model at construction time with lazy loading and caching
    - **Training**: Full training pipeline with support for pre-pruning, training, and post-pruning
    - **Multi-Seed Training**: Automatic multi-seed training with convergence tracking and best seed selection
    - **Evaluation**: Integrated encoder and decoder evaluation with caching
    - **Visualization**: Delegates plotting to Evaluator/Visualizer
    - **Device Management**: Easy model device movement (CPU/CUDA)
    
    **Basic Usage:**
    
    ```python
    from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
    from gradiend.trainer.core.arguments import TrainingArguments
    import pandas as pd
    
    # Initialize trainer with model and training arguments
    # Note: Use TextPredictionTrainer (or another concrete subclass), not the abstract Trainer directly
    args = TrainingArguments(
        experiment_dir="./results",
        train_batch_size=32,
        num_epochs=10,
        learning_rate=1e-3,
    )
    trainer = TextPredictionTrainer(
        model="gpt2",
        args=args,
        run_id="runs/experiment_gpt2",
        data=your_dataframe,  # Modality-specific data (could also be a HF dataset id)
        target_classes=["class1", "class2"],  # Target classes the GRADIEND -> these gets encoded as +-1
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate encoder and decoder
    enc_results = trainer.evaluate_encoder()
    dec_results = trainer.evaluate_decoder()
    
    # Plot results
    trainer.plot_encoder_distributions()
    trainer.plot_training_convergence()
    ```
    
    **Multi-Seed Training:**
    
    When `TrainingArguments.max_seeds > 1`, the trainer automatically runs multiple training
    runs with different random seeds. It tracks convergence metrics, selects the best seed based
    on selection scores, and writes a comprehensive seed report.
    
    ```python
    args = TrainingArguments(
        max_seeds=5,
        min_convergent_seeds=2,
        convergent_metric="correlation",
        convergent_score_threshold=0.5,
    )
    trainer = Trainer(model="gpt2", args=args)
    trainer.train()  # Runs 5 seeds, selects best
    ```
    
    **Pruning:**
    
    The trainer supports both pre-pruning (before training) and post-pruning (after training):
    
    ```python
    from gradiend.trainer.core.pruning import PrePruneConfig, PostPruneConfig
    
    # Pre-prune: gradient-based pruning before training
    pre_cfg = PrePruneConfig(n_samples=1000, topk=0.5, source="diff")
    args.pre_prune_config = pre_cfg
    
    # Post-prune: weight-based pruning after training
    post_cfg = PostPruneConfig(topk=0.3, part="decoder-weight")
    args.post_prune_config = post_cfg
    
    trainer.train()  # Automatically applies pre-prune and post-prune
    ```
    
    **Evaluation:**
    
    The trainer provides convenient methods for encoder and decoder evaluation:
    
    ```python
    # Encoder evaluation: analyze gradient encodings
    enc_results = trainer.evaluate_encoder(split="test", max_size=1000)
    # Returns: correlation, encoded values, mean_by_class, etc.
    
    # Decoder evaluation: grid search over feature_factor and learning_rate
    dec_results = trainer.evaluate_decoder(use_cache=True)
    # Returns: summary (best configs per metric) and grid (all results)
    
    # Combined evaluation
    results = trainer.evaluate()
    # Returns: {"encoder": enc_results, "decoder": dec_results}
    ```
    
    **Model Access:**
    
    ```python
    # Get the trained model (cached after first load)
    model = trainer.get_model()
    
    # Load a specific checkpoint
    model = trainer.get_model(load_directory="./results/model")
    
    # Move model to device
    trainer.cuda(device=0)  # or trainer.cpu()
    ```
    
    **Architecture:**
    
    The Trainer subclasses `FeatureLearningDefinition` and adds:

    - Model storage and lazy loading (`get_model()`)
    - Training arguments management (`training_args` property)
    - Lazy Evaluator initialization (`evaluator` property)
    - Experiment directory resolution (`experiment_dir` property)
    
    Training logic lives in `_train()`; subclasses can override this method to customize behavior.
    
    **Args:**
        model: Model identifier (string path) or ModelWithGradiend instance. If string,
            the model is loaded lazily on first access via `get_model()`.
        target_classes: Optional list of target class names for training. If None, subclasses
            can determine target_classes from data by setting self._target_classes during initialization
            or data loading. Default: None
        args: Optional TrainingArguments instance. Can also be passed as kwargs to `train()`.
        run_id: Optional run identifier. When set, creates subdirectory under experiment_dir.
        n_features: Number of latent features (default: 1).
        evaluator_class: Optional Evaluator class. Defaults to `Evaluator`.
        **kwargs: Additional attributes to set on the trainer instance.
    
    **Attributes:**
        training_args: TrainingArguments instance (if provided).
        experiment_dir: Resolved experiment directory (experiment_dir/run_id if run_id set).
        model_path: Current model path (initial model or path after training).
        evaluator: Lazy-initialized Evaluator instance.
    
    **Methods:**
        train(): Train GRADIEND model with optional pre/post-pruning.
        evaluate_encoder(): Analyze encoder performance (correlation, encodings).
        evaluate_decoder(): Grid search decoder configurations.
        evaluate(): Run both encoder and decoder evaluation.
        get_model(): Get the trainer's ModelWithGradiend instance (cached).
        load_model(): Load a ModelWithGradiend instance from a specific directory.
        pre_prune(): Run pre-pruning before training.
        post_prune(): Run post-pruning after training.
        plot_encoder_distributions(): Plot encoder distribution visualizations.
        plot_training_convergence(): Plot training convergence metrics.
        rewrite_base_model(): Rewrite base model(s) using decoder evaluation results, optionally save to disk.
    
    **See Also:**

        - `FeatureLearningDefinition`: Abstract base class providing data creation and evaluation protocols
        - `TextPredictionTrainer`: Concrete implementation for text-based models (MLM/CLM)
        - `TrainingArguments`: Configuration for training behavior
        - `Evaluator`: Evaluation and visualization orchestration
        - `PrePruneConfig`, `PostPruneConfig`: Pruning configuration
    
    **Note:**
        This class is abstract and cannot be instantiated directly. Use a concrete subclass
        such as `TextPredictionTrainer` that implements the required abstract methods.
    """

    def __init__(
        self,
        model: Union[str, Any],
        target_classes: Optional[List[str]] = None,
        args: Optional[Any] = None,
        run_id: Optional[str] = None,
        n_features: int = 1,
        evaluator_class: Optional[Type] = None,
    ):
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError(f"run_id must be str or None, got {type(run_id).__name__}")
        if not isinstance(n_features, int):
            raise TypeError(f"n_features must be int, got {type(n_features).__name__}")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")

        super().__init__(target_classes, run_id, n_features)
        self._model_arg = model
        self._base_model_arg = model  # Original model; never changes (model_path does after train)
        self._model_instance: Optional[Any] = None
        self._model_manually_unloaded: bool = False
        self._training_args = args
        self._evaluator_class = evaluator_class if evaluator_class is not None else Evaluator
        self._evaluator: Optional[Any] = None
        self._model_with_gradiend_cls: Optional[Type[Any]] = None

    @property
    def training_args(self) -> Optional[TrainingArguments]:
        return self._training_args

    def _write_cuda_oom_log(
        self,
        phase: str,
        exc: BaseException,
        *,
        root: Optional[str] = None,
        monitor: Any = None,
    ) -> Optional[str]:
        root = root or self.experiment_dir or getattr(self.training_args, "output_dir", None)
        path = write_cuda_oom_log(root, phase=phase, exc=exc, monitor_path=_monitor_path(monitor))
        if path:
            logger.error("CUDA OOM during %s; wrote diagnostic log to %s", phase, path)
        return path

    def _resolve_experiment_dir(self, base: Optional[str]) -> Optional[str]:
        """Join training_args.experiment_dir with run_id unless run_id is already the leaf name."""
        if base is None or not str(base).strip():
            return None
        exp = str(base).strip().rstrip("/\\")
        if not self.run_id or not str(self.run_id).strip():
            return exp
        rid = str(self.run_id).strip().strip("/\\")
        if os.path.basename(exp) == rid:
            return exp
        return os.path.join(exp, rid)

    def _experiment_dir(self) -> Optional[str]:
        """Root directory for this experiment (experiment_dir, or experiment_dir/run_id when run_id is set)."""
        if self.training_args is None:
            return None
        return self._resolve_experiment_dir(self.training_args.experiment_dir)

    @property
    def experiment_dir(self) -> Optional[str]:
        """
        Experiment directory for this trainer.

        If training_args.experiment_dir is set, returns that (with run_id subdir if run_id is set).

        """
        return self._experiment_dir()

    def __str__(self) -> str:
        return (
            f"Trainer(model={self._model_arg!r}, run_id={self.run_id!r}, "
            f"pair={self.pair}, target_classes={self._target_classes}, "
            f"experiment_dir={self.experiment_dir!r})"
        )

    def to(self, device: Any) -> "Trainer":
        """Move loaded model to the given device (str or torch.device). No-op if model not loaded."""
        model = self.get_model()
        if model is not None and hasattr(model, "to"):
            model.to(device)
        return self

    def cpu(self) -> "Trainer":
        """Move loaded model to CPU. No-op if model not loaded."""
        return self.to("cpu")

    def cuda(self, device: Any = None) -> "Trainer":
        """Move loaded model to CUDA. device: None (default cuda), int (cuda:N), or str/torch.device."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        return self.to(device)

    @property
    def base_model_path(self) -> str:
        """Original model passed at construction (base model id or path, e.g. 'bert-base-cased')."""
        arg = self._base_model_arg
        if isinstance(arg, str):
            return arg.rstrip("/\\")
        path = getattr(arg, "name_or_path", None) if arg is not None else None
        if path is None:
            raise ValueError("Could not determine base_model_path from _base_model_arg")
        return str(path).rstrip("/\\")

    @property
    def model_path(self) -> str:
        """Current model path: base model before training, GRADIEND output dir after train()."""
        if isinstance(self._model_arg, str):
            # Resolve to custom prediction head (e.g. decoder MLM head) when it exists
            path = self.resolve_model_path(self._model_arg)
        else:
            path = getattr(self._model_arg, "name_or_path", None) if self._model_arg is not None else None
            if path is None:
                path = "model"
        return path.rstrip("/\\")

    def load_model(
        self,
        load_directory: str,
        model_with_gradiend_cls: Optional[Type[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Load a ModelWithGradiend instance from a specific directory.

        This method loads a model from a different checkpoint/directory than the trainer's
        current model_path. Use this for loading different checkpoints (e.g., for comparison)
        or loading specific seed runs in multi-seed training.

        Args:
            load_directory: Directory or path to load the model from.
            model_with_gradiend_cls: Optional ModelWithGradiend subclass. If None, uses
                self.model_with_gradiend_cls or self.default_model_with_gradiend_cls.
            **kwargs: Passed to model_with_gradiend_cls.from_pretrained (e.g. feature_definition=self for text models).

        Returns:
            ModelWithGradiend instance loaded from the specified directory.
        """
        if not isinstance(load_directory, str):
            raise TypeError(f"load_directory must be str, got {type(load_directory).__name__}")

        # Pass feature_definition so text models get pair/classes when loading
        kwargs.setdefault("feature_definition", self)
        kwargs.setdefault("require_gradiend_model", True)  # load_model expects GRADIEND checkpoint
        if self._training_args is not None:
            kwargs.setdefault("training_args", self._training_args)
            if "trust_remote_code" not in kwargs:
                kwargs.setdefault("trust_remote_code", getattr(self._training_args, "trust_remote_code", False))
        return super().create_model_with_gradiend(load_directory, model_with_gradiend_cls=model_with_gradiend_cls, **kwargs)

    def get_model(
        self,
        use_cache: Optional[bool] = None,
        *,
        load_directory: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Get the trainer's ModelWithGradiend instance.

        Returns the in-memory model when set (e.g. during training), otherwise loads from
        load_directory or model_path. The loaded instance is always cached in memory for
        subsequent calls. After multi-seed training the cache is cleared so the next
        get_model() loads from the selected best-seed directory and then caches that instance.

        To load a model from a different directory, use load_model() or pass load_directory=.

        Note: use_cache (e.g. TrainingArguments.use_cache) applies only to disk/output
        caches (skip when files exist); the in-memory model from get_model() is always cached.

        Args:
            use_cache: Ignored; kept for API compatibility. Disk cache is controlled elsewhere.
            load_directory: If provided, load from this path (GRADIEND checkpoint expected).
            **kwargs: Passed to model_with_gradiend_cls.from_pretrained.

        Returns:
            ModelWithGradiend instance.
        """
        load_directory = load_directory if load_directory is not None else kwargs.pop("load_directory", None)
        # Return in-memory model when set (e.g. during training), unless loading from a specific path
        if load_directory is None and self._model_instance is not None:
            sync_model_source_target_from_training_args(self._model_instance, self._training_args)
            return self._model_instance
        # Pass definition so models get pair/classes when loading
        kwargs.setdefault("definition", self)
        if self._training_args is not None:
            kwargs.setdefault("training_args", self._training_args)
            if "trust_remote_code" not in kwargs:
                kwargs.setdefault("trust_remote_code", getattr(self._training_args, "trust_remote_code", False))
        load_directory = load_directory if load_directory is not None else self.model_path
        model = super().create_model_with_gradiend(load_directory, **kwargs)
        sync_model_source_target_from_training_args(model, self._training_args)
        # Always cache in memory; use_cache elsewhere is for disk/output only
        self._model_instance = model
        self._model_manually_unloaded = False
        return model

    def unload_model(self) -> None:
        """Release the in-memory model reference and try to move the model off GPU.

        After calling this, call :meth:`get_model` or :meth:`load_model` (and optionally
        :meth:`cuda`) yourself before evaluation if you want explicit control over loading
        and device placement. Evaluation methods will still reload from disk when needed
        but emit a warning.
        """
        model = self._model_instance
        self._model_instance = None
        self._model_manually_unloaded = True
        if model is not None and hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception:
                pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_explicit_cpu_device(device: Any) -> bool:
        if device is None:
            return False
        dev = torch.device(device) if isinstance(device, str) else device
        return getattr(dev, "type", None) == "cpu"

    def _model_primary_device(self, model: Any) -> Optional[torch.device]:
        if model is None:
            return None
        getter = getattr(model, "_get_base_forward_device", None)
        if callable(getter):
            try:
                return getter()
            except (StopIteration, RuntimeError):
                pass
        gradiend = getattr(model, "gradiend", None)
        if gradiend is not None:
            enc_dev = getattr(gradiend, "device_encoder", None)
            if enc_dev is not None:
                return enc_dev if isinstance(enc_dev, torch.device) else torch.device(enc_dev)
        return None

    def _warn_if_reloading_after_unload(self) -> None:
        if self._model_instance is not None:
            return
        if not self._model_manually_unloaded:
            return
        if getattr(self, "_suppress_expected_reload_warning", False):
            return
        logger.warning(
            "The in-memory model was released via unload_model(). Evaluation will reload the "
            "checkpoint from %s (typically on CUDA when available). Call trainer.get_model() "
            "and trainer.cuda() yourself beforehand if you want explicit control over loading "
            "and device placement.",
            self.model_path,
        )

    def _warn_if_eval_on_cpu_unexpected(
        self,
        model: Any,
        *,
        device: Any = None,
        context: str = "Evaluation",
    ) -> None:
        if self._is_explicit_cpu_device(device):
            return
        if not torch.cuda.is_available():
            return
        dev = self._model_primary_device(model)
        if dev is not None and dev.type == "cpu":
            logger.warning(
                "%s is running with the model on CPU while CUDA is available. GPU evaluation is "
                "recommended for speed. Call trainer.cuda() (or pass an explicit device to "
                "evaluate_encoder/evaluate_decoder) before evaluating, or pass device='cpu' to "
                "acknowledge CPU evaluation.",
                context,
            )

    def _apply_explicit_eval_device(self, model: Any, *, device: Any) -> Any:
        if device is None or model is None:
            return model
        if hasattr(model, "place_for_evaluation"):
            args = self.training_args or TrainingArguments()
            model.place_for_evaluation(
                device=device,
                encoder_decoder_same_device=getattr(args, "encoder_decoder_same_device", False),
            )
        elif self._is_explicit_cpu_device(device):
            model.cpu()
        elif hasattr(model, "to"):
            model.to(device)
        return model

    def _prepare_model_for_evaluation(
        self,
        model_with_gradiend: Optional[Any] = None,
        *,
        device: Any = None,
        context: str = "Encoder evaluation",
    ) -> Any:
        self._warn_if_reloading_after_unload()
        model = model_with_gradiend if model_with_gradiend is not None else self.get_model()
        model = self._apply_explicit_eval_device(model, device=device)
        self._warn_if_eval_on_cpu_unexpected(model, device=device, context=context)
        return model

    def _prepare_model_for_pre_prune_if_needed(
        self,
        model_with_gradiend: Any,
        training_args: Any,
    ) -> Any:
        """
        Ensure the model uses lazy_init when pre_prune_config is set.

        get_model() may return a cached eager-init instance (e.g. from an earlier call
        before pre_prune_config was set). Recreate from the base model path so encoder/
        decoder weights are deferred until after pre-prune.
        """
        pre_cfg = getattr(training_args, "pre_prune_config", None)
        if pre_cfg is None:
            return model_with_gradiend
        gradiend = getattr(model_with_gradiend, "gradiend", None)
        if gradiend is not None and getattr(gradiend, "_lazy_init", False):
            return model_with_gradiend
        getattr(self, "_ensure_data_for_training", lambda: None)()
        model_cls = self.model_with_gradiend_cls or self.default_model_with_gradiend_cls
        if model_cls is None:
            return model_with_gradiend
        load_path = self.resolve_model_path(self.base_model_path)
        base_model = getattr(model_with_gradiend, "base_model", None)
        create_kwargs: Dict[str, Any] = {
            "feature_definition": self,
            "training_args": training_args,
            "trust_remote_code": getattr(training_args, "trust_remote_code", False),
        }
        if base_model is not None:
            create_kwargs["base_model"] = base_model
        return create_model_with_gradiend(
            load_path,
            model_class=model_cls,
            **create_kwargs,
        )

    def get_seed_report_path(self) -> Optional[str]:
        candidates: List[str] = []
        exp = self.experiment_dir
        if exp and str(exp).strip():
            candidates.append(os.path.join(str(exp), "seeds", "seed_report.json"))
        args = self._training_args
        if args is not None and getattr(args, "seed_runs_dir", None):
            seed_runs_dir = str(args.seed_runs_dir).strip()
            if seed_runs_dir:
                candidates.append(os.path.join(seed_runs_dir, "seed_report.json"))
        model_path = getattr(self, "model_path", None)
        if model_path and str(model_path).strip():
            candidates.append(os.path.join(str(model_path), "seeds", "seed_report.json"))
        seen: set = set()
        for candidate in candidates:
            norm = os.path.normcase(os.path.normpath(candidate))
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.isfile(candidate):
                return candidate
        return None

    def get_seed_report(self) -> Optional[Dict[str, Any]]:
        report_path = self.get_seed_report_path()
        if not report_path:
            return None
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load seed report from %s: %s", report_path, exc)
            return None
        return payload if isinstance(payload, dict) else None

    def get_saved_seed_run_paths(self, selection: str = "all_convergent") -> List[str]:
        """Return saved model directories for selected seed runs.

        Args:
            selection: Which seed runs to return: ``"best"``,
                ``"all_convergent"``, or ``"all_tried"``.

        Returns:
            Existing seed-run output directories. Falls back to the current
            ``model_path`` when no seed report is available.
        """
        selection = str(selection).strip().lower()
        if selection == "best":
            return [str(self.model_path)]
        if selection not in {"all_convergent", "all_tried"}:
            raise ValueError("selection must be 'best', 'all_convergent', or 'all_tried'")
        report = self.get_seed_report()
        if not report:
            return [str(self.model_path)]
        runs = report.get("runs", [])
        if not isinstance(runs, list):
            return [str(self.model_path)]
        paths: List[str] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            output_dir = run.get("output_dir")
            if not isinstance(output_dir, str) or not output_dir:
                continue
            if selection == "all_convergent" and not bool(run.get("converged")):
                continue
            if os.path.isdir(output_dir):
                paths.append(output_dir)
        return paths or [str(self.model_path)]

    def get_best_seed_run_path(self) -> Optional[str]:
        report = self.get_seed_report()
        if not report:
            return None
        best_seed = report.get("best_seed")
        runs = report.get("runs", [])
        if not isinstance(best_seed, int) or not isinstance(runs, list):
            return None
        for run in runs:
            if not isinstance(run, dict):
                continue
            if int(run.get("seed", -1)) != best_seed:
                continue
            output_dir = run.get("output_dir")
            if isinstance(output_dir, str) and output_dir and os.path.isdir(output_dir):
                return output_dir
        return None

    def get_seed_run_entries(self, selection: str = "all_convergent") -> List[Tuple[int, str]]:
        """Return ``(seed, output_dir)`` pairs for multi-seed analysis.

        Args:
            selection: Which seed runs to include: ``"best"``,
                ``"all_convergent"``, or ``"all_tried"``.
        """
        return resolve_seed_run_entries(self, selection)

    def iter_encoder_eval_cache_dirs(self) -> List[str]:
        """Experiment directories to search for encoder eval cache (best dir, then source seed dir)."""
        dirs: List[str] = []
        exp = self.experiment_dir
        if exp and str(exp).strip():
            dirs.append(os.path.normpath(str(exp)))
        seed_path = self.get_best_seed_run_path()
        if isinstance(seed_path, str) and seed_path.strip():
            norm = os.path.normpath(seed_path)
            if norm not in dirs:
                dirs.append(norm)
        return dirs

    def multi_seed(
        self,
        *,
        selection: str = "all_convergent",
        aggregate: str = "mean",
        dispersion: Optional[str] = None,
        return_per_seed: bool = False,
    ) -> MultiSeedTrainerView:
        """
        Return a multi-seed view for evaluation and plotting across convergent seed runs.

        All eval/plot methods on the view run once per selected seed checkpoint, then
        aggregate results. Top-level evaluation metrics are aggregated scalars (default mean);
        seed-level detail lives under the ``seeds`` key.

        Args:
            selection: Which seed runs to include: ``best``, ``all_convergent``, or ``all_tried``.
            aggregate: How to combine scalar metrics: ``mean``, ``median``, ``min``, or ``max``.
            dispersion: Dispersion statistic for ``seeds.stats``: ``none``, ``std``, ``range``,
                or ``minmax``. Defaults to ``std`` when ``analyze_seed_stability=True``, else ``none``.
            return_per_seed: If True, include full per-seed payloads under ``seeds.per_seed``.
        """
        if dispersion is None:
            args = self._training_args
            stability = bool(getattr(args, "analyze_seed_stability", False)) if args is not None else False
            dispersion = "std" if stability else "none"
        return MultiSeedTrainerView(
            self,
            selection=selection,
            aggregate=aggregate,
            dispersion=dispersion,
            return_per_seed=return_per_seed,
        )

    def pre_prune(
        self,
        pre_cfg: Optional[Any] = None,
        *,
        inplace: bool = True,
    ) -> "Trainer":
        """
        Run pre-prune (gradient-mean then prune) and keep the pruned model in memory.
        The next train() will use this model. Does not save to disk; save explicitly if needed.

        Uses self._training_args.pre_prune_config when pre_cfg is None.

        Args:
            pre_cfg: Optional pre-prune configuration. Defaults to
                ``training_args.pre_prune_config``.
            inplace: If True, prune the current model in place and use it for
                subsequent training/evaluation.
        """
        cfg = pre_cfg if pre_cfg is not None else getattr(self._training_args, "pre_prune_config", None)
        if cfg is None:
            raise ValueError("Pre-prune config required: pass pre_cfg or set training_args.pre_prune_config.")
        if pre_cfg is None and not inplace:
            raise ValueError(
                "pre_prune(inplace=False) does not make sense when run automatically from train() "
                "(config from TrainingArguments). Use inplace=True so the pruned model is used for training."
            )
        logger.info("Pre-pruning: loading model and preparing data ...")
        model = self.get_model()
        if self._training_args is not None:
            model = self._prepare_model_for_pre_prune_if_needed(model, self._training_args)
        max_size = getattr(self._training_args, "train_max_size", None) if self._training_args else None
        training_data = self.create_training_data(model, batch_size=1, max_size=max_size)
        monitor = getattr(self, "_runtime_monitor", None)
        if monitor is not None:
            monitor.mark("trainer:pre_prune:create_training_data:done", size=len(training_data))
        cache_dir = self._pre_prune_cache_dir(self._training_args)
        reuse_cache = bool(getattr(self._training_args, "reuse_pre_prune", False))
        model = pre_prune_with_cache(
            model,
            training_data,
            cfg,
            definition=self,
            cache_dir=cache_dir,
            reuse_cache=reuse_cache,
            inplace=inplace,
            runtime_monitor=monitor,
        )
        self._model_arg = model
        self._model_instance = model
        return self

    def _pre_prune_cache_dir(self, args: Optional[Any] = None) -> Optional[str]:
        training_args = args if args is not None else self._training_args
        experiment_dir = None
        if training_args is not None:
            experiment_dir = self._resolve_experiment_dir(
                getattr(training_args, "experiment_dir", None)
            )
        if experiment_dir is None:
            experiment_dir = self.experiment_dir
        return resolve_pre_prune_cache_dir(experiment_dir)

    @staticmethod
    def _maybe_fail_on_non_convergence(
        args: Any,
        *,
        convergent_count: Optional[int],
        min_convergent: Optional[int],
        run_id: Optional[str] = None,
        model: Any = None,
        pair: Any = None,
        output_dir: Optional[str] = None,
        seed_report: Optional[Sequence[dict]] = None,
        convergence_metric: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> None:
        if not getattr(args, "fail_on_non_convergence", False):
            return
        if min_convergent is None or min_convergent <= 0:
            return
        actual = int(convergent_count or 0)
        if actual < min_convergent:
            raise RuntimeError(
                format_non_convergence_error(
                    actual=actual,
                    min_required=min_convergent,
                    training_args=args,
                    run_id=run_id,
                    model=model,
                    pair=pair,
                    output_dir=output_dir,
                    seed_report=seed_report,
                    convergence_metric=convergence_metric,
                    threshold=threshold,
                )
            )

    def _cleanup_pre_prune_cache(self, args: Optional[Any] = None) -> None:
        """Remove ephemeral pre-prune cache under experiment_dir after train() finishes."""
        training_args = args if args is not None else self._training_args
        experiment_dir = None
        if training_args is not None:
            experiment_dir = self._resolve_experiment_dir(
                getattr(training_args, "experiment_dir", None)
            )
        if experiment_dir is None:
            experiment_dir = self.experiment_dir
        if remove_pre_prune_cache(experiment_dir):
            logger.info("Removed ephemeral pre-prune cache under %s", experiment_dir)

    def post_prune(self, post_cfg: Optional[Any] = None) -> "Trainer":
        """
        Run post-prune (weight-based) on the current model and keep it in memory.
        Subsequent evaluation (e.g. evaluate_encoder) will use the pruned model. Does not save to disk.

        Uses self._training_args.post_prune_config when post_cfg is None.
        """
        cfg = post_cfg if post_cfg is not None else getattr(self._training_args, "post_prune_config", None)
        if cfg is None:
            raise ValueError("Post-prune config required: pass post_cfg or set training_args.post_prune_config.")
        if post_cfg is None and getattr(cfg, "inplace", True) is False:
            raise ValueError(
                "post_prune_config.inplace=False does not make sense when post-prune is run automatically "
                "from train(). Use inplace=True so the pruned model is kept for subsequent evaluation."
            )
        logger.info(
            "Post-pruning: loading trained model and applying weight-based prune (this may take a while) ...",
        )
        model = self.get_model()
        model = _post_prune(model, cfg)
        self._model_arg = model
        self._model_instance = model
        logger.info("Post-prune done.")
        return self

    @property
    def evaluator(self) -> Any:
        """Lazy-init Evaluator(trainer)."""
        if self._evaluator is None:
            cls = self._evaluator_class if self._evaluator_class is not None else Evaluator
            self._evaluator = cls(self)
        return self._evaluator

    def plot_training_convergence(
        self,
        *,
        plot_mean_by_class: bool = True,
        plot_mean_by_feature_class: Optional[bool] = None,
        plot_correlation: bool = True,
        class_spread: Optional[Literal["minmax", "iqr", "ci95"]] = None,
        output: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: str = "png",
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Plot convergence statistics collected during GRADIEND training.

        Args:
            plot_mean_by_class: Plot mean encoder values by label class.
            plot_mean_by_feature_class: Plot means grouped by feature class.
                ``None`` lets the visualizer decide from available statistics.
            plot_correlation: Plot correlation over training steps.
            class_spread: Optional spread band behind class means.
                ``"minmax"`` shades min-max encoded values, ``"iqr"`` shades Q1-Q3,
                ``"ci95"`` shades mean +/- 1.96 standard errors,
                and ``None`` disables spread shading.
            output: Optional explicit output file path.
            show: Whether to display the figure interactively.
            title: Plot title. ``True`` uses the default title, ``False`` omits it.
            figsize: Optional Matplotlib figure size.
            img_format: File format used for generated output paths.
            dpi: Optional figure DPI.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        return self.evaluator.plot_training_convergence(
            plot_mean_by_class=plot_mean_by_class,
            plot_mean_by_feature_class=plot_mean_by_feature_class,
            plot_correlation=plot_correlation,
            class_spread=class_spread,
            output=output,
            show=show,
            title=title,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_training_convergence.__doc__ = (
        plot_training_convergence.__doc__
        + see_implementation("gradiend.visualizer.convergence.plot_training_convergence")
    )

    def _train(
        self,
        output_dir: str,
        args: Any,
        model: Any,
        model_with_gradiend_cls: Any,
        callbacks: Any,
        runtime_monitor: Any = None,
    ) -> str:
        """
        Run GRADIEND training (cache check, model creation, data, core loop, save).
        Override in subclasses to customize behavior. Returns path to saved model.
        
        Args:
            output_dir: Directory to save the trained model.
            args: TrainingArguments instance.
            model: Model identifier (string path) or ModelWithGradiend instance.
            model_with_gradiend_cls: ModelWithGradiend subclass to use when creating model from string path.
            callbacks: Optional list of TrainingCallback instances.
        
        Returns:
            Path to saved model directory.
        """
        args.output_dir = output_dir
        config = args
        # Supervised_decoder: correlation is meaningless; skip evaluation and use loss for best checkpoint
        if getattr(config, "supervised_decoder", False):
            config.do_eval = False

        if should_reuse_training_cache(
            config.use_cache,
            output_dir,
            min_convergent_seeds=int(getattr(config, "min_convergent_seeds", 1) or 1),
            training_args=config,
        ):
            logger.info(
                "GRADIEND model already exists at %s and satisfies use_cache=%r; skipping training. "
                "Use use_cache=False to retrain.",
                output_dir,
                config.use_cache,
            )
            return output_dir
        if has_saved_model(output_dir):
            invalidate_experiment_caches(self.experiment_dir)

        if not isinstance(model, ModelWithGradiend):
            if runtime_monitor is not None:
                runtime_monitor.mark("trainer:create_model_with_gradiend:start")
            if model_with_gradiend_cls is None:
                raise ValueError(
                    "model_with_gradiend_cls is required when model is not already a ModelWithGradiend. "
                    "For text models, use: model_with_gradiend_cls=TextModelWithGradiend"
                )
            # Ensure data (and thus pair/target_classes) is loaded before creating the model
            # so from_pretrained can set feature_class_encoding_direction on the model
            getattr(self, "_ensure_data_for_training", lambda: None)()
            # Resolve to custom prediction head (e.g. decoder MLM head) when it exists
            load_path = self.resolve_model_path(model) if isinstance(model, str) else model
            model_with_gradiend = create_model_with_gradiend(
                load_path,
                feature_definition=self,
                model_class=model_with_gradiend_cls,
                training_args=config,
                trust_remote_code=getattr(config, "trust_remote_code", False),
            )
            if runtime_monitor is not None:
                runtime_monitor.mark(
                    "trainer:create_model_with_gradiend:done",
                    input_dim=int(getattr(getattr(model_with_gradiend, "gradiend", None), "input_dim", 0) or 0),
                    base_model_is_sharded=bool(getattr(model_with_gradiend, "base_model_is_sharded", False)),
                )
        else:
            model_with_gradiend = model

        sync_model_source_target_from_training_args(model_with_gradiend, config)

        # Store model instance so get_model() returns the training model during training
        self._model_instance = model_with_gradiend

        training_data = self.create_training_data(
            model_with_gradiend,
            batch_size=config.base_gradient_batch_size,
            max_size=config.train_max_size,
        )
        if runtime_monitor is not None:
            runtime_monitor.mark(
                "trainer:create_training_data:done",
                size=len(training_data),
                base_gradient_batch_size=config.base_gradient_batch_size,
            )

        if len(training_data) == 0:
            raise ValueError("Training data is empty; cannot train. Check create_training_data implementation and train_max_size.")

        gradient_dataset = self.create_gradient_training_dataset(
            training_data,
            model_with_gradiend,
            cache_dir=None,
            use_cached_gradients=config.use_cached_gradients,
            dtype=model_with_gradiend.gradiend.torch_dtype,
            device=model_with_gradiend.gradiend.device_encoder,
            timing_steps=config.gradient_timing_steps,
            timing_label="train-gradient",
        )
        if runtime_monitor is not None:
            runtime_monitor.mark("trainer:create_gradient_dataset:done", size=len(gradient_dataset))
        precompute_enabled = config.precompute_gradient_batches
        if precompute_enabled is None:
            precompute_enabled = (
                bool(getattr(model_with_gradiend, "base_model_is_sharded", False))
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 1
            )
            if precompute_enabled:
                logger.info(
                    "precompute_gradient_batches=None: auto-enabled for sharded base model on multi-GPU CUDA."
                )
        elif precompute_enabled:
            logger.info("precompute_gradient_batches=True: explicitly enabled.")
        if precompute_enabled:
            logger.info(
                "Asynchronously precomputing gradient rows with buffer_size=%s.",
                config.precompute_gradient_buffer_size,
            )
            gradient_dataset = PreComputedTrainingDataset(
                gradient_dataset,
                buffer_size=config.precompute_gradient_buffer_size,
            )
        dl_kwargs = {"batch_size": config.gradiend_batch_size, "shuffle": False}
        if getattr(config, "seed", None) is not None:
            g = torch.Generator()
            g.manual_seed(int(config.seed))
            dl_kwargs["generator"] = g
        dataloader = DataLoader(gradient_dataset, **dl_kwargs)
        if runtime_monitor is not None:
            runtime_monitor.mark(
                "trainer:dataloader:done",
                gradiend_batch_size=config.gradiend_batch_size,
                precompute_gradient_batches=bool(precompute_enabled),
            )

        eval_dataset = None
        if config.do_eval and config.eval_steps > 0:
            include_other = self._default_from_training_args(
                None, "include_other_classes", fallback=False
            )
            # Use dedicated train-time encoder eval cap when set, otherwise fall back to encoder_eval_max_size
            train_eval_max_size = getattr(config, "encoder_eval_train_max_size", None)
            if train_eval_max_size is None:
                train_eval_max_size = getattr(config, "encoder_eval_max_size", None)
            eval_dataset = self.create_eval_data(
                model_with_gradiend,
                split=self.resolve_split_for_role("eval"),
                source=config.source,
                max_size=train_eval_max_size,
                include_other_classes=include_other,
            )
            if len(eval_dataset) == 0:
                logger.warning("Evaluation dataset is empty; no in-training evaluation will be performed.")
                eval_dataset = None

        previous_evaluate_fn = config.evaluate_fn
        installed_default_evaluate = False
        if (
            config.evaluate_fn is None
            and eval_dataset is not None
            and config.do_eval
            and config.eval_steps > 0
        ):
            def _default_evaluate(config_dict=None, training_stats=None, **eval_kwargs):
                return self.evaluator.evaluate_encoder(
                    eval_data=eval_dataset,
                    use_cache=False,
                )
            config.evaluate_fn = _default_evaluate
            installed_default_evaluate = True
            logger.debug(
                "Using default in-training evaluation (evaluator.evaluate_encoder on validation data); "
                "correlation and mean_by_class will be tracked."
            )

        best_output_dir = f"{output_dir}_best"
        try:
            core_train(
                model_with_gradiend,
                dataloader,
                training_args=config,
                callbacks=callbacks,
                runtime_monitor=runtime_monitor,
            )
        finally:
            if installed_default_evaluate:
                config.evaluate_fn = previous_evaluate_fn
        # core_train handles best-checkpoint selection. Do not overwrite with the
        # in-memory final-step model when a selected model already exists at output_dir.
        if getattr(config, "save_only_best", False) and has_saved_model(best_output_dir):
            logger.info("Promoting best checkpoint selected during training from %s to %s", best_output_dir, output_dir)
        elif getattr(config, "save_only_best", False) and has_saved_model(output_dir):
            logger.info("Using best checkpoint selected during training at %s", output_dir)
        else:
            model_with_gradiend.save_pretrained(output_dir)
            logger.info(f"Saved trained model to {output_dir}")

        # Select model path for post-training usage:
        # - save_only_best=True: best has been moved to output_dir by core_train
        # - save_only_best=False: keep output_dir as final, but use output_dir_best when available
        selected_output_dir = output_dir
        if not getattr(config, "save_only_best", False) and has_saved_model(best_output_dir):
            selected_output_dir = best_output_dir
            logger.info("Selected best checkpoint for post-training model state: %s", selected_output_dir)
        # Release GPU memory before next seed in multi-seed training
        if getattr(config, "max_seeds", 1) > 1:
            self._model_instance = None
            del model_with_gradiend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return selected_output_dir

    def train(
        self,
        output_dir: Optional[str] = None,
        model: Optional[Union[str, Any]] = None,
        model_with_gradiend_cls: Optional[Type[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        **training_args_overrides: Any,
    ) -> Union["Trainer", Dict[int, Any]]:
        """
        Train GRADIEND using stored TrainingArguments (and optional overrides).

        Args:
            output_dir: Directory to save the trained model. If None, resolved from
                experiment_dir (from TrainingArguments) or uses a temporary directory.
            model: Model to train. If None, uses the model passed at Trainer initialization.
                Can be a string path or ModelWithGradiend instance.
            model_with_gradiend_cls: ModelWithGradiend subclass to use when creating model from string path.
                Required if model is a string. If None, uses self.model_with_gradiend_cls or
                self.default_model_with_gradiend_cls (set by subclasses like TextPredictionTrainer).
                Examples: TextModelWithGradiend for text models.
            callbacks: Optional list of TrainingCallback instances for custom training behavior.
                If None, default callbacks are used (evaluation, normalization, checkpoint, logging).
            **training_args_overrides: Keyword arguments that override TrainingArguments values.
                These are merged with self.training_args (if set) or used to create new TrainingArguments.
                Examples: learning_rate=1e-3, num_epochs=10, experiment_dir="./results".

        Returns:
            Trainer instance (for single-seed training) or Dict[int, Any] mapping seed -> model
            (when ``saved_seed_runs='all_tried'`` in multi-seed training).

        Multi-seed behavior (when TrainingArguments.max_seeds > 1):

        - Each seed is trained from the same base model (or checkpoint path) but with a different

          random seed applied to PyTorch, Python's random, and NumPy.

        - For each seed, training statistics are collected (including encoder correlation

          and best checkpoints). A training-time score ("training_score") is derived from these.

        - Optionally, an additional encoder evaluation on the validation split is run via

          evaluate_encoder(split="validation"), capped by seed_selection_eval_max_size (or
          encoder_eval_max_size when unset). Its correlation becomes "eval_correlation".

        - The "selection_score" for each seed is:
            * eval_correlation when available,
            * otherwise training_score.
        - A convergence metric (correlation or loss) and threshold are used to count how many

          seeds "converged" (see TrainingArguments.convergent_metric and convergent_score_threshold).

        After the loop, a seed_report.json is written under <experiment_dir>/seeds containing:

        - top-level convergence info (metric, threshold, best seed, etc.)
        - a per-seed breakdown with training_score, eval_correlation, selection_score,

          convergence_metric_value, and convergence flags.
        """
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError(f"output_dir must be str or None, got {type(output_dir).__name__}")

        # Set env vars for reproducibility before any CUDA/BLAS use (PyTorch docs: set early)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        # Resolve model: use provided model or fall back to initialization model
        model = model or self._model_arg
        
        # Resolve model_with_gradiend_cls: use provided, then instance property, then default property
        if model_with_gradiend_cls is None:
            model_with_gradiend_cls = self.model_with_gradiend_cls or self.default_model_with_gradiend_cls

        # Merge TrainingArguments: start with stored args, then apply overrides
        args: Optional[TrainingArguments] = self._training_args
        if args is not None:
            args = TrainingArguments.from_dict({**args.to_dict(), **training_args_overrides})
        elif training_args_overrides:
            args = TrainingArguments.from_dict(training_args_overrides)
        else:
            args = TrainingArguments()
        args.__post_init__()  # validate e.g. not both supervised_encoder and supervised_decoder

        # Apply seed early so data loading, model init, and training are deterministic when seed is set
        if getattr(args, "seed", None) is not None:
            _apply_seed(int(args.seed))
        
        # Support output_dir in training_args_overrides
        if output_dir is None and "output_dir" in training_args_overrides:
            output_dir = training_args_overrides.pop("output_dir")

        # Resolve output_dir from merged args (so experiment_dir in training_args_overrides is respected; path matches cache check)
        exp_dir = self._resolve_experiment_dir(args.experiment_dir)
        output_dir = resolve_output_path(exp_dir, output_dir, ARTIFACT_MODEL)
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="gradiend_train_")
            logger.debug(
                "No output path or experiment_dir set; using temp dir %s. "
                "Model will be saved there; copy or save elsewhere if you need it after this run.",
                output_dir,
            )

        # Check cache before pre_prune/train/post_prune so we skip all of them when reusing
        if should_reuse_training_cache(
            args.use_cache,
            output_dir,
            min_convergent_seeds=int(getattr(args, "min_convergent_seeds", 1) or 1),
            training_args=args,
        ):
            logger.info(
                "GRADIEND model already exists at %s and satisfies use_cache=%r; skipping training. "
                "Use use_cache=False to retrain.",
                output_dir,
                args.use_cache,
            )
            self._model_arg = output_dir
            self._model_instance = None
            self._last_train_used_cache = True
            self._cleanup_pre_prune_cache(args)
            return self

        self._last_train_used_cache = False
        def _set_seed(seed_value: int) -> None:
            _apply_seed(seed_value)

        def _cleanup_seed_model_files(seed_model_dir: str) -> None:
            if not os.path.isdir(seed_model_dir):
                return
            remove_files = [
                "model.safetensors",
                "pytorch_model.bin",
                "config.json",
            ]
            for fn in remove_files:
                path = os.path.join(seed_model_dir, fn)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            for fn in os.listdir(seed_model_dir):
                if fn.startswith("mapping_") or fn.startswith("input_index_map"):
                    path = os.path.join(seed_model_dir, fn)
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
            model_subdir = os.path.join(seed_model_dir, "model")
            if os.path.isdir(model_subdir):
                try:
                    shutil.rmtree(model_subdir)
                except Exception:
                    pass

        def _best_score_from_stats(stats: Optional[Dict[str, Any]]) -> Optional[float]:
            if not stats:
                return None
            bsc = stats.get("best_score_checkpoint") or {}
            if args.supervised_decoder:
                val = bsc.get("loss")
                if isinstance(val, (int, float)):
                    return -float(val)
            val = bsc.get("correlation")
            if isinstance(val, (int, float)):
                return float(val)
            ts = stats.get("training_stats") or {}
            val = ts.get("correlation")
            return float(val) if isinstance(val, (int, float)) else None

        def _numeric_or_none(value: Any) -> Optional[float]:
            return float(value) if isinstance(value, (int, float)) else None

        def _select_best_seed_run(
            runs: List[Dict[str, Any]],
            *,
            convergence_metric: str,
        ) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
            if not runs:
                return None, None, "no_runs"
            converged_runs = [run for run in runs if bool(run.get("converged"))]
            if converged_runs:
                best_run = max(
                    converged_runs,
                    key=lambda run: (
                        _numeric_or_none(run.get("selection_score"))
                        if _numeric_or_none(run.get("selection_score")) is not None
                        else float("-inf")
                    ),
                )
                return best_run, _numeric_or_none(best_run.get("selection_score")), "best_convergent_selection_score"

            if convergence_metric == "correlation":
                best_run = max(
                    runs,
                    key=lambda run: (
                        _numeric_or_none(run.get("eval_correlation"))
                        if _numeric_or_none(run.get("eval_correlation")) is not None
                        else (
                            _numeric_or_none(run.get("convergence_metric_value"))
                            if _numeric_or_none(run.get("convergence_metric_value")) is not None
                            else (
                                _numeric_or_none(run.get("training_score"))
                                if _numeric_or_none(run.get("training_score")) is not None
                                else float("-inf")
                            )
                        )
                    ),
                )
                best_corr = _numeric_or_none(best_run.get("eval_correlation"))
                if best_corr is None:
                    best_corr = _numeric_or_none(best_run.get("convergence_metric_value"))
                if best_corr is None:
                    best_corr = _numeric_or_none(best_run.get("training_score"))
                return best_run, best_corr, "highest_correlation_fallback"

            best_run = max(
                runs,
                key=lambda run: (
                    _numeric_or_none(run.get("selection_score"))
                    if _numeric_or_none(run.get("selection_score")) is not None
                    else float("-inf")
                ),
            )
            return best_run, _numeric_or_none(best_run.get("selection_score")), "best_available_selection_score"

        def _compute_seed_topk_stability(
            *,
            convergent_seed_values: List[int],
            seed_results_by_seed: Dict[int, str],
        ) -> Optional[Dict[str, Any]]:
            if len(convergent_seed_values) < 2:
                return None
            if not (isinstance(args.min_convergent_seeds, int) and args.min_convergent_seeds > 1):
                return None
            if getattr(args, "seed_stability_topk", None) is None:
                return None

            topk_indices_by_run: Dict[str, List[int]] = {}
            run_metadata: Dict[str, Dict[str, Any]] = {}
            for seed_value in convergent_seed_values:
                seed_path = seed_results_by_seed.get(seed_value)
                if not seed_path:
                    continue
                try:
                    seed_model = self.load_model(seed_path, use_cache=False)
                    run_id = f"seed_{seed_value}"
                    topk_indices_by_run[run_id] = seed_model.get_topk_weights(
                        part=args.seed_stability_part,
                        topk=args.seed_stability_topk,
                    )
                    run_metadata[run_id] = {
                        "seed": seed_value,
                        "output_dir": seed_path,
                    }
                except Exception as e:
                    logger.warning(
                        "Could not compute top-k stability payload for convergent seed %s at %s: %s",
                        seed_value,
                        seed_path,
                        e,
                    )
                finally:
                    seed_model = None
                    self._model_instance = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if len(topk_indices_by_run) < 2:
                return None
            return trainer_stats.summarize_topk_stability(
                topk_indices_by_run,
                run_metadata=run_metadata,
                topk=args.seed_stability_topk,
                part=args.seed_stability_part,
            )

        # Multi-seed training loop
        if getattr(args, "max_seeds", 1) > 1:
            try:
                if not exp_dir:
                    seed_runs_dir = args.seed_runs_dir or tempfile.mkdtemp(prefix="gradiend_seeds_")
                else:
                    seed_runs_dir = args.seed_runs_dir or os.path.join(exp_dir, "seeds")
                os.makedirs(seed_runs_dir, exist_ok=True)

                if args.seed is None:
                    seeds = list(range(args.max_seeds))
                else:
                    seeds = [int(args.seed) + i for i in range(args.max_seeds)]

                convergent_metric = (args.convergent_metric or ("loss" if args.supervised_decoder else "correlation")).lower()
                threshold = args.convergent_score_threshold
                min_convergent = args.min_convergent_seeds
                convergent_count = 0
                # Use dedicated seed-selection eval cap when set, otherwise fall back to encoder_eval_max_size
                seed_selection_eval_max_size = getattr(args, "seed_selection_eval_max_size", None)
                if seed_selection_eval_max_size is None:
                    seed_selection_eval_max_size = getattr(args, "encoder_eval_max_size", None)

                if not isinstance(model, str):
                    model_path = getattr(model, "name_or_path", None)
                    if not model_path:
                        raise ValueError("Multi-seed training requires a model path (string) or a model with name_or_path.")
                    model_for_runs = model_path
                else:
                    model_for_runs = model
                # Resolve to custom prediction head (e.g. decoder MLM head) BEFORE experiment_dir override
                load_model_path = (
                    self.resolve_model_path(model_for_runs) if isinstance(model_for_runs, str) else model_for_runs
                )

                best_seed = None
                best_score = None
                seed_results: Dict[int, str] = {}
                seed_report: List[Dict[str, Any]] = []
                early_stop_reason: Optional[str] = None
                split_cycle_length: Optional[int] = None
                split_cycle_queue: Optional[List[int]] = None
                if (
                    getattr(args, "split_resplit_per_seed", False)
                    and getattr(args, "split_resplit_strategy", "random") == "balanced_cycle"
                ):
                    split_cycle_length = int(min_convergent or args.max_seeds or 1)
                    split_cycle_queue = list(range(split_cycle_length))

                def _clear_seed_gpu_state() -> None:
                    self._model_instance = None
                    self._evaluator = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                for seed_value in seeds:
                    split_cycle_index = None
                    if split_cycle_queue:
                        split_cycle_index = split_cycle_queue.pop(0)
                    _set_seed(seed_value)
                    refresh_splits = getattr(self, "_refresh_data_splits_for_seed", None)
                    if callable(refresh_splits):
                        refresh_splits(
                            seed_value,
                            args,
                            split_cycle_index=split_cycle_index,
                            split_cycle_length=split_cycle_length,
                        )
                    seed_dir = os.path.join(seed_runs_dir, f"seed_{seed_value}")
                    seed_output_dir = seed_dir
                    os.makedirs(seed_output_dir, exist_ok=True)

                    used_cache = False
                    trained = False
                    model_instance = None
                    if should_reuse_seed_training_cache(args.use_cache, seed_output_dir, training_args=args):
                        logger.info(
                            "Seed %s: convergent cached model found at %s; skipping training.",
                            seed_value,
                            seed_output_dir,
                        )
                        out_path = seed_output_dir
                        used_cache = True
                    else:
                        logger.info("Seed %s: starting training with output_dir=%s", seed_value, seed_output_dir)
                        model_instance = load_model_path
                        if args.pre_prune_config is not None:
                            try:
                                with RuntimeMonitor.from_training_args(args, seed_output_dir) as runtime_monitor:
                                    self._runtime_monitor = runtime_monitor
                                    runtime_monitor.mark("trainer:seed:pre_prune:start", seed=seed_value)
                                    getattr(self, "_ensure_data_for_training", lambda: None)()
                                    runtime_monitor.mark("trainer:seed:create_model_with_gradiend:start", seed=seed_value)
                                    model_instance = create_model_with_gradiend(
                                        load_model_path,
                                        feature_definition=self,
                                        model_class=model_with_gradiend_cls,
                                        training_args=args,
                                        trust_remote_code=getattr(args, "trust_remote_code", False),
                                    )
                                    runtime_monitor.mark(
                                        "trainer:seed:create_model_with_gradiend:done",
                                        seed=seed_value,
                                        input_dim=int(getattr(getattr(model_instance, "gradiend", None), "input_dim", 0) or 0),
                                        base_model_is_sharded=bool(getattr(model_instance, "base_model_is_sharded", False)),
                                    )
                                    max_size = args.train_max_size
                                    training_data = self.create_training_data(model_instance, batch_size=1, max_size=max_size)
                                    runtime_monitor.mark("trainer:seed:pre_prune:create_training_data:done", seed=seed_value, size=len(training_data))
                                    model_instance = pre_prune_with_cache(
                                        model_instance,
                                        training_data,
                                        args.pre_prune_config,
                                        definition=self,
                                        cache_dir=resolve_pre_prune_cache_dir(exp_dir),
                                        reuse_cache=bool(getattr(args, "reuse_pre_prune", False)),
                                        inplace=True,
                                        runtime_monitor=runtime_monitor,
                                    )
                                    runtime_monitor.mark("trainer:seed:pre_prune:done", seed=seed_value)
                                    self._runtime_monitor = None
                            except BaseException as exc:
                                self._runtime_monitor = None
                                if is_cuda_oom_error(exc):
                                    self._write_cuda_oom_log(f"train:seed_{seed_value}:pre_prune", exc, root=seed_output_dir)
                                model_instance = None
                                _clear_seed_gpu_state()
                                raise
                        # Temporarily override experiment_dir to seed-specific directory to avoid cache collisions
                        original_experiment_dir = args.experiment_dir
                        original_fail_on_non_convergence = args.fail_on_non_convergence
                        args.experiment_dir = seed_output_dir
                        args.fail_on_non_convergence = False
                        try:
                            try:
                                with RuntimeMonitor.from_training_args(args, seed_output_dir) as runtime_monitor:
                                    self._runtime_monitor = runtime_monitor
                                    runtime_monitor.mark("trainer:seed:train:start", seed=seed_value)
                                    out_path = self._train(
                                        output_dir=seed_output_dir,
                                        args=args,
                                        model=model_instance,
                                        model_with_gradiend_cls=model_with_gradiend_cls,
                                        callbacks=callbacks,
                                        runtime_monitor=runtime_monitor,
                                    )
                                    runtime_monitor.mark("trainer:seed:train:done", seed=seed_value, output_dir=out_path)
                            except BaseException as exc:
                                if is_cuda_oom_error(exc):
                                    self._write_cuda_oom_log(f"train:seed_{seed_value}", exc, root=seed_output_dir)
                                model_instance = None
                                _clear_seed_gpu_state()
                                raise
                        finally:
                            # Restore original experiment_dir
                            args.experiment_dir = original_experiment_dir
                            args.fail_on_non_convergence = original_fail_on_non_convergence
                            self._runtime_monitor = None
                        trained = True
                        if args.post_prune_config is not None:
                            logger.info("Seed %s: running post-prune after training ...", seed_value)
                            seed_model = self.get_model(load_directory=out_path, use_cache=False)
                            seed_model = _post_prune(seed_model, args.post_prune_config)
                            seed_model.save_pretrained(out_path)
                            logger.info("Seed %s: saved post-pruned model to %s", seed_value, out_path)
                            seed_model = None

                    # The next seed reloads a fresh base model; release all seed-local model references first.
                    model_instance = None
                    _clear_seed_gpu_state()

                    seed_results[seed_value] = out_path
                    stats = self.get_training_stats(out_path)
                    score = _best_score_from_stats(stats)

                    prev_model_arg = self._model_arg
                    prev_model_instance = self._model_instance
                    eval_corr = None
                    eval_result = None
                    # Optionally skip expensive full validation eval:
                    # - never needed for loss-based convergence (convergent_metric == "loss")
                    # - can be skipped when training_score (score) is clearly below threshold
                    # - skip when we already have a convergent seed (no need to recompute encoder analysis)
                    run_full_eval = convergent_metric != "loss"
                    if (
                        run_full_eval
                        and threshold is not None
                        and isinstance(score, (int, float))
                        and score < threshold
                    ):
                        run_full_eval = False
                    if run_full_eval and seed_value != seeds[0] and convergent_count >= 1:
                        run_full_eval = False
                    try:
                        self._model_arg = out_path
                        self._model_instance = None
                        if run_full_eval:
                            eval_result = self.evaluate_encoder(
                                split=self.resolve_split_for_role("eval"),
                                max_size=seed_selection_eval_max_size,
                                use_cache=False,
                            )
                            if isinstance(eval_result, dict):
                                eval_corr = eval_result.get("correlation")
                                if not isinstance(eval_corr, (int, float)):
                                    eval_corr = None
                    except Exception as e:
                        if is_cuda_oom_error(e):
                            self._write_cuda_oom_log(f"evaluate_encoder:seed_{seed_value}", e, root=out_path)
                        logger.warning("Seed %s: full validation encoder eval failed: %s", seed_value, e)
                    finally:
                        self._model_arg = prev_model_arg
                        # Don't restore _model_instance in multi-seed: keep it cleared to free GPU memory
                        if getattr(args, "max_seeds", 1) <= 1:
                            self._model_instance = prev_model_instance
                        else:
                            self._model_instance = None
                        prev_model_instance = None
                        _clear_seed_gpu_state()

                    selection_score = eval_corr if eval_corr is not None else score

                    metric_val = None
                    if stats:
                        bsc = stats.get("best_score_checkpoint") or {}
                        if convergent_metric == "loss":
                            metric_val = bsc.get("loss")
                            if metric_val is None:
                                metric_val = (stats.get("training_stats") or {}).get("loss")
                        else:
                            metric_val = bsc.get("correlation")
                            if metric_val is None:
                                metric_val = (stats.get("training_stats") or {}).get("correlation")

                    converged = False
                    best_step_ok = trainer_stats._best_checkpoint_step_is_after_initial(bsc if stats else {})
                    sign_ok = True
                    target_mean_product = None
                    min_target_class_abs_mean = None
                    if isinstance(metric_val, (int, float)):
                        if convergent_metric == "loss":
                            if best_step_ok and metric_val <= threshold:
                                convergent_count += 1
                                converged = True
                        else:
                            target_mean_product = trainer_stats._best_step_target_class_mean_product(
                                (stats or {}).get("training_stats") or {},
                                (stats or {}).get("best_score_checkpoint") or {},
                            )
                            if args.convergent_mean_by_class_threshold is not None:
                                min_target_class_abs_mean = trainer_stats._best_step_min_target_class_abs_mean(
                                    (stats or {}).get("training_stats") or {},
                                    (stats or {}).get("best_score_checkpoint") or {},
                                )
                            mean_ok = (
                                args.convergent_mean_by_class_threshold is None
                                or (
                                    isinstance(min_target_class_abs_mean, (int, float))
                                    and min_target_class_abs_mean >= args.convergent_mean_by_class_threshold
                                )
                            )
                            sign_ok = isinstance(target_mean_product, (int, float)) and target_mean_product < 0
                            if best_step_ok and metric_val >= threshold and mean_ok and sign_ok:
                                convergent_count += 1
                                converged = True

                    seed_report.append(
                        {
                            "seed": seed_value,
                            "output_dir": out_path,
                            "trained": trained,
                            "used_cache": used_cache,
                            "training_score": score,
                            "eval_correlation": eval_corr,
                            "selection_score": selection_score,
                            "convergence_metric": convergent_metric,
                            "convergence_metric_value": metric_val,
                            "best_checkpoint_global_step": bsc.get("global_step") if stats else None,
                            "threshold": threshold,
                            "convergent_mean_by_class_threshold": args.convergent_mean_by_class_threshold,
                            "convergent_min_target_class_abs_mean": min_target_class_abs_mean,
                            "converged": converged,
                            "split_cycle_index": split_cycle_index,
                            "split_cycle_length": split_cycle_length,
                        }
                    )
                    if (
                        split_cycle_queue is not None
                        and split_cycle_index is not None
                        and not converged
                        and (min_convergent is None or convergent_count < min_convergent)
                    ):
                        split_cycle_queue.append(split_cycle_index)
                    # Per-seed summary: path, converged, reason, convergent count / min required, max seeds
                    conv_status = "Yes" if converged else "No"
                    reason = ""
                    if convergent_metric == "loss":
                        reason = f"loss={metric_val}" if isinstance(metric_val, (int, float)) else "no metric"
                    else:
                        reason = f"{convergent_metric}={metric_val:.4f}" if isinstance(metric_val, (int, float)) else "no metric"
                        if not converged and isinstance(metric_val, (int, float)) and threshold is not None:
                            if not best_step_ok:
                                reason += f" (best checkpoint global_step={bsc.get('global_step')}; requires > 0)"
                            elif metric_val < threshold:
                                reason += f" (below threshold {threshold:.4f})"
                            elif not sign_ok:
                                if isinstance(target_mean_product, (int, float)):
                                    reason += f" (target-class mean sign criterion not met; product={target_mean_product:.4f})"
                                else:
                                    reason += " (target-class mean sign criterion unavailable)"
                            elif not mean_ok:
                                if isinstance(min_target_class_abs_mean, (int, float)) and args.convergent_mean_by_class_threshold is not None:
                                    reason += (
                                        f" (min |mean| among target classes={min_target_class_abs_mean:.4f} "
                                        f"< {args.convergent_mean_by_class_threshold:.4f})"
                                    )
                                else:
                                    reason += " (per-class mean criterion not met)"
                    min_req = min_convergent if min_convergent is not None else "—"
                    logger.info(
                        "Finished seed %s (%s). Converged: %s (%s). Convergent: %s / min_required: %s, max_seeds: %s.",
                        seed_value,
                        out_path,
                        conv_status,
                        reason,
                        convergent_count,
                        min_req,
                        args.max_seeds,
                    )

                    if min_convergent is not None and convergent_count >= min_convergent:
                        if convergent_metric == "loss":
                            logger.info(
                                "Convergence reached: %s seeds meet loss threshold %.4f.",
                                convergent_count,
                                float(threshold),
                            )
                        else:
                            mean_thr = args.convergent_mean_by_class_threshold
                            if mean_thr is not None:
                                logger.info(
                                    "Convergence reached: %s seeds meet correlation threshold %.4f and "
                                    "per-class |mean| threshold %.4f.",
                                    convergent_count,
                                    float(threshold),
                                    float(mean_thr),
                                )
                            else:
                                logger.info(
                                    "Convergence reached: %s seeds meet correlation threshold %.4f and opposite-sign target-class means.",
                                    convergent_count,
                                    float(threshold),
                                )
                        early_stop_reason = (
                            f"min_convergent_seeds reached: {convergent_count} >= {min_convergent} "
                            f"with metric={convergent_metric} threshold={threshold}"
                        )
                        eval_result = None
                        _clear_seed_gpu_state()
                        break

                    # Release GPU memory before loading next seed to avoid accumulation
                    eval_result = None
                    _clear_seed_gpu_state()

                selected_run, selected_score, selection_strategy = _select_best_seed_run(
                    seed_report,
                    convergence_metric=convergent_metric,
                )
                if selected_run is not None:
                    best_seed = int(selected_run["seed"])
                    best_score = selected_score

                if best_seed is None:
                    raise RuntimeError("Multi-seed training finished, but no valid training stats were found.")

                convergent_seeds = [
                    int(run["seed"])
                    for run in seed_report
                    if bool(run.get("converged"))
                ]
                seed_stability = _compute_seed_topk_stability(
                    convergent_seed_values=convergent_seeds,
                    seed_results_by_seed=seed_results,
                )
                if seed_stability and seed_stability.get("computed"):
                    logger.info(
                        "Top-k stability across convergent seeds (%s, topk=%s): mean_overlap=%.4f, "
                        "min=%.4f, max=%.4f, intersection=%s, union=%s.",
                        args.seed_stability_part,
                        args.seed_stability_topk,
                        float(seed_stability.get("mean_pairwise_overlap_fraction", 0.0)),
                        float(seed_stability.get("min_pairwise_overlap_fraction", 0.0)),
                        float(seed_stability.get("max_pairwise_overlap_fraction", 0.0)),
                        int(seed_stability.get("intersection_size", 0)),
                        int(seed_stability.get("union_size", 0)),
                    )

                report = {
                    "convergence_metric": convergent_metric,
                    "threshold": threshold,
                    "convergent_mean_by_class_threshold": args.convergent_mean_by_class_threshold,
                    "min_convergent_seeds": min_convergent,
                    "max_seeds": args.max_seeds,
                    "seeds_tried": [r.get("seed") for r in seed_report],
                    "convergent_count": convergent_count,
                    "convergent_seeds": convergent_seeds,
                    "best_seed": best_seed,
                    "best_selection_score": best_score,
                    "best_seed_selection_strategy": selection_strategy,
                    "early_stop_reason": early_stop_reason,
                    "runs": seed_report,
                }
                if seed_stability is not None:
                    report["topk_stability"] = seed_stability
                try:
                    report_path = os.path.join(seed_runs_dir, "seed_report.json")
                    with open(report_path, "w") as f:
                        json.dump(report, f, indent=2)
                    if not is_under_temp_dir(report_path):
                        logger.info("Wrote seed report to %s", report_path)
                except Exception as e:
                    logger.warning("Failed to write seed report: %s", e)

                best_path = seed_results[best_seed]
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                shutil.copytree(best_path, output_dir)
                if not is_under_temp_dir(output_dir):
                    logger.info("Selected best seed=%s -> %s", best_seed, output_dir)

                # Clear cached model so the next get_model() loads from output_dir (the selected best seed).
                # That first load is then cached; subsequent get_model() calls reuse the same instance.
                self._model_arg = output_dir
                self._model_instance = None

                # Check convergence and warn if non-convergent
                if convergent_count == 0 and min_convergent is not None and min_convergent > 0:
                    logger.warning(
                        "Multi-seed training completed but no seeds converged: "
                        "convergent_count=0 (required: %s) for metric=%s threshold=%.4f. "
                        "Model may not have reached the convergence threshold during training.",
                        min_convergent,
                        convergent_metric,
                        threshold if threshold is not None else 0.0,
                    )

                self._maybe_fail_on_non_convergence(
                    args,
                    convergent_count=convergent_count,
                    min_convergent=min_convergent,
                    run_id=self.run_id,
                    model=model_for_runs,
                    pair=getattr(self, "pair", None),
                    output_dir=output_dir,
                    seed_report=seed_report,
                    convergence_metric=convergent_metric,
                    threshold=threshold,
                )

                if getattr(args, "analyze_seed_stability", False):
                    if min_convergent is not None and convergent_count < min_convergent:
                        raise RuntimeError(
                            f"analyze_seed_stability=True requires at least {min_convergent} convergent "
                            f"seed(s), but only {convergent_count} converged. "
                            f"Increase max_seeds, relax convergent_score_threshold, or set "
                            f"analyze_seed_stability=False."
                        )
                    if min_convergent == 1:
                        logger.warning(_ANALYZE_SEED_STABILITY_MIN_ONE_WARNING)

                # Update training.json in output_dir with convergence info
                best_run = next(
                    (run for run in seed_report if int(run.get("seed", -1)) == int(best_seed)),
                    None,
                )
                convergence_info = {
                    "converged": convergent_count > 0 if min_convergent is not None and min_convergent > 0 else True,
                    "convergent_count": convergent_count,
                    "min_convergent_seeds": min_convergent,
                    "convergence_metric": convergent_metric,
                    "threshold": threshold,
                    "convergent_mean_by_class_threshold": args.convergent_mean_by_class_threshold,
                    "convergent_min_target_class_abs_mean": (
                        best_run.get("convergent_min_target_class_abs_mean") if isinstance(best_run, dict) else None
                    ),
                }
                if seed_stability is not None:
                    convergence_info["topk_stability"] = seed_stability
                try:
                    stats = trainer_stats.load_training_stats(output_dir)
                    if stats:
                        trainer_stats.write_training_stats(
                            output_dir,
                            training_stats=stats.get("training_stats", {}),
                            best_score_checkpoint=stats.get("best_score_checkpoint", {}),
                            training_args=stats.get("training_args", {}),
                            time_stats=stats.get("time"),
                            losses=stats.get("losses"),
                            convergence_info=convergence_info,
                            seed_stability=seed_stability,
                        )
                except Exception as e:
                    logger.debug("Could not update convergence_info in training.json: %s", e)

                saved_seed_runs = str(getattr(args, "saved_seed_runs", "best_only")).strip().lower()
                convergent_seed_set = {int(seed) for seed in convergent_seeds}
                if saved_seed_runs == "best_only":
                    for seed_value, seed_path in seed_results.items():
                        _cleanup_seed_model_files(seed_path)
                elif saved_seed_runs == "all_convergent":
                    for seed_value, seed_path in seed_results.items():
                        if int(seed_value) not in convergent_seed_set:
                            _cleanup_seed_model_files(seed_path)
                elif saved_seed_runs != "all_tried":
                    raise ValueError(
                        "saved_seed_runs must be 'best_only', 'all_convergent', or 'all_tried', "
                        f"got {saved_seed_runs!r}"
                    )

                # Auto-save convergence plot when experiment_dir is set
                if exp_dir:
                    try:
                        path = self.plot_training_convergence(show=False, experiment_dir=exp_dir)
                        if path:
                            logger.info("Saved training convergence plot: %s", path)
                    except Exception as e:
                        logger.warning("Failed to save training convergence plot: %s", e)

                return self
            finally:
                self._cleanup_pre_prune_cache(args)

        try:
            if args.seed is not None:
                _set_seed(int(args.seed))

            try:
                with RuntimeMonitor.from_training_args(args, output_dir) as runtime_monitor:
                    self._runtime_monitor = runtime_monitor
                    runtime_monitor.mark("trainer:train:start", output_dir=output_dir)
                    logger.info(f"Starting GRADIEND training with output_dir={output_dir}")

                    if getattr(args, "pre_prune_config", None) is not None:
                        runtime_monitor.mark("trainer:pre_prune:start")
                        self.pre_prune(inplace=True)
                        runtime_monitor.mark("trainer:pre_prune:done")
                        model = self._model_arg

                    out_path = self._train(
                        output_dir=output_dir,
                        args=args,
                        model=model,
                        model_with_gradiend_cls=model_with_gradiend_cls,
                        callbacks=callbacks,
                        runtime_monitor=runtime_monitor,
                    )
                    runtime_monitor.mark("trainer:train:done", output_dir=out_path)
                    self._model_arg = out_path
                    # Keep selected model in memory so immediate evaluation uses the chosen checkpoint.
                    try:
                        runtime_monitor.mark("trainer:reload_selected_model:start", output_dir=out_path)
                        self._model_instance = self.get_model(load_directory=out_path, use_cache=False)
                        runtime_monitor.mark("trainer:reload_selected_model:done", output_dir=out_path)
                    except Exception as e:
                        logger.warning("Could not load selected model into memory from %s: %s", out_path, e)
                        runtime_monitor.mark("trainer:reload_selected_model:failed", output_dir=out_path, error=str(e))
                        self._model_instance = None

                    if getattr(args, "post_prune_config", None) is not None:
                        runtime_monitor.mark("trainer:post_prune:start")
                        logger.info("Post-prune config set: running post-prune after training ...")
                        self.post_prune()
                        # Persist pruned model in place to the same output directory
                        self.get_model().save_pretrained(out_path)
                        logger.info("Saved post-pruned model to %s", out_path)
                        runtime_monitor.mark("trainer:post_prune:done", output_dir=out_path)
                    self._runtime_monitor = None
            except BaseException as exc:
                self._runtime_monitor = None
                if is_cuda_oom_error(exc):
                    self._write_cuda_oom_log("train", exc, root=output_dir)
                raise

            # Auto-save convergence plot when experiment_dir is set
            if exp_dir:
                try:
                    path = self.plot_training_convergence(show=False, experiment_dir=exp_dir)
                    if path:
                        logger.info("Saved training convergence plot: %s", path)
                except Exception as e:
                    logger.warning("Failed to save training convergence plot: %s", e)

            return self
        finally:
            self._cleanup_pre_prune_cache(args)

    def post_training(self, model_with_gradiend, **kwargs):
        """
        Optional post-training hook.

        Subclasses can override this to perform additional evaluation,
        logging, or analysis after training. The default implementation
        is a no-op so that definitions are not required to implement it.

        Args:
            model_with_gradiend: Trained GRADIEND model instance.
            **kwargs: Additional training context supplied by subclasses.
        """
        return None

    def encode(self, **kwargs: Any) -> Any:
        """Encode eval data and return encoded scalar values.

        Args:
            **kwargs: Forwarded to ``create_eval_data``.
        """
        model = self.get_model()
        eval_data = self.create_eval_data(model, **kwargs)
        return [model.encode(entry["source"], return_float=True) for entry in eval_data]

    def evaluate(self, *, kwargs_encoder: dict = None, kwargs_decoder: dict = None, **kwargs: Any) -> Dict[str, Any]:
        """Run encoder and decoder evaluation and return a combined result.

        Args:
            kwargs_encoder: Optional keyword arguments passed only to
                ``evaluate_encoder``.
            kwargs_decoder: Optional keyword arguments passed only to
                ``evaluate_decoder``.
            **kwargs: Additional keyword arguments applied to both evaluator
                calls.

        Returns:
            Dict with ``"encoder"`` and ``"decoder"`` entries.
        """
        if kwargs_encoder is not None and not isinstance(kwargs_encoder, dict):
            raise TypeError(f"kwargs_encoder must be dict or None, got {type(kwargs_encoder).__name__}")
        if kwargs_decoder is not None and not isinstance(kwargs_decoder, dict):
            raise TypeError(f"kwargs_decoder must be dict or None, got {type(kwargs_decoder).__name__}")
        return self.evaluator.evaluate(kwargs_encoder=kwargs_encoder, kwargs_decoder=kwargs_decoder, **kwargs)


    def plot_encoder_distributions(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        output: Optional[str] = None,
        output_dir: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        target_and_neutral_only: bool = True,
        split_plot_mode: str = "facet",
        include_neutral: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: str = "png",
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Plot encoder value distributions for target and optional neutral rows.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            output: Optional explicit output file path.
            output_dir: Optional output directory used when ``output`` is omitted.
            show: Whether to display the figure interactively.
            title: Plot title. ``True`` uses the default title, ``False`` omits it.
            target_and_neutral_only: If True, omit identity/auxiliary training rows.
            split_plot_mode: How split-aware data is shown, e.g. ``"facet"``.
            include_neutral: If True, include neutral evaluation rows when present.
            figsize: Optional Matplotlib figure size.
            img_format: File format used for generated output paths.
            dpi: Optional figure DPI.
            highlight_non_convergence: Override non-convergence markers.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        return self.evaluator.plot_encoder_distributions(
            encoder_df=encoder_df,
            output=output,
            output_dir=output_dir,
            show=show,
            title=title,
            target_and_neutral_only=target_and_neutral_only,
            split_plot_mode=split_plot_mode,
            include_neutral=include_neutral,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_encoder_distributions.__doc__ = (
        plot_encoder_distributions.__doc__
        + see_implementation("gradiend.visualizer.encoder_distributions.plot_encoder_distributions")
    )

    def plot_encoder_scatter(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        color_by: str = "label",
        x_col: Optional[str] = None,
        label_name_mapping: Optional[dict] = None,
        max_points: Optional[int] = None,
        show: bool = True,
        title: Optional[str] = None,
        height: int = 500,
        split: str = "test",
        highlight_non_convergence: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create an interactive Plotly scatter plot for encoder outlier inspection.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            color_by: Column used for point color.
            x_col: Optional column for the categorical x-axis.
            label_name_mapping: Optional mapping from labels to display names.
            max_points: Optional cap on plotted rows.
            show: Whether to display the Plotly figure.
            title: Optional plot title.
            height: Plot height in pixels.
            split: Split to evaluate/load when ``encoder_df`` is omitted.
            highlight_non_convergence: Override non-convergence markers.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        return self.evaluator.plot_encoder_scatter(
            encoder_df=encoder_df,
            color_by=color_by,
            x_col=x_col,
            label_name_mapping=label_name_mapping,
            max_points=max_points,
            show=show,
            title=title,
            height=height,
            split=split,
            highlight_non_convergence=highlight_non_convergence,
            **kwargs,
        )
    plot_encoder_scatter.__doc__ = (
        plot_encoder_scatter.__doc__
        + see_implementation("gradiend.visualizer.encoder_scatter.plot_encoder_scatter")
    )

    def plot_encoder_strip_by_split(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        include_neutral: bool = False,
        title: Optional[str] = None,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        jitter: float = 0.08,
        dodge: bool = True,
        point_size: float = 5.0,
        label_points: Union[bool, Literal["outliers", "outliers+sample", "sample"], str] = False,
        highlight_non_convergence: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Plot encoder values as a strip plot grouped by split.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            include_neutral: If True, include neutral rows when available.
            title: Optional plot title.
            output: Optional explicit output file path.
            show: Whether to display the Matplotlib figure.
            figsize: Optional Matplotlib figure size.
            jitter: Horizontal jitter for points.
            dodge: If True, separate split hues within each group.
            point_size: Marker size.
            label_points: Whether and how to label points.
            highlight_non_convergence: Override non-convergence markers.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        return self.evaluator.plot_encoder_strip_by_split(
            encoder_df=encoder_df,
            include_neutral=include_neutral,
            title=title,
            output=output,
            show=show,
            figsize=figsize,
            jitter=jitter,
            dodge=dodge,
            point_size=point_size,
            label_points=label_points,
            highlight_non_convergence=highlight_non_convergence,
            **kwargs,
        )
    plot_encoder_strip_by_split.__doc__ = (
        plot_encoder_strip_by_split.__doc__
        + see_implementation("gradiend.visualizer.encoder_strip_split.plot_encoder_strip_by_split")
    )

    def plot_encoder_by_target(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        plot_style: Literal["strip", "box", "violin"] = "strip",
        title: Optional[str] = None,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        jitter: float = 0.25,
        dodge: bool = True,
        point_size: float = 1.5,
        interactive: bool = False,
        height: int = 520,
        legend_loc: str = "upper right",
        highlight_non_convergence: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Plot encoder values grouped by target token within feature class.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            plot_style: Plot style: ``"strip"``, ``"box"``, or ``"violin"``.
            title: Optional plot title.
            output: Optional explicit output file path.
            show: Whether to display the plot.
            figsize: Optional Matplotlib figure size.
            jitter: Horizontal jitter for strip points.
            dodge: If True, separate split hues within each target.
            point_size: Marker size for strip plots.
            interactive: If True, create the interactive Plotly variant.
            height: Plotly height in pixels for interactive plots.
            legend_loc: Static Matplotlib legend location.
            highlight_non_convergence: Override non-convergence markers.
            **kwargs: Additional keyword arguments forwarded to the evaluator.
        """
        return self.evaluator.plot_encoder_by_target(
            encoder_df=encoder_df,
            plot_style=plot_style,
            title=title,
            output=output,
            show=show,
            figsize=figsize,
            jitter=jitter,
            dodge=dodge,
            point_size=point_size,
            interactive=interactive,
            height=height,
            legend_loc=legend_loc,
            highlight_non_convergence=highlight_non_convergence,
            **kwargs,
        )
    plot_encoder_by_target.__doc__ = (
        plot_encoder_by_target.__doc__
        + see_implementation("gradiend.visualizer.encoder_by_target.plot_encoder_by_target")
    )

    def evaluate_decoder(
        self,
        lrs: Optional[Sequence[float]] = None,
        feature_factors: Optional[Sequence[float]] = None,
        use_cache: Optional[bool] = None,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        training_like_df: Optional[pd.DataFrame] = None,
        neutral_df: Optional[pd.DataFrame] = None,
        selector: Optional[Any] = None,
        summary_extractor: Optional[Any] = None,
        summary_metrics: Optional[Sequence[str]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        increase_target_probabilities: bool = True,
        plot: bool = False,
        show: Optional[bool] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        decoder_lms_mode: Optional[str] = None,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run decoder grid evaluation for one direction (strengthen or weaken).

        Delegates to `evaluator.evaluate_decoder`. Only the datasets and feature-factor combinations
        required for the chosen direction are computed. When use_cache=True and experiment_dir is set,
        cached grid results are reused when available.

        Args:
            lrs: Optional sequence of learning rates to evaluate. If None, defaults are taken from
                `TrainingArguments.decoder_eval_lrs`.
            feature_factors: Optional sequence of feature factors to evaluate. If None, defaults are taken
                from `TrainingArguments.decoder_eval_feature_factors`.
            use_cache: If True, reuse cached decoder grid results when available under experiment_dir.
                If None, defaults are taken from `TrainingArguments.use_cache` (default: False).
            max_size_training_like: Maximum number of samples per variant for training-like decoder
                evaluation data. If None, defaults are taken from
                `TrainingArguments.decoder_eval_max_size_training_like`.
            max_size_neutral: Maximum number of samples per variant for neutral decoder evaluation data
                (and LMS text cap). If None, defaults are taken from
                `TrainingArguments.decoder_eval_max_size_neutral`.
            eval_batch_size: Optional batch size used during decoder evaluation (e.g. for LMS calls).
                If None, an appropriate default is chosen by the evaluator.
            training_like_df: Optional pre-computed training-like DataFrame. When provided, this is used
                instead of creating training-like evaluation data inside the evaluator.
            neutral_df: Optional pre-computed neutral DataFrame. When provided, this is used instead of
                creating neutral evaluation data inside the evaluator.
            selector: Optional selection policy (e.g. `SelectionPolicy`) controlling how the best candidate
                per metric is chosen.
            summary_extractor: Optional callable that post-processes raw decoder results and attaches
                derived metrics (e.g. bpi, fpi, mpi) before summarization.
            summary_metrics: Optional sequence of metric names to summarize (e.g. ["bpi", "fpi", "mpi"]).
            target_class: Optional target class id (or list of ids) to evaluate. When set (e.g. "3SG"),
                restricts evaluation to that class (or classes) for efficiency. When None, all
                target classes are evaluated.
            increase_target_probabilities: If True (default), compute **strengthen** summaries only
                (keys like "3SG"). If False, compute **weaken** summaries only (keys like "3SG_weaken").
            plot: If True, after selection run any missing evaluations needed for plotting, update cache,
                then create decoder plots.
            show: Controls whether plots are shown when plot=True. If True, display plots; if False,
                only save them. When None and plot=True, defaults to True.
            plot_kwargs: Optional dict of options forwarded to plot_probability_shifts when
                plot=True. E.g. plot_kwargs=dict(figsize=(5, 3), show=False). The ``show`` argument
                overrides plot_kwargs[\"show\"] when set.
            decoder_lms_mode: Optional override for classification decoder LMS. One of "lm", "classification_accuracy",
                or "both". If None, uses trainer config (e.g. TextClassificationConfig.decoder_lms_mode).
                Ignored for non-classification trainers.
            device: Optional device for evaluation (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
                When omitted, the model's current device is used. Freshly loaded checkpoints
                are placed on CUDA by default. If the model was moved to CPU (e.g. via
                :meth:`cpu`) or released via :meth:`unload_model`, call :meth:`cuda` /
                :meth:`get_model` yourself or pass an explicit device; a warning is logged
                when evaluation runs on CPU while CUDA is available.

        Returns:
            Dict with flattened decoder summaries. For strengthen, keys like dec["3SG"]; for weaken,
            keys like dec["3SG_weaken"]. Each summary entry includes value, feature_factor, learning_rate,
            id, strengthen, lms, base_lms. The dict always includes "grid", and when plot=True also
            "plot_paths" and/or "plot_path".
        """
        if not isinstance(increase_target_probabilities, bool):
            raise TypeError(f"increase_target_probabilities must be bool, got {type(increase_target_probabilities).__name__}")
        if not isinstance(plot, bool):
            raise TypeError(f"plot must be bool, got {type(plot).__name__}")
        if show is not None and not isinstance(show, bool):
            raise TypeError(f"show must be bool or None, got {type(show).__name__}")
        if max_size_training_like is not None and not isinstance(max_size_training_like, int):
            raise TypeError(f"max_size_training_like must be int or None, got {type(max_size_training_like).__name__}")
        if max_size_training_like is not None and max_size_training_like < 0:
            raise ValueError(f"max_size_training_like must be >= 0, got {max_size_training_like}")
        if max_size_neutral is not None and not isinstance(max_size_neutral, int):
            raise TypeError(f"max_size_neutral must be int or None, got {type(max_size_neutral).__name__}")
        if max_size_neutral is not None and max_size_neutral < 0:
            raise ValueError(f"max_size_neutral must be >= 0, got {max_size_neutral}")
        if eval_batch_size is not None and not isinstance(eval_batch_size, int):
            raise TypeError(f"eval_batch_size must be int or None, got {type(eval_batch_size).__name__}")
        if eval_batch_size is not None and eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")

        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        max_size_training_like = self._default_from_training_args(
            max_size_training_like, "decoder_eval_max_size_training_like"
        )
        max_size_neutral = self._default_from_training_args(
            max_size_neutral, "decoder_eval_max_size_neutral"
        )
        lrs = self._default_from_training_args(lrs, "decoder_eval_lrs")
        feature_factors = self._default_from_training_args(feature_factors, "decoder_eval_feature_factors")
        # Pass decoder_lms_mode via trainer attribute so modality-agnostic evaluator stays unchanged.
        prev_override = getattr(self, "_decoder_lms_mode_override", None)
        try:
            self._decoder_lms_mode_override = decoder_lms_mode
            monitor_args = self.training_args or TrainingArguments()
            with RuntimeMonitor.from_training_args(monitor_args, self.experiment_dir) as runtime_monitor:
                runtime_monitor.mark("trainer:evaluate_decoder:start")
                try:
                    eval_model = self._prepare_model_for_evaluation(
                        device=device,
                        context="Decoder evaluation",
                    )
                    result = self.evaluator.evaluate_decoder(
                        lrs=lrs,
                        feature_factors=feature_factors,
                        use_cache=use_cache,
                        max_size_training_like=max_size_training_like,
                        max_size_neutral=max_size_neutral,
                        eval_batch_size=eval_batch_size,
                        training_like_df=training_like_df,
                        neutral_df=neutral_df,
                        selector=selector,
                        summary_extractor=summary_extractor,
                        summary_metrics=summary_metrics,
                        target_class=target_class,
                        increase_target_probabilities=increase_target_probabilities,
                        plot=plot,
                        show=show,
                        plot_kwargs=plot_kwargs,
                        model_with_gradiend=eval_model,
                    )
                    runtime_monitor.mark("trainer:evaluate_decoder:done")
                    return result
                except BaseException as exc:
                    if is_cuda_oom_error(exc):
                        self._write_cuda_oom_log("evaluate_decoder", exc, monitor=runtime_monitor)
                    raise
        finally:
            self._decoder_lms_mode_override = prev_override

    def evaluate_encoder(
        self,
        model_with_gradiend: Optional[Any] = None,
        encoder_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        split: EncoderSplit = "test",
        source: Optional[str] = None,
        max_size: Optional[int] = None,
        neutral_data_df: Optional[pd.DataFrame] = None,
        use_cache: Optional[bool] = None,
        return_df: bool = False,
        plot: bool = False,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        is_decoder_only_model: Optional[bool] = None,
        pre_load_gradients: Optional[bool] = None,
        include_other_classes: Optional[bool] = None,
        use_all_transitions: bool = False,
        transition_selection: Optional[Any] = None,
        device: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run encoder analysis and return correlation metrics; default model to current trainer model.

        When encoder_df is provided (DataFrame or dict with "encoder_df" key), skips encoding
        and computes metrics from that DataFrame. Otherwise uses _analyze_encoder to produce
        the DataFrame (training + neutral variants), then delegates to EncoderEvaluator.

        Args:
            model_with_gradiend: Optional ModelWithGradiend instance to evaluate.
                If None, uses the trainer's current model (via get_model()).
            encoder_df: Optional DataFrame or dict with "encoder_df" key. If provided, skips
                encoding and computes metrics from this data. Use evaluate_encoder(return_df=True)
                to get such a dict, or pass a pre-computed DataFrame directly.
            split: Dataset split(s) for evaluation. A single name (default ``"test"``),
                ``"all"``, or a sequence such as ``["train", "test"]``. Multiple splits
                are encoded together and tagged with ``data_split`` for plots and metrics.
            source: Source type for gradient creation. If None, uses default from training args
                or "factual". Options: "factual", "alternative", "diff".
            max_size: Maximum number of samples per variant to encode. If None, uses
                encoder_eval_max_size from training args.
            neutral_data_df: Optional DataFrame with neutral examples (neutral_dataset variant).
                If provided, these will be encoded in addition to training data.
            use_cache: If True, use cached encoder evaluation when available.
                If None, uses use_cache from training args (default: False).
            return_df: If True, include encoder_df (full DataFrame with type column) in result.
            plot: If True, create encoder distribution plot from analyzed data.
            plot_kwargs: Optional dict of options forwarded to plot_encoder_distributions when
                plot=True. E.g. plot_kwargs=dict(target_and_neutral_only=True, show=False).
                Any argument accepted by plot_encoder_distributions can be passed here.
            is_decoder_only_model: Whether the model is decoder-only (causal LM).
                If None, inferred from the model.
            pre_load_gradients: If True, pre-load cached gradients when available.
                If None, uses use_cached_gradients from training args (default: False).
            include_other_classes:
                Coarse evaluation shortcut that includes all transitions already
                present in the selected split. Prefer ``use_all_transitions`` for
                new code. Ignored when ``transition_selection`` is provided.
                If None, uses ``include_other_classes`` from training args (default: False).
            use_all_transitions:
                If True, include all available transitions in the selected split
                during encoder evaluation. This is the simplest way to inspect
                non-target updates alongside the target pair.
            transition_selection:
                Optional explicit transition specs, e.g.
                ``[pair("happy", "sad", symmetric=True), identity("calm")]``.
                These transitions are included in addition to the active target
                pair during evaluation. Non-target probe transitions keep label
                ``0``. When provided, this overrides ``include_other_classes``
                and ``use_all_transitions``.
            device: Optional device for encoding / evaluation (e.g. ``"cuda"``,
                ``"cuda:0"``, ``"cpu"``). When omitted, the model's current device
                is used. Freshly loaded checkpoints are placed on CUDA by default.
                If the model was moved to CPU (e.g. via :meth:`cpu`) or released via
                :meth:`unload_model`, call :meth:`cuda` / :meth:`get_model` yourself
                or pass an explicit device; a warning is logged when evaluation runs
                on CPU while CUDA is available.
            **kwargs: Additional arguments passed to _analyze_encoder and
                create_eval_data.

        Returns:
            Dict with unified encoder metrics: n_samples, all_data, training_only,
            target_classes_only, correlation, mean_by_class, mean_by_type, boundaries;
            optionally neutral_mean_by_type, mean_by_feature_class, label_value_to_class_name.
            If return_df=True, includes "encoder_df" key.
        """
        if isinstance(split, (str, bytes)):
            pass
        elif isinstance(split, Sequence):
            if len(split) == 0:
                raise ValueError("split sequence must not be empty")
        else:
            raise TypeError(f"split must be str or a sequence of str, got {type(split).__name__}")
        if max_size is not None and not isinstance(max_size, int):
            raise TypeError(f"max_size must be int or None, got {type(max_size).__name__}")
        if max_size is not None and max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        if not isinstance(return_df, bool):
            raise TypeError(f"return_df must be bool, got {type(return_df).__name__}")
        if not isinstance(plot, bool):
            raise TypeError(f"plot must be bool, got {type(plot).__name__}")
        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        if source is not None:
            validate_source_target("source", source)

        resolved_encoder_df = _resolve_encoder_df(encoder_df)
        if resolved_encoder_df is not None:
            encoder_df_out = resolved_encoder_df
        else:
            mwg = self._prepare_model_for_evaluation(
                model_with_gradiend,
                device=device,
                context="Encoder evaluation",
            )
            if max_size is None:
                max_size = self._default_from_training_args(max_size, "encoder_eval_max_size")
            if source is not None:
                kwargs["source"] = source
            if is_decoder_only_model is not None:
                kwargs["is_decoder_only_model"] = is_decoder_only_model
            if pre_load_gradients is not None:
                kwargs["pre_load_gradients"] = pre_load_gradients
            kwargs["include_other_classes"] = self._default_from_training_args(
                include_other_classes, "include_other_classes", fallback=False
            )
            kwargs["use_all_transitions"] = use_all_transitions
            kwargs["transition_selection"] = transition_selection
            monitor_args = self.training_args or TrainingArguments()
            with RuntimeMonitor.from_training_args(monitor_args, self.experiment_dir) as runtime_monitor:
                runtime_monitor.mark("trainer:evaluate_encoder:analyze:start", split=split, max_size=max_size)
                try:
                    encoder_df_out = self._analyze_encoder(
                        mwg,
                        split=split,
                        max_size=max_size,
                        neutral_data_df=neutral_data_df,
                        use_cache=use_cache,
                        plot=False,
                        **kwargs,
                    )
                    runtime_monitor.mark("trainer:evaluate_encoder:analyze:done", split=split, max_size=max_size)
                except BaseException as exc:
                    if is_cuda_oom_error(exc):
                        self._write_cuda_oom_log("evaluate_encoder", exc, monitor=runtime_monitor)
                    raise
        monitor_args = self.training_args or TrainingArguments()
        with RuntimeMonitor.from_training_args(monitor_args, self.experiment_dir) as runtime_monitor:
            runtime_monitor.mark("trainer:evaluate_encoder:start", split=split, max_size=max_size)
            try:
                result = self.evaluator.evaluate_encoder(
                    encoder_df=encoder_df_out,
                    use_cache=use_cache,
                    split=split,
                    max_size=max_size,
                    **{k: v for k, v in kwargs.items() if k not in ("return_df", "plot", "plot_kwargs")},
                )
                runtime_monitor.mark("trainer:evaluate_encoder:done", split=split, max_size=max_size)
            except BaseException as exc:
                if is_cuda_oom_error(exc):
                    self._write_cuda_oom_log("evaluate_encoder", exc, monitor=runtime_monitor)
                raise
        if return_df:
            result["encoder_df"] = encoder_df_out
        if plot:
            try:
                plot_args = dict(plot_kwargs or {})
                if "output" not in plot_args and self.experiment_dir:
                    plot_key_split = encoder_split_cache_key(
                        split,
                        available=(
                            self.combined_data["split"].dropna().astype(str).tolist()
                            if getattr(self, "combined_data", None) is not None
                            and hasattr(self.combined_data, "columns")
                            and "split" in self.combined_data.columns
                            else None
                        ),
                    )
                    output = resolve_encoder_plot_path(
                        self.experiment_dir,
                        split=plot_key_split,
                        max_size=max_size,
                    )
                    if output:
                        plot_args["output"] = output
                if "show" not in plot_args:
                    plot_args["show"] = True
                if hasattr(self, "config") and getattr(self.config, "img_format", None) is not None:
                    plot_args.setdefault("img_format", self.config.img_format)
                self.plot_encoder_distributions(encoder_df=encoder_df_out, **plot_args)
            except ImportError as e:
                logger.warning("Skipping encoder distribution plot: %s", e)
        return result

    def get_training_stats(self, model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load training stats; default model_path to current trainer model path."""
        path = model_path if model_path is not None else self.model_path
        return super().get_training_stats(path)

    def get_encodings(self, model_path: Optional[str] = None, **kwargs: Any) -> Any:
        """Get encodings; default ``model_path`` to the current trainer model path.

        Args:
            model_path: Optional model path. Defaults to ``self.model_path``.
            **kwargs: Additional arguments passed to the base implementation.
        """
        path = model_path if model_path is not None else self.model_path
        return super().get_encodings(path, **kwargs)

    def get_encoder_metrics(
        self,
        model_path: Optional[str] = None,
        encoder_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Get unified encoder metrics from encoder_df or from cached results.

        Args:
            model_path: Path to the model; defaults to the trainer's current model path.
            encoder_df: Optional DataFrame with encoded values and labels. If provided, metrics
                are computed directly from this DataFrame (same format as evaluate_encoder output).
                Use when you already have encoder outputs, e.g. from evaluate_encoder(return_df=True).
            **kwargs: Additional arguments passed to the base implementation (e.g. split, use_cache).
                When using cache instead of encoder_df, pass the same kwargs you use for
                evaluate_encoder.

        Returns:
            Dict with n_samples, correlation, mean_by_class, etc., or None if encoder_df is empty.
        """
        path = model_path if model_path is not None else self.model_path
        return super().get_encoder_metrics(path, encoder_df=encoder_df, **kwargs)

    def rewrite_base_model(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        increase_target_probabilities: bool = True,
        output_dir: Optional[str] = None,
        base_model: Optional[Any] = None,
        decoder_stats_metric_name: Optional[str] = None,
        **decoder_stats_kwargs: Any,
    ) -> Union[Any, List[Any], str, List[str]]:
        """
        Rewrite the base model by applying GRADIEND decoder updates based on decoder evaluation results.

        The decoder evaluation selects a feature factor and learning rate per target class and direction
        (strengthening vs weakening). This method applies the selected config: by default it strengthens
        the given target class(es); use increase_target_probabilities=False to apply the weakening config
        instead (evaluate_decoder currently only produces strengthen summaries).

        ``learning_rate`` and ``feature_factor`` from decoder results (and probability-shift plots)
        are passed through unchanged to :meth:`ModelWithGradiend.rewrite_base_model` for every
        encoder source.

        Accepts/loads internally the cached decoder results when experiment_dir is set. Optionally saves
        the rewritten model(s) to disk if output_dir is provided.

        When called on a Trainer, the trained model is used automatically (base_model not needed).
        When experiment_dir is set, decoder_results can be omitted and will be loaded from cache
        (requires evaluate_decoder to have been run with use_cache=True so the decoder stats cache exists).

        Args:
            decoder_results: Output from evaluate_decoder. Optional when
                experiment_dir is set (then loaded from cache).
            target_class: Target class(es) to rewrite for. Must be key(s) present in decoder_results
                summary (e.g. per-class ids like "3SG", "masc_nom", or "combined_score"). Pass a single
                string for one model, or a list of strings for one rewritten model per class. For
                strengthening, use the class id (e.g. "masc_nom"); for weakening, the summary key is
                "<class>_weaken" but you pass the class id here and set increase_target_probabilities=False.
            increase_target_probabilities: If True (default), apply the config that strengthens the
                target class (higher probability for that class). If False, apply the config that weakens
                it (uses opposite feature factor; evaluate_decoder currently only produces strengthen summaries).
            output_dir: Optional directory where the rewritten model(s) should be saved.
                If provided, models are saved to disk. If None, models are returned in memory only.
                For a single target_class, this is the exact save directory. For multiple target_classes,
                experiment_dir must be set (output_dir is only used for a single target_class).
            base_model: ModelWithGradiend to rewrite; if None, uses current trainer model.
            decoder_stats_metric_name: When loading decoder_results from cache, the summary key
                used to locate the cache file. Defaults to first target_class or "combined_score".
            **decoder_stats_kwargs: Used to locate the cache file (feature_factors, lrs, etc.).

        Returns:
            If output_dir is None:
                Rewritten model when target_class is a single string; list of models when target_class is a list.
            If output_dir is provided:
                Path to the saved rewritten model directory, or list of paths when target_class is a list.
        """
        if base_model is None:
            base_model = self.get_model()
        if base_model is None:
            raise ValueError(
                "Base model is required to rewrite base model. "
                "Provide a ModelWithGradiend instance or ensure the trainer has a valid model path."
            )

        if decoder_results is None:
            experiment_dir = self.experiment_dir
            if not experiment_dir:
                raise ValueError(
                    "decoder_results is required when experiment_dir is not set. "
                    "Run evaluate_decoder() first or set experiment_dir on TrainingArguments."
                )
            load_metric = (
                decoder_stats_metric_name
                or (target_class[0] if isinstance(target_class, list) and target_class else target_class)
                or "combined_score"
            )
            stats_file = resolve_decoder_stats_path(
                experiment_dir,
                metric_name=load_metric,
                feature_factors=decoder_stats_kwargs.get("feature_factors"),
                lrs=decoder_stats_kwargs.get("lrs"),
                topk=decoder_stats_kwargs.get("topk"),
                part=decoder_stats_kwargs.get("part"),
                topk_part=decoder_stats_kwargs.get("topk_part"),
            )
            if stats_file and os.path.isfile(stats_file):
                try:
                    decoder_results = read_decoder_stats_file(stats_file)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load decoder results from cache at {stats_file}: {e}. "
                        "Run evaluate_decoder() first or provide decoder_results explicitly."
                    )
            else:
                raise ValueError(
                    f"No decoder results cache found at {stats_file}. "
                    "Run evaluate_decoder() first or provide decoder_results explicitly."
                )

        _reserved = {"grid", "plot_path", "plot_paths"}
        # Prefer nested \"summary\" dict (evaluate_decoder typical format), fall back to top-level keys.
        if isinstance(decoder_results.get("summary"), dict):
            summary_source = decoder_results["summary"]
        else:
            summary_source = {k: v for k, v in decoder_results.items() if k not in _reserved}
        raw_keys: List[str] = (
            [target_class] if isinstance(target_class, str) else (target_class or [])
        )
        if not raw_keys:
            raise ValueError(
                "target_class must be a non-empty string or list of strings present in decoder results. "
                f"Available keys: {list(summary_source.keys())}"
            )
        keys_to_process: List[str] = [
            k if increase_target_probabilities else f"{k}_weaken" for k in raw_keys
        ]

        # When output_dir is passed but empty, user intended to save but gave no path
        if output_dir is not None and not str(output_dir).strip():
            raise ValueError(
                "Cannot save rewritten model: no output path. "
                "Set experiment_dir on TrainingArguments or pass a non-empty output_dir to rewrite_base_model."
            )

        # Check if we need to save
        should_save = output_dir is not None and str(output_dir).strip()
        if should_save:
            has_experiment_dir = bool(self.experiment_dir and str(self.experiment_dir).strip())
            has_output_dir = bool(output_dir is not None and str(output_dir).strip())
            if not has_experiment_dir and not has_output_dir:
                raise ValueError(
                    "Cannot save rewritten model: no output path. "
                    "Set experiment_dir on TrainingArguments or pass output_dir to rewrite_base_model."
                )
            if len(keys_to_process) > 1 and not has_experiment_dir:
                raise ValueError(
                    "Cannot save multiple rewritten models without experiment_dir. "
                    "Set experiment_dir on TrainingArguments (output_dir is only used for a single target_class)."
                )

        rewritten_models: List[Any] = []
        for key in keys_to_process:
            summary = summary_source.get(key)
            if not summary or "feature_factor" not in summary or "learning_rate" not in summary:
                hint = ""
                if not increase_target_probabilities and not key.endswith("_weaken"):
                    hint = " evaluate_decoder currently only produces strengthen summaries."
                raise ValueError(
                    f"Decoder results do not contain summary for metric '{key}'. "
                    f"Available keys: {list(summary_source.keys())}.{hint}"
                )
            feature_factor = summary["feature_factor"]
            lr = summary["learning_rate"]
            rewritten = base_model.rewrite_base_model(
                learning_rate=lr,
                feature_factor=feature_factor,
            )
            rewritten_models.append(rewritten)

        # Save if output_dir was provided
        if should_save:
            saved_paths: List[str] = []
            for i, key in enumerate(keys_to_process):
                summary = summary_source.get(key)
                feature_factor = summary["feature_factor"]
                lr = summary["learning_rate"]
                explicit = output_dir if len(keys_to_process) == 1 else None
                key_output_dir = require_output_path(
                    self.experiment_dir, explicit, ARTIFACT_MODEL_CHANGED, target_class=key
                )
                os.makedirs(key_output_dir, exist_ok=True)
                rewritten_models[i].save_pretrained(key_output_dir)
                logger.info(
                    f"Saved rewritten model to {key_output_dir} "
                    f"(feature_factor={feature_factor}, lr={lr}, metric={key})"
                )
                saved_paths.append(key_output_dir)
            return saved_paths[0] if len(saved_paths) == 1 else saved_paths

        # Return models in memory
        return rewritten_models[0] if len(rewritten_models) == 1 else rewritten_models
