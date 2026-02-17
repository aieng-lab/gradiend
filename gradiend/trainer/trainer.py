"""
Trainer: HF-like API with model at creation time and lazy Evaluator.

Trainer subclasses FeatureLearningDefinition and adds: model at construction, training_arguments (optional),
get_model(), evaluator_class with lazy Evaluator. Training logic lives in _train();
override _train() in subclasses to customize.
"""

import os
import tempfile
import shutil
import random
import numpy as np
import json
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import pandas as pd
import torch

from torch.utils.data import DataLoader

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
)
from gradiend.evaluator.decoder_eval_utils import read_decoder_stats_file
from gradiend.util.logging import get_logger
from gradiend.trainer.core.training import train as core_train
from gradiend.trainer.factory import create_model_with_gradiend
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.evaluator import Evaluator
from gradiend.trainer.core.pruning import pre_prune as _pre_prune

logger = get_logger(__name__)


class Trainer(FeatureLearningDefinition):
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
        select_changed_model(): Return modified model(s) in memory from decoder evaluation.
        select_and_save_changed_model(): Save modified models based on decoder evaluation.
    
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
        super().__init__(target_classes, run_id, n_features)
        self._model_arg = model
        self._base_model_arg = model  # Original model; never changes (model_path does after train)
        self._model_instance: Optional[Any] = None
        self._training_args = args
        self._evaluator_class = evaluator_class if evaluator_class is not None else Evaluator
        self._evaluator: Optional[Any] = None
        self._model_with_gradiend_cls: Optional[Type[Any]] = None

    @property
    def training_args(self) -> Optional[TrainingArguments]:
        return self._training_args

    def _experiment_dir(self) -> Optional[str]:
        """Root directory for this experiment (experiment_dir, or experiment_dir/run_id when run_id is set)."""
        if self.training_args is None:
            return None
        exp = self.training_args.experiment_dir
        if not exp:
            return None
        exp = exp.rstrip("/\\")
        if self.run_id and str(self.run_id).strip():
            return os.path.join(exp, str(self.run_id).strip().strip("/\\"))
        return exp

    @property
    def experiment_dir(self) -> Optional[str]:
        """
        Experiment directory for this trainer.

        If training_args.experiment_dir is set, returns that (with run_id subdir if run_id is set).

        """
        return self._experiment_dir()

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
        **kwargs: Any,
    ) -> Any:
        """
        Get the trainer's ModelWithGradiend instance.

        Returns the in-memory model when set (e.g. during training), otherwise loads from
        load_directory or model_path. When use_cache=True (default), caches the loaded
        instance for subsequent calls.

        To load a model from a different directory, use load_model() or pass load_directory=.

        Args:
            use_cache: If True, cache and reuse the instance after first load.
            load_directory: If provided, load from this path (GRADIEND checkpoint expected).
            **kwargs: Passed to model_with_gradiend_cls.from_pretrained.

        Returns:
            ModelWithGradiend instance.
        """
        load_directory = kwargs.pop("load_directory", None)
        # Return in-memory model when set (e.g. during training), unless loading from a specific path
        if load_directory is None and self._model_instance is not None:
            return self._model_instance
        use_cache = self._default_from_training_args(use_cache, "use_cache", fallback=False)
        # Pass definition so models get pair/classes when loading
        kwargs.setdefault("definition", self)
        if self._training_args is not None:
            kwargs.setdefault("training_args", self._training_args)
            if "trust_remote_code" not in kwargs:
                kwargs.setdefault("trust_remote_code", getattr(self._training_args, "trust_remote_code", False))
        load_directory = load_directory if load_directory is not None else self.model_path
        model = super().create_model_with_gradiend(load_directory, **kwargs)
        if use_cache:
            self._model_instance = model
        return model

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
        max_size = getattr(self._training_args, "train_max_size", None) if self._training_args else None
        training_data = self.create_training_data(model, batch_size=1, max_size=max_size)
        model = _pre_prune(model, training_data, cfg, definition=self, inplace=inplace)
        self._model_arg = model
        self._model_instance = model
        return self

    def post_prune(self, post_cfg: Optional[Any] = None) -> "Trainer":
        """
        Run post-prune (weight-based) on the current model and keep it in memory.
        Subsequent evaluation (e.g. evaluate_encoder) will use the pruned model. Does not save to disk.

        Uses self._training_args.post_prune_config when post_cfg is None.
        """
        from gradiend.trainer.core.pruning import post_prune as _post_prune

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

    def plot_training_convergence(self, **kwargs: Any) -> Any:
        """
        Plot training convergence (means by class/feature_class and correlation).
        Delegates to Evaluator/Visualizer. If experiment_dir is set, output path is auto-resolved.
        """
        return self.evaluator.plot_training_convergence(**kwargs)

    def _train(
        self,
        output_dir: str,
        args: Any,
        model: Any,
        model_with_gradiend_cls: Any,
        callbacks: Any,
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

        if config.use_cache and has_saved_model(output_dir):
            logger.info(
                f"GRADIEND model already exists at {output_dir}, skipping training. Use use_cache=False to retrain."
            )
            return output_dir
        if has_saved_model(output_dir):
            invalidate_experiment_caches(self.experiment_dir)

        if isinstance(model, str):
            if model_with_gradiend_cls is None:
                raise ValueError(
                    "model_with_gradiend_cls is required when model is a string. "
                    "For text models, use: model_with_gradiend_cls=TextModelWithGradiend"
                )
            # Ensure data (and thus pair/target_classes) is loaded before creating the model
            # so from_pretrained can set feature_class_encoding_direction on the model
            getattr(self, "_ensure_data_for_training", lambda: None)()
            # Resolve to custom prediction head (e.g. decoder MLM head) when it exists
            load_path = self.resolve_model_path(model)
            model_with_gradiend = create_model_with_gradiend(
                load_path,
                feature_definition=self,
                model_class=model_with_gradiend_cls,
                training_args=config,
                trust_remote_code=getattr(config, "trust_remote_code", False),
            )
        else:
            model_with_gradiend = model

        # Store model instance so get_model() returns the training model during training
        self._model_instance = model_with_gradiend

        training_data = self.create_training_data(
            model_with_gradiend,
            batch_size=config.train_batch_size,
            max_size=config.train_max_size,
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
        )
        dataloader = DataLoader(gradient_dataset, batch_size=1, shuffle=False)

        eval_dataset = None
        if config.do_eval and config.eval_steps > 0:
            include_other = (
                getattr(getattr(self, "training_args", None), "add_identity_for_other_classes", False)
                and getattr(self, "classes", None)
                and len(getattr(self, "classes", [])) > 2
            )
            # Use dedicated train-time encoder eval cap when set, otherwise fall back to encoder_eval_max_size
            train_eval_max_size = getattr(config, "encoder_eval_train_max_size", None)
            if train_eval_max_size is None:
                train_eval_max_size = getattr(config, "encoder_eval_max_size", None)
            eval_dataset = self.create_eval_data(
                model_with_gradiend,
                split="val",
                source=config.source,
                max_size=train_eval_max_size,
                include_other_classes=include_other,
            )
            if len(eval_dataset) == 0:
                logger.warning("Evaluation dataset is empty; no in-training evaluation will be performed.")
                eval_dataset = None

        if (
            config.evaluate_fn is None
            and eval_dataset is not None
            and config.do_eval
            and config.eval_steps > 0
        ):
            def _default_evaluate(config_dict=None, training_stats=None, **eval_kwargs):
                # Use the same evaluation logic as post-training evaluation
                return self.evaluator.evaluate_encoder(
                    eval_data=eval_dataset,
                    use_cache=False,
                )
            config.evaluate_fn = _default_evaluate
            logger.info(
                "Using default in-training evaluation (evaluator.evaluate_encoder on val data); "
                "correlation and mean_by_class will be tracked."
            )

        core_train(
            model_with_gradiend,
            dataloader,
            training_args=config,
            callbacks=callbacks,
        )
        model_with_gradiend.save_pretrained(output_dir)
        logger.info(f"Saved trained model to {output_dir}")
        return output_dir

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
            (when keep_seed_runs=True in multi-seed training).

        Multi-seed behavior (when TrainingArguments.max_seeds > 1):

        - Each seed is trained from the same base model (or checkpoint path) but with a different
          random seed applied to PyTorch, Python's random, and NumPy.
        - For each seed, training statistics are collected (including encoder correlation
          and best checkpoints). A training-time score ("training_score") is derived from these.
        - Optionally, an additional encoder evaluation on the validation split is run via
          evaluate_encoder(split="val"), capped by seed_selection_eval_max_size (or
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
        
        # Handle output_dir in training_args_overrides (for backward compatibility)
        if output_dir is None and "output_dir" in training_args_overrides:
            output_dir = training_args_overrides.pop("output_dir")

        # Resolve output_dir from merged args (so experiment_dir in training_args_overrides is respected; path matches cache check)
        exp_dir = args.experiment_dir
        if exp_dir and self.run_id and str(self.run_id).strip():
            exp_dir = os.path.join(exp_dir.rstrip("/\\"), str(self.run_id).strip().strip("/\\"))
        elif exp_dir:
            exp_dir = exp_dir.rstrip("/\\")
        output_dir = resolve_output_path(exp_dir, output_dir, ARTIFACT_MODEL)
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="gradiend_train_")
            logger.debug(
                "No output path or experiment_dir set; using temp dir %s. "
                "Model will be saved there; copy or save elsewhere if you need it after this run.",
                output_dir,
            )

        # Check cache before pre_prune/train/post_prune so we skip all of them when reusing
        if getattr(args, "use_cache", True) and has_saved_model(output_dir):
            logger.info(
                f"GRADIEND model already exists at {output_dir}, skipping training. Use use_cache=False to retrain."
            )
            self._model_arg = output_dir
            self._model_instance = None
            self._last_train_used_cache = True
            return self

        self._last_train_used_cache = False
        def _set_seed(seed_value: int) -> None:
            torch.manual_seed(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)

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

        # Multi-seed training loop
        if getattr(args, "max_seeds", 1) > 1:
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

            for seed_value in seeds:
                _set_seed(seed_value)
                seed_dir = os.path.join(seed_runs_dir, f"seed_{seed_value}")
                seed_output_dir = seed_dir
                os.makedirs(seed_output_dir, exist_ok=True)

                used_cache = False
                trained = False
                if args.use_cache and has_saved_model(seed_output_dir):
                    logger.info("Seed %s: cached model found at %s; skipping training.", seed_value, seed_output_dir)
                    out_path = seed_output_dir
                    used_cache = True
                else:
                    logger.info("Seed %s: starting training with output_dir=%s", seed_value, seed_output_dir)
                    model_instance = load_model_path
                    if args.pre_prune_config is not None:
                        getattr(self, "_ensure_data_for_training", lambda: None)()
                        model_instance = create_model_with_gradiend(
                            load_model_path,
                            feature_definition=self,
                            model_class=model_with_gradiend_cls,
                            trust_remote_code=getattr(args, "trust_remote_code", False),
                        )
                        max_size = args.train_max_size
                        training_data = self.create_training_data(model_instance, batch_size=1, max_size=max_size)
                        model_instance = _pre_prune(
                            model_instance,
                            training_data,
                            args.pre_prune_config,
                            definition=self,
                            inplace=True,
                        )
                    # Temporarily override experiment_dir to seed-specific directory to avoid cache collisions
                    original_experiment_dir = args.experiment_dir
                    args.experiment_dir = seed_output_dir
                    try:
                        out_path = self._train(
                            output_dir=seed_output_dir,
                            args=args,
                            model=model_instance,
                            model_with_gradiend_cls=model_with_gradiend_cls,
                            callbacks=callbacks,
                        )
                    finally:
                        # Restore original experiment_dir
                        args.experiment_dir = original_experiment_dir
                    trained = True
                    if args.post_prune_config is not None:
                        logger.info("Seed %s: running post-prune after training ...", seed_value)
                        from gradiend.trainer.core.pruning import post_prune as _post_prune
                        seed_model = self.get_model(load_directory=out_path, use_cache=False)
                        seed_model = _post_prune(seed_model, args.post_prune_config)
                        seed_model.save_pretrained(out_path)
                        logger.info("Seed %s: saved post-pruned model to %s", seed_value, out_path)

                seed_results[seed_value] = out_path
                stats = self.get_training_stats(out_path)
                score = _best_score_from_stats(stats)

                prev_model_arg = self._model_arg
                prev_model_instance = self._model_instance
                eval_corr = None
                # Optionally skip expensive full validation eval:
                # - never needed for loss-based convergence (convergent_metric == "loss")
                # - can be skipped when training_score (score) is clearly below threshold
                run_full_eval = convergent_metric != "loss"
                if (
                    run_full_eval
                    and threshold is not None
                    and isinstance(score, (int, float))
                    and score < threshold
                ):
                    run_full_eval = False
                try:
                    self._model_arg = out_path
                    self._model_instance = None
                    if run_full_eval:
                        eval_result = self.evaluate_encoder(
                            split="val",
                            max_size=seed_selection_eval_max_size,
                            use_cache=False,
                        )
                        if isinstance(eval_result, dict):
                            eval_corr = eval_result.get("correlation")
                            if not isinstance(eval_corr, (int, float)):
                                eval_corr = None
                except Exception as e:
                    logger.warning("Seed %s: full validation encoder eval failed: %s", seed_value, e)
                finally:
                    self._model_arg = prev_model_arg
                    self._model_instance = prev_model_instance

                selection_score = eval_corr if eval_corr is not None else score
                if selection_score is not None and (best_score is None or selection_score > best_score):
                    best_score = selection_score
                    best_seed = seed_value

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
                if isinstance(metric_val, (int, float)):
                    if convergent_metric == "loss":
                        if metric_val <= threshold:
                            convergent_count += 1
                            converged = True
                    else:
                        if metric_val >= threshold:
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
                        "threshold": threshold,
                        "converged": converged,
                    }
                )

                if min_convergent is not None and convergent_count >= min_convergent:
                    logger.info(
                        "Convergence reached: %s seeds meet %s threshold %.4f.",
                        convergent_count, convergent_metric, float(threshold),
                    )
                    early_stop_reason = (
                        f"min_convergent_seeds reached: {convergent_count} >= {min_convergent} "
                        f"with metric={convergent_metric} threshold={threshold}"
                    )
                    break

            if best_seed is None:
                raise RuntimeError("Multi-seed training finished, but no valid training stats were found.")

            report = {
                "convergence_metric": convergent_metric,
                "threshold": threshold,
                "min_convergent_seeds": min_convergent,
                "max_seeds": args.max_seeds,
                "seeds_tried": [r.get("seed") for r in seed_report],
                "convergent_count": convergent_count,
                "best_seed": best_seed,
                "best_selection_score": best_score,
                "early_stop_reason": early_stop_reason,
                "runs": seed_report,
            }
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
            
            # Update training.json in output_dir with convergence info
            convergence_info = {
                "converged": convergent_count > 0 if min_convergent is not None and min_convergent > 0 else True,
                "convergent_count": convergent_count,
                "min_convergent_seeds": min_convergent,
                "convergence_metric": convergent_metric,
                "threshold": threshold,
            }
            try:
                from gradiend.trainer.core.stats import load_training_stats, write_training_stats
                stats = load_training_stats(output_dir)
                if stats:
                    write_training_stats(
                        output_dir,
                        training_stats=stats.get("training_stats", {}),
                        best_score_checkpoint=stats.get("best_score_checkpoint", {}),
                        training_args=stats.get("training_args", {}),
                        time_stats=stats.get("time"),
                        losses=stats.get("losses"),
                        convergence_info=convergence_info,
                    )
            except Exception as e:
                logger.debug("Could not update convergence_info in training.json: %s", e)

            if not args.keep_seed_runs:
                for seed_value, seed_path in seed_results.items():
                    _cleanup_seed_model_files(seed_path)

            self._model_arg = output_dir
            self._model_instance = None

            # Auto-save convergence plot when experiment_dir is set
            if exp_dir:
                try:
                    path = self.plot_training_convergence(show=False, experiment_dir=exp_dir)
                    if path:
                        logger.info("Saved training convergence plot: %s", path)
                except Exception as e:
                    logger.warning("Failed to save training convergence plot: %s", e)

            if args.keep_seed_runs:
                seed_models: Dict[int, Any] = {}
                for seed_value, seed_path in seed_results.items():
                    try:
                        seed_models[seed_value] = self.load_model(
                            seed_path,
                            device="cpu",
                            device_encoder="cpu",
                            device_decoder="cpu",
                            device_base_model="cpu",
                        )
                    except Exception:
                        seed_models[seed_value] = seed_path
                return seed_models

            return self

        if args.seed is not None:
            _set_seed(int(args.seed))

        logger.info(f"Starting GRADIEND training with output_dir={output_dir}")

        if getattr(args, "pre_prune_config", None) is not None:
            self.pre_prune(inplace=True)
            model = self._model_arg

        out_path = self._train(
            output_dir=output_dir,
            args=args,
            model=model,
            model_with_gradiend_cls=model_with_gradiend_cls,
            callbacks=callbacks,
        )
        self._model_arg = out_path
        self._model_instance = None  # invalidate so get_model() loads from new path

        if getattr(args, "post_prune_config", None) is not None:
            logger.info("Post-prune config set: running post-prune after training ...")
            self.post_prune()
            # Persist pruned model in place to the same output directory
            self.get_model().save_pretrained(out_path)
            logger.info("Saved post-pruned model to %s", out_path)

        # Auto-save convergence plot when experiment_dir is set
        if exp_dir:
            try:
                path = self.plot_training_convergence(show=False, experiment_dir=exp_dir)
                if path:
                    logger.info("Saved training convergence plot: %s", path)
            except Exception as e:
                logger.warning("Failed to save training convergence plot: %s", e)

        return self

    def post_training(self, model_with_gradiend, **kwargs):
        """
        Optional post-training hook.

        Subclasses can override this to perform additional evaluation,
        logging, or analysis after training. The default implementation
        is a no-op so that definitions are not required to implement it.
        """
        return None

    def encode(self, **kwargs: Any) -> Any:
        """Encode eval data; return list of encoded values."""
        model = self.get_model()
        eval_data = self.create_eval_data(model, **kwargs)
        return [model.encode(entry["source"], return_float=True) for entry in eval_data]

    def evaluate(self, *, kwargs_encoder: dict = None, kwargs_decoder: dict = None, **kwargs: Any) -> Dict[str, Any]:
        """Run encoder and decoder evaluation; return combined dict."""
        return self.evaluator.evaluate(kwargs_encoder=kwargs_encoder, kwargs_decoder=kwargs_decoder, **kwargs)


    def plot_encoder_distributions(self, **kwargs: Any) -> Any:
        """Delegate to evaluator.plot_encoder_distributions."""
        return self.evaluator.plot_encoder_distributions(**kwargs)

    def plot_encoder_scatter(self, **kwargs: Any) -> Any:
        """Delegate to evaluator.plot_encoder_scatter (interactive Plotly scatter)."""
        return self.evaluator.plot_encoder_scatter(**kwargs)

    def evaluate_decoder(
            self,
            use_cache: Optional[bool] = None,
            max_size_training_like: Optional[int] = None,
            max_size_neutral: Optional[int] = None,
            eval_batch_size: Optional[int] = None,
            training_like_df: Optional[pd.DataFrame] = None,
            neutral_df: Optional[pd.DataFrame] = None,
            selector: Optional[Any] = None,
            summary_extractor: Optional[Any] = None,
            summary_metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Delegate to evaluator.evaluate_decoder. When use_cache=True and experiment_dir is set, uses cached grid results.
        Pass selector (SelectionPolicy), summary_extractor (to add custom metrics to candidates), and summary_metrics (e.g. ["bpi", "fpi", "mpi"]) for custom decoder summary behavior."""
        use_cache = self._default_from_training_args(use_cache, "use_cache", fallback=False)
        max_size_training_like = self._default_from_training_args(
            max_size_training_like, "decoder_eval_max_size_training_like"
        )
        max_size_neutral = self._default_from_training_args(
            max_size_neutral, "decoder_eval_max_size_neutral"
        )
        return self.evaluator.evaluate_decoder(
            use_cache=use_cache,
            max_size_training_like=max_size_training_like,
            max_size_neutral=max_size_neutral,
            eval_batch_size=eval_batch_size,
            training_like_df=training_like_df,
            neutral_df=neutral_df,
            selector=selector,
            summary_extractor=summary_extractor,
            summary_metrics=summary_metrics,
        )

    def evaluate_encoder(
        self,
        model_with_gradiend: Optional[Any] = None,
        encoder_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        split: str = "test",
        source: Optional[str] = None,
        max_size: Optional[int] = None,
        neutral_data_df: Optional[pd.DataFrame] = None,
        use_cache: Optional[bool] = None,
        return_df: bool = False,
        plot: bool = False,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        is_decoder_only_model: Optional[bool] = None,
        pre_load_gradients: Optional[bool] = None,
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
            split: Dataset split to use for evaluation. Default: "test".
            source: Source type for gradient creation. If None, uses default from training args
                or "factual". Options: "factual", "counterfactual", etc.
            max_size: Maximum number of samples per variant to encode. If None, uses
                encoder_eval_max_size from training args.
            neutral_data_df: Optional DataFrame with neutral examples (neutral_dataset variant).
                If provided, these will be encoded in addition to training data.
            use_cache: If True, use cached encoder evaluation when available.
                If None, uses use_cache from training args (default: False).
            return_df: If True, include encoder_df (full DataFrame with type column) in result.
            plot: If True, create encoder distribution plot from analyzed data.
            plot_kwargs: Optional kwargs passed to plot_encoder_distributions when plot=True.
            is_decoder_only_model: Whether the model is decoder-only (causal LM).
                If None, inferred from the model.
            pre_load_gradients: If True, pre-load cached gradients when available.
                If None, uses use_cached_gradients from training args (default: False).
            **kwargs: Additional arguments passed to _analyze_encoder and create_eval_data.

        Returns:
            Dict with unified encoder metrics: n_samples, all_data, training_only,
            target_classes_only, correlation, mean_by_class, mean_by_type, boundaries;
            optionally neutral_mean_by_type, mean_by_feature_class, label_value_to_class_name.
            If return_df=True, includes "encoder_df" key.
        """
        resolved_encoder_df = _resolve_encoder_df(encoder_df)
        if resolved_encoder_df is not None:
            encoder_df_out = resolved_encoder_df
        else:
            mwg = model_with_gradiend if model_with_gradiend is not None else self.get_model()
            if max_size is None:
                max_size = self._default_from_training_args(max_size, "encoder_eval_max_size")
            if source is not None:
                kwargs["source"] = source
            if is_decoder_only_model is not None:
                kwargs["is_decoder_only_model"] = is_decoder_only_model
            if pre_load_gradients is not None:
                kwargs["pre_load_gradients"] = pre_load_gradients
            encoder_df_out = self._analyze_encoder(
                mwg,
                split=split,
                max_size=max_size,
                neutral_data_df=neutral_data_df,
                use_cache=use_cache,
                plot=False,
                **kwargs,
            )
        result = self.evaluator.evaluate_encoder(
            encoder_df=encoder_df_out,
            use_cache=use_cache,
            split=split,
            max_size=max_size,
            **{k: v for k, v in kwargs.items() if k not in ("return_df", "plot", "plot_kwargs")},
        )
        if return_df:
            result["encoder_df"] = encoder_df_out
        if plot:
            plot_args = dict(plot_kwargs or {})
            if "output" not in plot_args and self.experiment_dir:
                output = resolve_encoder_plot_path(
                    self.experiment_dir,
                    split=split,
                    max_size=max_size,
                )
                if output:
                    plot_args["output"] = output
            if "show" not in plot_args:
                plot_args["show"] = True
            if hasattr(self, "config") and getattr(self.config, "img_format", None) is not None:
                plot_args.setdefault("img_format", self.config.img_format)
            self.plot_encoder_distributions(encoder_df=encoder_df_out, **plot_args)
        return result

    def get_training_stats(self, model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load training stats; default model_path to current trainer model path."""
        path = model_path if model_path is not None else self.model_path
        return super().get_training_stats(path)

    def get_encodings(self, model_path: Optional[str] = None, **kwargs: Any) -> Any:
        """Get encodings; default model_path to current trainer model path."""
        path = model_path if model_path is not None else self.model_path
        return super().get_encodings(path, **kwargs)

    def get_encoder_metrics(self, model_path: Optional[str] = None, **kwargs: Any) -> Any:
        """Get encoder metrics; default model_path to current trainer model path."""
        path = model_path if model_path is not None else self.model_path
        return super().get_encoder_metrics(path, **kwargs)

    def select_changed_model(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        metric_key: Optional[Union[str, List[str]]] = None,
        base_model: Optional[Any] = None,
        decoder_stats_metric_name: Optional[str] = None,
        **decoder_stats_kwargs: Any,
    ) -> Union[Any, List[Any]]:
        """
        Select best decoder configuration from decoder evaluation results and return changed model(s) in memory.

        Does not save to disk. Use select_and_save_changed_model() to also persist.

        When called on a Trainer, the trained model is used automatically (base_model not needed).
        When experiment_dir is set, decoder_results can be omitted and will be loaded from cache
        (requires evaluate_decoder to have been run with use_cache=True so the decoder stats cache exists).

        Args:
            decoder_results: Output from evaluate_decoder. Optional when
                experiment_dir is set (then loaded from cache).
            metric_key: Metric name(s) present in decoder_results (e.g. "combined_score",
                or per-class key like "white"). Pass a list to get one changed model per key.
            base_model: ModelWithGradiend to modify; if None, uses current trainer model.
            decoder_stats_metric_name: When loading decoder_results from cache, metric_name
                used to locate the cache file. Defaults to first metric_key or "combined_score".
            **decoder_stats_kwargs: Used to locate the cache file (feature_factors, lrs, etc.).

        Returns:
            Modified model when metric_key is a single string; list of models when metric_key is a list.
        """
        if base_model is None:
            base_model = self.get_model()
        if base_model is None:
            raise ValueError(
                "Base model is required to select changed model. "
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
                or (metric_key[0] if isinstance(metric_key, list) and metric_key else metric_key)
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

        summary_source = decoder_results.get("summary", decoder_results)
        keys_to_save: List[str] = (
            [metric_key] if isinstance(metric_key, str) else (metric_key or [])
        )
        if not keys_to_save:
            raise ValueError(
                "metric_key must be a non-empty string or list of strings present in decoder results. "
                f"Available keys: {list(summary_source.keys())}"
            )

        changed_models: List[Any] = []
        for key in keys_to_save:
            summary = summary_source.get(key)
            if summary is None:
                prefixed = f"prob::{key}"
                summary = summary_source.get(prefixed)
                if summary is not None:
                    key = prefixed
            if not summary or "feature_factor" not in summary or "learning_rate" not in summary:
                raise ValueError(
                    f"Decoder results do not contain summary for metric '{key}'. "
                    f"Available keys: {list(summary_source.keys())}"
                )
            feature_factor = summary["feature_factor"]
            lr = summary["learning_rate"]
            changed = base_model.modify_model(
                learning_rate=lr,
                feature_factor=feature_factor,
            )
            changed_models.append(changed)

        return changed_models[0] if len(changed_models) == 1 else changed_models

    def select_and_save_changed_model(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        metric_key: Optional[Union[str, List[str]]] = None,
        output_dir: Optional[str] = None,
        base_model: Optional[Any] = None,
        decoder_stats_metric_name: Optional[str] = None,
        **decoder_stats_kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Select best decoder configuration from decoder evaluation results and save changed model(s).

        Calls select_changed_model() then saves to disk. Requires experiment_dir on TrainingArguments
        or output_dir (for a single metric_key) so that a save path is available.

        When called on a Trainer, the trained model is used automatically (base_model not needed).
        When experiment_dir is set, decoder_results can be omitted and will be loaded from cache
        (requires evaluate_decoder to have been run with use_cache=True so the decoder stats cache exists).

        Args:
            decoder_results: Output from evaluate_decoder. Optional when
                experiment_dir is set (then loaded from cache).
            metric_key: Metric name(s) present in decoder_results (e.g. "combined_score",
                or per-class key like "white"). Pass a list to save one changed model per key.
            output_dir: Directory where the changed model should be saved (single key), or
                parent directory when multiple metric_keys. If None, resolved from experiment_dir.
            base_model: ModelWithGradiend to modify; if None, uses current trainer model.
            decoder_stats_metric_name: When loading decoder_results from cache, metric_name
                used to locate the cache file. Defaults to first metric_key or "combined_score".
            **decoder_stats_kwargs: Used to locate the cache file (feature_factors, lrs, etc.).

        Returns:
            Path to the saved changed model directory, or list of paths when metric_key is a list.
        """
        # Resolve base model and decoder results to know keys_to_save (needed for save-path check)
        if base_model is None:
            base_model = self.get_model()
        if base_model is None:
            raise ValueError(
                "Base model is required to select and save changed model. "
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
                or (metric_key[0] if isinstance(metric_key, list) and metric_key else metric_key)
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

        summary_source = decoder_results.get("summary", decoder_results)
        keys_to_save: List[str] = (
            [metric_key] if isinstance(metric_key, str) else (metric_key or [])
        )
        if not keys_to_save:
            raise ValueError(
                "metric_key must be a non-empty string or list of strings present in decoder results. "
                f"Available keys: {list(summary_source.keys())}"
            )

        # Require a save path: experiment_dir or output_dir (single key only)
        has_experiment_dir = bool(self.experiment_dir and str(self.experiment_dir).strip())
        has_output_dir = bool(output_dir is not None and str(output_dir).strip())
        if not has_experiment_dir and not has_output_dir:
            raise ValueError(
                "Cannot save changed model: no output path. "
                "Set experiment_dir on TrainingArguments or pass output_dir to select_and_save_changed_model."
            )
        if len(keys_to_save) > 1 and not has_experiment_dir:
            raise ValueError(
                "Cannot save multiple changed models without experiment_dir. "
                "Set experiment_dir on TrainingArguments (output_dir is only used for a single metric_key)."
            )

        changed_models = self.select_changed_model(
            decoder_results=decoder_results,
            metric_key=metric_key,
            base_model=base_model,
            decoder_stats_metric_name=decoder_stats_metric_name,
            **decoder_stats_kwargs,
        )
        if not isinstance(changed_models, list):
            changed_models = [changed_models]

        saved_paths: List[str] = []
        for i, key in enumerate(keys_to_save):
            summary = summary_source.get(key)
            if summary is None:
                prefixed = f"prob::{key}"
                summary = summary_source.get(prefixed)
                if summary is not None:
                    key = prefixed
            feature_factor = summary["feature_factor"]
            lr = summary["learning_rate"]
            explicit = output_dir if len(keys_to_save) == 1 else None
            key_output_dir = require_output_path(
                self.experiment_dir, explicit, ARTIFACT_MODEL_CHANGED, metric_key=key
            )
            os.makedirs(key_output_dir, exist_ok=True)
            changed_models[i].save_pretrained(key_output_dir)
            logger.info(
                f"Saved changed model to {key_output_dir} "
                f"(feature_factor={feature_factor}, lr={lr}, metric={key})"
            )
            saved_paths.append(key_output_dir)

        return saved_paths[0] if len(saved_paths) == 1 else saved_paths
