"""
FeatureLearningDefinition: Base class for GRADIEND feature learning definitions.

A FeatureLearningDefinition defines everything needed to train and evaluate GRADIEND for a specific feature:
- Data creation (via create_training_data)
- Evaluation (via evaluate_encoder, evaluate_decoder, etc.)

This class implements the DataProvider, Evaluator, and FeatureAnalyzer protocols.
Trainer subclasses FeatureLearningDefinition and adds model-at-construction and training entry point.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, FrozenSet, Type, Tuple
import pandas as pd
import os

from gradiend.trainer.core.protocols import DataProvider
from gradiend.util.paths import (
    resolve_custom_prediction_head_dir,
    resolve_output_path,
    ARTIFACT_CACHE_GRADIENTS,
)
from gradiend.trainer.core.stats import load_training_stats
from gradiend.util.logging import get_logger
from gradiend.model import ModelWithGradiend
from gradiend.util.paths import resolve_encoder_analysis_path
from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe, get_model_metrics

logger = get_logger(__name__)


def _resolve_encoder_df(value: Any) -> Optional[pd.DataFrame]:
    """Resolve encoder data from dict (with encoder_df key) or DataFrame."""
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        if "encoder_df" in value:
            return value["encoder_df"]
        raise ValueError(
            "evaluate_encoder(encoder_df=...): dict input must contain 'encoder_df' key. "
            "Use evaluate_encoder(return_df=True) to get such a dict."
        )
    raise TypeError(
        f"evaluate_encoder(encoder_df=...): expected DataFrame or dict with 'encoder_df' key, got {type(value).__name__}"
    )


class FeatureLearningDefinition(DataProvider, ABC):
    """
    Base class for GRADIEND feature learning definitions (superclass of Trainer).

    A FeatureLearningDefinition defines data creation, evaluation, and feature analysis for a feature.
    Trainer subclasses this and adds model-at-construction and train() entry point.

    Args:
        target_classes: Optional list of target class names for training. If None, subclasses
            can determine target_classes from data by setting self._target_classes during initialization
            or data loading. Must be unique when provided. Default: None
        run_id: Optional run identifier (subdir under experiment_dir and for display). Default: None
        n_features: Number of features (latent dimensions). Default: 1
    """

    @staticmethod
    def _validate_target_classes_unique(classes: Optional[List[str]]) -> None:
        """
        Ensure all target classes are unique. Raises ValueError if duplicates.

        By convention, index 0 maps to +1 and index 1 to -1 for binary pairs.
        """
        if not classes:
            return
        seen: set = set()
        for c in classes:
            if c in seen:
                raise ValueError(f"target_classes must be unique, got duplicate {c!r} in {classes}")
            seen.add(c)

    @staticmethod
    def _canonicalize_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
        """
        Canonicalize the ordered pair (class_pos, class_neg).

        By convention, index 0 is mapped to +1 and index 1 to -1.
        Ensures the two entries are different; more advanced logic can be added later if needed.
        """
        a, b = pair
        FeatureLearningDefinition._validate_target_classes_unique([a, b])
        return (a, b)

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate target_classes for uniqueness when assigning _target_classes."""
        if name == "_target_classes" and value is not None:
            self._validate_target_classes_unique(value)
        super().__setattr__(name, value)

    def __init__(self, target_classes: Optional[List[str]] = None, run_id: Optional[str] = None, n_features: int = 1):
        self._target_classes = target_classes
        self.run_id = run_id
        self.n_features = n_features
        self.version_map = {}
        self._model_with_gradiend_cls: Optional[Type[ModelWithGradiend]] = None

    def resolve_custom_prediction_head_dir(self) -> Optional[str]:
        """
        Return the directory path for a custom prediction head if one exists for this definition.
        
        Override in subclasses to specify where custom prediction heads are stored. For example,
        TextPredictionTrainer overrides this to return the decoder-only MLM head directory.
        
        Returns:
            Path to custom prediction head directory if it exists, None otherwise.
            Default implementation uses resolve_custom_prediction_head_dir from gradiend.util.paths.
        """
        return resolve_custom_prediction_head_dir(self.experiment_dir)
    
    def resolve_model_path(self, model: str) -> str:
        """
        Resolve a model name/path to the path to use for loading/training.
        
        This method checks if a custom prediction head exists (via resolve_custom_prediction_head_dir)
        and uses it if available. Otherwise, returns the original model path.
        
        The logic prioritizes custom prediction heads:
        1. Check if a custom prediction head exists for this definition (using self.experiment_dir)
        2. If custom head exists and is a valid directory, use it (preferred!)
        3. Otherwise, return the original model path
        
        Args:
            model: Model name or path string
            
        Returns:
            Resolved model path (custom prediction head if available, otherwise original model)
        """
        path = str(model).strip()
        
        # Always check for custom prediction head first (preferred when available)
        custom_head_dir = self.resolve_custom_prediction_head_dir()
        if custom_head_dir and os.path.isdir(custom_head_dir):
            logger.info(
                "Using custom prediction head: %s (instead of base model: %s)",
                custom_head_dir,
                model,
            )
            return custom_head_dir
        
        # Fall back to original model path
        return path

    def _experiment_dir(self) -> Optional[str]:
        """
        Root directory for this experiment (from TrainingArguments.experiment_dir).
        Base returns None. Trainer overrides to return self._training_args.experiment_dir.
        Subclasses rely on this for output path resolution; do not override output path logic.
        """
        return None

    @property
    def experiment_dir(self) -> Optional[str]:
        """Experiment directory for this definition."""
        return self._experiment_dir()

    def _get_training_arg(self, name: str) -> Any:
        args = getattr(self, "training_args", None)
        return getattr(args, name, None) if args is not None else None

    def _default_from_training_args(self, value: Any, name: str, fallback: Any = None) -> Any:
        if value is not None:
            return value
        arg_val = self._get_training_arg(name)
        return arg_val if arg_val is not None else fallback

    def _apply_training_arg_defaults(self, kwargs: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        for param, arg_name in mapping.items():
            if kwargs.get(param) is None:
                arg_val = self._get_training_arg(arg_name)
                if arg_val is not None:
                    kwargs[param] = arg_val
        return kwargs

    def get_target_feature_class_ids(self) -> Optional[List[Any]]:
        """
        Feature class IDs used for target classes (for stratification, e.g. pre_prune).
        Neutral/identity classes are excluded. Override in subclasses; base returns None.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_target_feature_class_ids() "
            "if using target feature classes."
        )

    def get_target_feature_classes(self) -> Optional[List[str]]:
        """
        Return target feature classes as string IDs when possible.

        Uses get_target_feature_class_ids() and maps via map_target_feature_class_ids().
        This is used by the target_classes property.
        """
        ids = self.get_target_feature_class_ids()
        mapped = self.map_target_feature_class_ids(ids)
        if mapped is None:
            return None
        return [str(v) for v in mapped]

    def map_target_feature_class_ids(self, ids: Optional[List[Any]]) -> Optional[List[Any]]:
        """
        Map target feature class IDs to string class IDs when possible.

        Default behavior:
        - If `pair` is available (exactly 2 target classes) and ids are 0/1 (int or float), map to pair[0]/pair[1].
        - Else if target_classes are available and ids are integer indices, map to target_classes[idx].
        - Otherwise return ids unchanged.
        """
        if not ids:
            return ids

        pair = self.pair
        if pair is not None:
            mapped: List[Any] = []
            for v in ids:
                if isinstance(v, (int, float)) and v in (0, 1):
                    mapped.append(pair[int(v)])
                else:
                    mapped.append(v)
            return mapped

        target_classes = self.target_classes
        if target_classes:
            mapped = []
            for v in ids:
                if isinstance(v, int) and 0 <= v < len(target_classes):
                    mapped.append(target_classes[v])
                else:
                    mapped.append(v)
            return mapped

        return ids

    @property
    def target_classes(self) -> Optional[List[str]]:
        """
        Target classes as string IDs (the classes we're training on).
        
        Set during initialization by subclasses. Should be derived from data automatically.
        Returns None if not set.
        """
        return self._target_classes

    @property
    def pair(self) -> Optional[Tuple[str, str]]:
        """
        Target classes as a tuple of exactly two classes.
        
        Returns a tuple (target_classes[0], target_classes[1]) if len(target_classes) == 2.
        Returns None otherwise. This provides a guarantee that exactly two target classes exist.
        
        Computed from target_classes - no setter to avoid inconsistencies.
        """
        target_classes = self.target_classes
        if target_classes is None or len(target_classes) != 2:
            return None
        return target_classes[0], target_classes[1]

    @property
    def all_classes(self) -> Optional[List[Any]]:
        """
        All classes available in the dataset (may include non-target/neutral classes).

        Set during initialization by subclasses. Should be derived from data automatically.
        Returns target classes if not set, assuming no non-target classes.
        """
        return getattr(self, "_all_classes", self.target_classes)

    @property
    def non_target_classes(self) -> Optional[List[Any]]:
        """
        Non-target classes (classes in all_classes that are not in target_classes).
        
        Automatically derived from all_classes and target_classes.
        Returns None if either all_classes or target_classes is None.
        """
        all_cls = self.all_classes
        target_cls = self.target_classes
        if all_cls is None or target_cls is None:
            return None
        target_set = set(target_cls)
        return [c for c in all_cls if c not in target_set]

    # ------------------------------------------------------------------
    # Cache path helpers
    # ------------------------------------------------------------------

    def _encoder_cache_path(self, model_path: str, **encoder_kwargs: Any) -> Optional[str]:
        """
        Return the encoder-analysis CSV path, or None when no experiment_dir is set.
        Cache under experiment_dir; includes split/max_size in cache key.
        """
        experiment_dir = self.experiment_dir
        split = encoder_kwargs.get("split")
        max_size = encoder_kwargs.get("max_size")
        key_kwargs: Dict[str, Any] = {}
        if split is not None:
            key_kwargs["split"] = split
        if max_size is not None:
            key_kwargs["max_size"] = max_size
        return resolve_encoder_analysis_path(experiment_dir, None, **key_kwargs)

    # ------------------------------------------------------------------
    # Convenience stats accessors
    # ------------------------------------------------------------------

    def get_training_stats(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Load training statistics and metadata for a saved GRADIEND model.
        """
        return load_training_stats(model_path)

    @property
    def model_with_gradiend_cls(self) -> Optional[Type[ModelWithGradiend]]:
        """
        ModelWithGradiend subclass to use for this instance (can be set per-instance).
        
        Returns None if not set. Subclasses can override this property or set
        _model_with_gradiend_cls attribute in __init__.
        """
        return self._model_with_gradiend_cls
    
    @property
    def default_model_with_gradiend_cls(self) -> Optional[Type[ModelWithGradiend]]:
        """
        Default ModelWithGradiend subclass for this trainer type.
        
        Subclasses should override this property to return their default model class.
        For example, TextPredictionTrainer returns TextModelWithGradiend.
        
        Returns None if not overridden.
        """
        return None

    def create_model_with_gradiend(
        self,
        load_directory: str,
        model_with_gradiend_cls: Optional[Type[ModelWithGradiend]] = None,
        **kwargs,
    ) -> ModelWithGradiend:
        """
        Create a ModelWithGradiend instance from a saved checkpoint directory.

        By default, uses ``self.model_with_gradiend_cls`` or ``self.default_model_with_gradiend_cls``.
        Otherwise, ``model_with_gradiend_cls`` must be provided explicitly.
        
        Args:
            load_directory: Directory containing the saved model.
            model_with_gradiend_cls: ModelWithGradiend subclass to use. If None, uses
                self.model_with_gradiend_cls or self.default_model_with_gradiend_cls.
            **kwargs: Additional arguments passed to model_class.from_pretrained.
        
        Returns:
            ModelWithGradiend instance.
        """
        cls = model_with_gradiend_cls or self.model_with_gradiend_cls or self.default_model_with_gradiend_cls
        if cls is None:
            raise ValueError(
                f"{type(self).__name__}.create_model_with_gradiend() requires a model_with_gradiend_cls "
                "argument or a default_model_with_gradiend_cls property on the definition."
            )
        return cls.from_pretrained(load_directory, **kwargs)

    @abstractmethod
    def create_training_data(self, *args, **kwargs):
        """
        Create training dataset without gradient computation (for efficiency).

        Returns:
            Training dataset compatible with GradientTrainingDataset
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement create_training_data")

    @abstractmethod
    def create_gradient_training_dataset(self, *args, **kwargs):
        """
        Create training dataset with gradient computation, wrapping the raw training data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement create_gradient_training_dataset")

    def create_eval_data(
        self,
        model_with_gradiend: Any,
        split: str = "val",
        source: Optional[str] = None,
        max_size: Optional[int] = None,
        is_decoder_only_model: Optional[bool] = None,
        pre_load_gradients: Optional[bool] = None,
        include_other_classes: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create evaluation data for encoder/decoder evaluation.

        Generic implementation: create raw training data via create_training_data,
        then wrap via create_gradient_training_dataset (modality-specific).

        Uses encoder_eval_balance from training args to set balance_column for
        create_training_data; when True, balances eval samples by feature_class_id.
        """
        source = self._default_from_training_args(source, "source", fallback="factual")
        target = self._default_from_training_args(None, "target", fallback="diff")
        max_size = self._default_from_training_args(max_size, "encoder_eval_max_size")
        pre_load_gradients = self._default_from_training_args(pre_load_gradients, "use_cached_gradients", fallback=False)
        encoder_eval_balance = self._default_from_training_args(
            kwargs.get("encoder_eval_balance"), "encoder_eval_balance", fallback=True
        )
        balance_column = "feature_class_id" if encoder_eval_balance else None

        create_kwargs = {k: v for k, v in kwargs.items() if k not in ("include_other_classes", "encoder_eval_balance")}
        raw = self.create_training_data(
            model_with_gradiend,
            split=split,
            batch_size=1,
            max_size=max_size,
            is_decoder_only_model=is_decoder_only_model,
            balance_column=balance_column,
            include_other_classes=include_other_classes,
            **create_kwargs,
        )
        cache_dir = (
            resolve_output_path(self.experiment_dir, None, ARTIFACT_CACHE_GRADIENTS)
            if pre_load_gradients
            else None
        )
        grad_kwargs = {k: v for k, v in kwargs.items() if k not in ("include_other_classes", "encoder_eval_balance")}
        return self.create_gradient_training_dataset(
            raw,
            model_with_gradiend,
            source=source,
            target=target,
            cache_dir=cache_dir,
            use_cached_gradients=pre_load_gradients,
            **grad_kwargs,
        )

    def _get_expected_encoder_keys(self, source_type: str) -> FrozenSet[Any]:
        """
        Expected (source_id, target_id) or source_id keys for encoder analysis, without iterating eval data.
        Modality-independent; uses definition.target_classes, definition.pair, definition.training_args.add_identity_for_other_classes.
        When add_identity_for_other_classes=False: every pair of classes (excluding identities).
        When add_identity_for_other_classes=True: the two training transitions + identity pairs for all other classes.
        Call after create_eval_data (or anything that sets self.target_classes) so self.pair is available.
        """
        target_classes = self.target_classes or []
        pair = self.pair
        if not target_classes:
            return frozenset()
        neutral_aug = getattr(getattr(self, "training_args", None), "add_identity_for_other_classes", False)
        if source_type == "factual":
            return frozenset(target_classes)
        if not neutral_aug:
            return frozenset((s, t) for s in target_classes for t in target_classes if s != t)
        if pair is None:
            return frozenset()
        c1, c2 = pair[0], pair[1]
        training = frozenset({(c1, c2), (c2, c1)})
        identity_others = frozenset((c, c) for c in target_classes if c not in (c1, c2))
        return training | identity_others

    @abstractmethod
    def _get_decoder_eval_dataframe(
        self,
        tokenizer,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        cached_training_like_df: Optional[pd.DataFrame] = None,
        cached_neutral_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get DataFrame for decoder evaluation.

        Args:
            tokenizer: Tokenizer
            max_size_training_like: Maximum number of generated training-like samples
            max_size_neutral: Maximum number of generated neutral samples
            cached_training_like_df: Optional cached training-like DataFrame to reuse
            cached_neutral_df: Optional cached neutral DataFrame to reuse

        Returns:
            Tuple (training_like_df, neutral_df):
            - training_like_df: Data for probability-style scoring.
            - neutral_df: Data for LMS scoring.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_decoder_eval_dataframe")

    @abstractmethod
    def _get_decoder_eval_targets(self) -> Dict[str, List[str]]:
        """
        Get target tokens for decoder evaluation, grouped by feature class.

        Returns:
            Dict mapping feature_class -> list of target tokens
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_decoder_eval_targets")

    @abstractmethod
    def evaluate_base_model(
        self,
        model,
        tokenizer,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model for decoder evaluation.

        Returns:
            Dict with 'feature_score' (bias/feature probability score) and 'lms' (language modeling score)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement evaluate_base_model")

    @staticmethod
    def _model_for_decoder_eval(model_with_gradiend: Any) -> Any:
        """
        Return the model to use for decoder evaluation.

        When a model uses a specialized head for gradient creation/training,
        evaluation should use the original underlying model to measure real change.

        This method checks if the base model implements to_original_model().
        If so, returns a ModelWithGradiend with base_model set to the original model.
        Otherwise, returns the model unchanged.
        """
        # Resolve the base model to inspect
        base = getattr(model_with_gradiend, "base_model", model_with_gradiend)

        # Check if base has to_original_model() method (duck typing)
        if hasattr(base, "to_original_model") and callable(getattr(base, "to_original_model")):
            original_model = base.to_original_model()
            logger.debug(
                "GRADIEND decoder evaluation: using original model (via to_original_model()) instead of specialized head."
            )
            # Return a ModelWithGradiend with base_model replaced by original_model
            # This preserves gradiend and modality-specific attributes (name_or_path, tokenizer, etc.)
            return model_with_gradiend.with_original_base_model(original_model)

        # No to_original_model() method - return model unchanged
        return model_with_gradiend




    @abstractmethod
    def _analyze_encoder(
        self,
        model_with_gradiend: Any,
        split: str = "test",
        neutral_data_df: Optional[pd.DataFrame] = None,
        max_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        plot: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Analyze encoder by encoding gradients from training data and optional neutral data.

        This method processes all variants in a single call:
        1. Training data (always processed)
        2. Neutral variant 1 (if decoder_eval_targets configured)
        3. Neutral variant 2 (if neutral_data_df provided)

        This method handles caching. If cached data exists and use_cache=True, it is loaded
        and returned. Otherwise, the analysis is performed and results are cached.

        Args:
            model_with_gradiend: ModelWithGradiend instance
            split: Dataset split to use
            neutral_data_df: Optional DataFrame with neutral examples (variant 2)
            max_size: Maximum number of samples per variant to encode
            use_cache: If True, use cached encoder analysis when available.
            plot: If True, create encoder distribution plot from analyzed data.
            **kwargs: Additional arguments passed to create_eval_data and modality-specific helpers

        Returns:
            DataFrame with required columns: encoded, label, source_id, target_id, type.
            The 'type' column indicates the variant: 'training', 'neutral_training_masked', or 'neutral_dataset'
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _analyze_encoder")

    def get_encodings(
        self,
        model_path: str,
        **encoder_kwargs: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Load cached encoder analysis results from disk, if available.

        Resolves the cache path via _encoder_cache_path (under experiment_dir,
        keyed by model_path and encoder_kwargs such as split and max_size).
        Does not run encoder analysis; only reads an existing CSV written by
        _analyze_encoder or equivalent.

        Args:
            model_path: Model path or identifier used for the cache key.
            **encoder_kwargs: Options used for the cache key (e.g. split, max_size).
                Defaults: split="test" if not provided.

        Returns:
            DataFrame with encoder analysis rows (e.g. encoded, label, source_id,
            target_id, type), or None if no experiment_dir, cache path cannot be
            resolved, or the cache file does not exist.
        """
        encoder_kwargs = dict(encoder_kwargs)
        encoder_kwargs.setdefault("split", "test")
        cache_path = self._encoder_cache_path(model_path, **encoder_kwargs)
        if cache_path is None:
            return None
        if os.path.exists(cache_path):
            logger.debug(f"Loading cached encoder analysis from {cache_path}")
            return pd.read_csv(cache_path)
        logger.info(f"No cached encoder analysis found at {cache_path}")
        return None

    def get_encoder_metrics(
        self,
        model_path: str,
        encoder_df: Optional[pd.DataFrame] = None,
        **encoder_kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Get unified encoder metrics from encoder_df or from cached results.
        Same format as evaluate_encoder metrics.

        Pass encoder_df explicitly or use_cache=True to load cached metrics.
        Pass the same **encoder_kwargs you use for evaluate_encoder when using cache.
        """
        encoder_kwargs = dict(encoder_kwargs)
        encoder_df_provided = encoder_kwargs.pop("encoder_df", encoder_df)

        if encoder_df_provided is not None:
            if encoder_df_provided.empty:
                return None
            return get_encoder_metrics_from_dataframe(encoder_df_provided)

        encoder_kwargs.setdefault("split", "test")
        use_cache = self._default_from_training_args(encoder_kwargs.pop("use_cache", None), "use_cache", fallback=None)
        if not use_cache:
            raise ValueError(
                "get_encoder_metrics requires encoder_df or use_cache=True to load cached metrics"
            )
        cache_path = self._encoder_cache_path(model_path, **encoder_kwargs)
        if cache_path is None:
            raise ValueError(
                f"Cannot resolve cache path for model_path={model_path} with encoder_kwargs={encoder_kwargs}"
            )
        if os.path.exists(cache_path):
            return get_model_metrics(cache_path, use_cache=use_cache)

        raise ValueError(
            f"No cached encoder analysis found at {cache_path} for model_path={model_path} "
            f"with encoder_kwargs={encoder_kwargs}. Pass encoder_df or run evaluate_encoder first."
        )
