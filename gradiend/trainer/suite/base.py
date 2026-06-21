"""Base TrainerSuite orchestration."""

from __future__ import annotations

from .definitions import *


class TrainerSuite(ABC):
    """
    Orchestrate one trainer instance per unordered pair of target classes.

    Example:
        suite = PositiveTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            args=train_args,
            data=training_df,
            target_classes=["positive", "negative", "neutral"],
            run_id="feature_suite",
        )

        suite.train(use_cache=True)
        suite.plot_similarity_heatmap(measure="cosine")
    """

    def __init__(
        self,
        trainer_cls: Type[Trainer],
        *trainer_args: Any,
        target_classes: Optional[Sequence[str]] = None,
        target_pairs: Optional[Sequence[Sequence[str]]] = None,
        pair_definitions: Optional[Sequence[Any]] = None,
        run_id: Optional[str] = None,
        pair_filter: Optional[Callable[[Tuple[str, str]], bool]] = None,
        pair_id_fn: Optional[Callable[[Tuple[str, str]], str]] = None,
        pair_label_fn: Optional[Callable[[Tuple[str, str]], str]] = None,
        retain_models_in_memory: bool = True,
        model_device: str = "cpu",
        **trainer_kwargs: Any,
    ) -> None:
        classes = _infer_target_classes_from_inputs(
            target_classes=target_classes,
            trainer_kwargs=trainer_kwargs,
            trainer_cls=trainer_cls,
            trainer_args=trainer_args,
            pair_definitions=pair_definitions,
            target_pairs=target_pairs,
        )
        generated_incomplete_classes = _load_generated_incomplete_classes_from_kwargs(dict(trainer_kwargs))
        if generated_incomplete_classes:
            incomplete_set = set(generated_incomplete_classes)
            removed = [cls for cls in classes if cls in incomplete_set]
            if removed:
                logger.warning(
                    "Excluding generated incomplete classes from TrainerSuite target classes: %s. "
                    "Pairs containing these classes will not be built.",
                    removed,
                )
                classes = [cls for cls in classes if cls not in incomplete_set]
        if len(classes) < 2:
            raise ValueError("TrainerSuite requires at least 2 target classes")
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError(f"run_id must be str or None, got {type(run_id).__name__}")
        if model_device not in {"cpu", "cuda"}:
            raise ValueError("model_device must be 'cpu' or 'cuda'")
        if not issubclass(trainer_cls, Trainer):
            raise TypeError("trainer_cls must be a Trainer subclass")

        self.trainer_cls = trainer_cls
        self._trainer_args = trainer_args
        self._trainer_kwargs = dict(trainer_kwargs)
        self.target_classes = classes
        self.target_pairs = _normalize_target_pairs(target_pairs)
        if generated_incomplete_classes and self.target_pairs is not None:
            incomplete_set = set(generated_incomplete_classes)
            skipped = [pair for pair in self.target_pairs if pair[0] in incomplete_set or pair[1] in incomplete_set]
            if skipped:
                logger.warning("Skipping TrainerSuite target pairs with generated incomplete classes: %s", skipped)
            self.target_pairs = [
                pair for pair in self.target_pairs
                if pair[0] not in incomplete_set and pair[1] not in incomplete_set
            ]
        self._input_pair_definitions = _normalize_pair_definitions(
            pair_definitions,
            pair_id_fn=pair_id_fn or _default_child_id,
            pair_label_fn=pair_label_fn or _default_child_label,
        )
        self.run_id = run_id
        self.pair_filter = pair_filter
        self.pair_id_fn = pair_id_fn or _default_child_id
        self.pair_label_fn = pair_label_fn or _default_child_label
        self.retain_models_in_memory = bool(retain_models_in_memory)
        self.model_device = model_device

        self.pairs: List[Tuple[str, str]] = []
        self.pair_by_id: Dict[str, Tuple[str, str]] = {}
        self.pair_definitions: Dict[str, SuitePairDefinition] = {}
        self.trainers: Dict[str, Trainer] = {}
        self.label_mapping: Dict[str, str] = {}
        self._models: Dict[str, Any] = {}
        self._shared_base_model: Optional[Any] = None
        self._shared_tokenizer: Optional[Any] = None
        self._shared_model_key: Optional[str] = None

        candidate_pair_definitions = self._resolve_pair_definitions()
        if generated_incomplete_classes:
            incomplete_set = set(generated_incomplete_classes)
            filtered_definitions: List[SuitePairDefinition] = []
            skipped = 0
            for definition in candidate_pair_definitions:
                filtered = _filter_generated_incomplete_pair_definition(definition, incomplete_set)
                if filtered is None:
                    skipped += 1
                    continue
                filtered_definitions.append(filtered)
            if skipped:
                logger.warning(
                    "Skipped %s TrainerSuite pair definitions because generated incomplete classes are excluded.",
                    skipped,
                )
            candidate_pair_definitions = filtered_definitions
        if not candidate_pair_definitions:
            raise ValueError(f"{self.__class__.__name__} did not resolve any pair definitions")
        _validate_pair_definitions_against_data(
            trainer_kwargs=self._trainer_kwargs,
            pair_definitions=candidate_pair_definitions,
            trainer_cls=self.trainer_cls,
            trainer_args=self._trainer_args,
        )

        for definition in candidate_pair_definitions:
            pair = (str(definition.target_classes[0]), str(definition.target_classes[1]))
            if self.pair_filter is not None and not bool(self.pair_filter(pair)):
                continue
            child_id = str(definition.child_id) if definition.child_id is not None else self.pair_id_fn(pair)
            if child_id in self.trainers:
                raise ValueError(f"Duplicate child id generated for pair {pair}: {child_id!r}")
            self.pairs.append(pair)
            self.pair_by_id[child_id] = pair
            self.pair_definitions[child_id] = definition
            self.label_mapping[child_id] = str(definition.label) if definition.label is not None else self.pair_label_fn(pair)
            self.trainers[child_id] = self._build_child_trainer(definition=definition, child_id=child_id)
        self._validate_shared_model_compatibility()

    @abstractmethod
    def _resolve_pair_definitions(self) -> List[SuitePairDefinition]:
        raise NotImplementedError

    def get_pair_definition(self, child_id: str) -> SuitePairDefinition:
        """Return the pair definition for one child trainer.

        Args:
            child_id: Child trainer id.
        """
        return self.pair_definitions[child_id]

    def _child_run_id(self, child_id: str) -> str:
        if self.run_id and str(self.run_id).strip():
            return os.path.join(self.run_id, child_id)
        return child_id

    def _validate_shared_model_compatibility(self) -> None:
        resolved_keys = {
            child_id: _resolved_shared_model_key(trainer)
            for child_id, trainer in self.trainers.items()
        }
        non_empty = {child_id: key for child_id, key in resolved_keys.items() if key}
        if not non_empty:
            self._shared_model_key = None
            return
        unique_keys = sorted(set(non_empty.values()))
        if len(unique_keys) > 1:
            details = ", ".join(f"{child_id}={key}" for child_id, key in sorted(non_empty.items()))
            raise ValueError(
                "TrainerSuite requires all child trainers to resolve to the same base/head model path "
                f"before sharing models across children. Found: {details}"
            )
        self._shared_model_key = unique_keys[0]

    def _resolve_suite_seed_selection(self, seed_selection: Optional[str]) -> str:
        if seed_selection is not None:
            selected = str(seed_selection).strip().lower()
            if selected not in {"best", "all_convergent", "all_tried"}:
                raise ValueError(
                    f"seed_selection must be one of ['best', 'all_convergent', 'all_tried'], got {seed_selection!r}"
                )
            return selected
        for trainer in self.trainers.values():
            if resolve_default_seed_selection(trainer, None) != "best":
                return "all_convergent"
        return "best"

    def _resolve_suite_dispersion(self, dispersion: Optional[str]) -> str:
        if dispersion is not None:
            return str(dispersion).strip().lower()
        for trainer in self.trainers.values():
            args = getattr(trainer, "_training_args", None) or getattr(trainer, "training_args", None)
            if args is not None and getattr(args, "analyze_seed_stability", False):
                return "std"
        return "none"

    def _build_child_trainer(self, *, definition: SuitePairDefinition, child_id: str) -> Trainer:
        child_kwargs = dict(self._trainer_kwargs)
        if "args" in child_kwargs and child_kwargs["args"] is not None:
            child_kwargs["args"] = copy.copy(child_kwargs["args"])
        merge_map = copy.deepcopy(definition.class_merge_map)
        transition_groups = copy.deepcopy(definition.class_merge_transition_groups)
        if "config" in child_kwargs and child_kwargs["config"] is not None:
            cfg = copy.copy(child_kwargs["config"])
            setattr(cfg, "target_classes", list(definition.target_classes))
            setattr(cfg, "class_merge_map", merge_map)
            setattr(cfg, "class_merge_transition_groups", transition_groups)
            setattr(cfg, "run_id", self._child_run_id(child_id))
            child_kwargs["config"] = cfg
        else:
            child_kwargs["class_merge_map"] = merge_map
            child_kwargs["class_merge_transition_groups"] = transition_groups
        child_kwargs["target_classes"] = list(definition.target_classes)
        child_kwargs["run_id"] = self._child_run_id(child_id)
        return self.trainer_cls(*self._trainer_args, **child_kwargs)

    def _build_annotation_trainer(self) -> Trainer:
        annotation_kwargs = dict(self._trainer_kwargs)
        if "args" in annotation_kwargs and annotation_kwargs["args"] is not None:
            annotation_kwargs["args"] = copy.copy(annotation_kwargs["args"])
        if "config" in annotation_kwargs and annotation_kwargs["config"] is not None:
            cfg = copy.copy(annotation_kwargs["config"])
            setattr(cfg, "target_classes", list(self.target_classes))
            setattr(cfg, "run_id", None)
            annotation_kwargs["config"] = cfg
        annotation_kwargs["target_classes"] = list(self.target_classes)
        annotation_kwargs["run_id"] = None
        return self.trainer_cls(*self._trainer_args, **annotation_kwargs)

    def _retain_model(self, child_id: str, trainer: Trainer) -> Any:
        load_kwargs: Dict[str, Any] = {}
        trainer_key = _resolved_shared_model_key(trainer)
        can_share = bool(self._shared_base_model is not None and self._shared_model_key and trainer_key == self._shared_model_key)
        if can_share:
            load_kwargs["base_model"] = self._shared_base_model
        if can_share and self._shared_tokenizer is not None:
            load_kwargs["tokenizer"] = self._shared_tokenizer
        if load_kwargs:
            model = trainer.load_model(trainer.model_path, **load_kwargs)
        else:
            model = trainer.get_model()
        if self._shared_base_model is None and self._shared_model_key and getattr(model, "base_model", None) is not None:
            self._shared_base_model = model.base_model
        if self._shared_tokenizer is None and self._shared_model_key and getattr(model, "tokenizer", None) is not None:
            self._shared_tokenizer = model.tokenizer
        if self.model_device == "cpu" and hasattr(model, "cpu"):
            model = model.cpu()
        elif self.model_device == "cuda" and hasattr(model, "cuda"):
            model = model.cuda()
        self._models[child_id] = model
        if hasattr(trainer, "_model_instance"):
            trainer._model_instance = model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def _effective_label_mapping(
        self,
        label_mapping: Optional[Dict[str, str]] = None,
        *,
        include_defaults: bool = False,
    ) -> Dict[str, str]:
        mapping = dict(self.label_mapping) if include_defaults else {}
        if label_mapping:
            mapping.update({str(k): str(v) for k, v in label_mapping.items()})
        return mapping

    def _transform_pretty_groups(
        self,
        pretty_groups: Optional[Dict[str, List[str]]],
        label_mapping: Dict[str, str],
    ) -> Optional[Dict[str, List[str]]]:
        if pretty_groups is None:
            return None
        return {
            group_name: [label_mapping.get(child_id, child_id) for child_id in child_ids]
            for group_name, child_ids in pretty_groups.items()
        }

    def __len__(self) -> int:
        return len(self.trainers)

    def __iter__(self) -> Iterator[str]:
        return iter(self.trainers)

    def keys(self) -> Iterable[str]:
        return self.trainers.keys()

    def values(self) -> Iterable[Trainer]:
        return self.trainers.values()

    def items(self) -> Iterable[Tuple[str, Trainer]]:
        return self.trainers.items()

    def get_trainer(self, child_id: str) -> Trainer:
        """Return one child trainer by id.

        Args:
            child_id: Child trainer id.
        """
        return self.trainers[child_id]

    def get_trainers(self) -> Dict[str, Trainer]:
        return dict(self.trainers)

    def clear_model_cache(self) -> None:
        self._models.clear()
        for trainer in self.trainers.values():
            if hasattr(trainer, "unload_model"):
                trainer.unload_model()
            elif hasattr(trainer, "_model_instance"):
                trainer._model_instance = None
        self._shared_base_model = None
        self._shared_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_model_refs(
        self,
        *,
        trainer: Optional[Trainer] = None,
        model: Optional[Any] = None,
        child_id: Optional[str] = None,
    ) -> None:
        released_model = model
        if child_id is not None:
            self._models.pop(child_id, None)
        if trainer is not None and hasattr(trainer, "_model_instance"):
            trainer_model = getattr(trainer, "_model_instance", None)
            if released_model is None:
                released_model = trainer_model
        if trainer is not None and hasattr(trainer, "unload_model"):
            trainer.unload_model()
            released_model = None
        elif trainer is not None and hasattr(trainer, "_model_instance"):
            trainer._model_instance = None
        if released_model is not None and hasattr(released_model, "cpu"):
            try:
                released_model.cpu()
            except Exception:
                pass
        del released_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _suite_oom_error(self, action: str, child_id: str) -> RuntimeError:
        return RuntimeError(
            f"TrainerSuite hit an out-of-memory condition while running {action} for child {child_id!r}. "
            "If you are evaluating or iterating over many child trainers, consider using "
            "retain_models_in_memory=False so the suite unloads each finished child model instead of keeping "
            "trainer-local and suite-level references alive across iterations."
        )

    def _load_eval_model(self, trainer: Trainer, *, use_cache: bool = True) -> Any:
        load_kwargs: Dict[str, Any] = {}
        trainer_key = _resolved_shared_model_key(trainer)
        can_share = bool(self._shared_base_model is not None and self._shared_model_key and trainer_key == self._shared_model_key)
        if can_share:
            load_kwargs["base_model"] = self._shared_base_model
        if can_share and self._shared_tokenizer is not None:
            load_kwargs["tokenizer"] = self._shared_tokenizer
        return trainer.load_model(trainer.model_path, **load_kwargs)

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Call a method on every child trainer.

        Args:
            method_name: Name of the child trainer method to call.
            *args: Positional arguments forwarded to the child method.
            **kwargs: Keyword arguments forwarded to the child method.

        Returns:
            Dict mapping ``child_id`` to child method results.
        """
        results: Dict[str, Any] = {}
        total = len(self.trainers)
        for index, (child_id, trainer) in enumerate(self.trainers.items(), start=1):
            child_kwargs = dict(kwargs)
            trainer_key = _resolved_shared_model_key(trainer)
            can_share = bool(self._shared_base_model is not None and self._shared_model_key and trainer_key == self._shared_model_key)
            if method_name == "train" and can_share:
                trainer._model_arg = self._shared_base_model
                if self._shared_tokenizer is not None and "tokenizer" not in child_kwargs:
                    child_kwargs["tokenizer"] = self._shared_tokenizer
            label = getattr(trainer, "run_id", None) or child_id
            logger.info(
                "TrainerSuite %s progress %s/%s: starting child %s",
                method_name,
                index,
                total,
                label,
            )
            try:
                result = getattr(trainer, method_name)(*args, **child_kwargs)
                results[child_id] = result
                logger.info(
                    "TrainerSuite %s progress %s/%s: finished child %s",
                    method_name,
                    index,
                    total,
                    label,
                )
                if method_name == "train":
                    used_cache = bool(getattr(trainer, "_last_train_used_cache", False))
                    if not used_cache and self._shared_base_model is None and self._shared_model_key:
                        try:
                            trained_model = trainer.get_model()
                            if getattr(trained_model, "base_model", None) is not None:
                                self._shared_base_model = trained_model.base_model
                        except Exception:
                            pass
                    if not used_cache and self.retain_models_in_memory:
                        try:
                            self._retain_model(child_id, trainer)
                        except (RuntimeError, MemoryError) as e:
                            raise RuntimeError(
                                "TrainerSuite failed to keep all trained models in memory. "
                                "Retry with retain_models_in_memory=False or call clear_model_cache() "
                                "before loading additional models."
                            ) from e
            except torch.OutOfMemoryError as exc:
                raise self._suite_oom_error(method_name, child_id) from exc
            except Exception:
                logger.exception(
                    "TrainerSuite %s progress %s/%s: child %s failed",
                    method_name,
                    index,
                    total,
                    label,
                )
                raise
            finally:
                skip_release = (
                    method_name == "train"
                    and bool(getattr(trainer, "_last_train_used_cache", False))
                )
                if not self.retain_models_in_memory and not skip_release:
                    self._release_model_refs(trainer=trainer, child_id=child_id)
        return results

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        trainer_attr = getattr(self.trainer_cls, name, None)
        if callable(trainer_attr):
            def _forward(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                return self.call(name, *args, **kwargs)
            _forward.__name__ = name
            _forward.__doc__ = f"Forward {name} to every child trainer and return child_id -> result."
            return _forward
        raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Train every child trainer and return ``child_id -> result``.

        Args:
            *args: Positional arguments forwarded to child ``train``.
            **kwargs: Keyword arguments forwarded to child ``train``.
        """
        return self.call("train", *args, **kwargs)

    def annotate_data(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Annotate the full shared suite dataset once using the shared base model and full class set.

        This intentionally does not run one annotation pass per child trainer because all suite
        children share the same base model and underlying data source.

        Args:
            *args: Positional arguments forwarded to ``Trainer.annotate_data``.
            **kwargs: Keyword arguments forwarded to ``Trainer.annotate_data``.
        """
        trainer = self._build_annotation_trainer()
        df = trainer.annotate_data(*args, **kwargs)
        return df

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate every child trainer and return ``child_id -> result``.

        Args:
            *args: Positional arguments forwarded to child ``evaluate``.
            **kwargs: Keyword arguments forwarded to child ``evaluate``.
        """
        return self.call("evaluate", *args, **kwargs)

    def evaluate_encoder(
        self,
        *args: Any,
        full_eval: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run encoder evaluation for every child trainer.

        Args:
            *args: Positional arguments forwarded to child ``evaluate_encoder``.
            full_eval: If True, include non-target transitions by default. When
                omitted, defaults to True for ``split="test"`` and False
                otherwise.
            **kwargs: Keyword arguments forwarded to child ``evaluate_encoder``.
        """
        if (
            "include_other_classes" not in kwargs
            and "use_all_transitions" not in kwargs
            and kwargs.get("transition_selection") is None
        ):
            split = kwargs.get("split", "test")
            resolved_full_eval = (str(split).lower() == "test") if full_eval is None else bool(full_eval)
            kwargs["include_other_classes"] = resolved_full_eval
        results: Dict[str, Any] = {}
        for child_id, trainer in self.trainers.items():
            child_kwargs = dict(kwargs)
            eval_model = None
            use_cache_requested = bool(child_kwargs.get("use_cache", True))
            if "model_with_gradiend" not in child_kwargs and not use_cache_requested:
                eval_model = self._load_eval_model(trainer, use_cache=bool(child_kwargs.get("use_cache", True)))
                child_kwargs["model_with_gradiend"] = eval_model
            try:
                results[child_id] = trainer.evaluate_encoder(*args, **child_kwargs)
            except torch.OutOfMemoryError as exc:
                raise self._suite_oom_error("evaluate_encoder", child_id) from exc
            finally:
                if not self.retain_models_in_memory:
                    self._release_model_refs(trainer=trainer, model=eval_model, child_id=child_id)
                else:
                    if eval_model is not None:
                        del eval_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return results

    def evaluate_decoder(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run decoder evaluation for every child trainer.

        Args:
            *args: Positional arguments forwarded to child ``evaluate_decoder``.
            **kwargs: Keyword arguments forwarded to child ``evaluate_decoder``.
        """
        return self.call("evaluate_decoder", *args, **kwargs)

    def get_models(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        load_missing: bool = True,
        use_cache: bool = True,
        seed_selection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load suite child models for comparison or plotting.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            load_missing: If True, load models not already retained in memory.
            use_cache: Forwarded to child model loading.
            seed_selection: Optional seed selection for multi-seed child runs.
        """
        effective_labels = self._effective_label_mapping(label_mapping, include_defaults=False)
        models: Dict[str, Any] = {}
        for child_id, trainer in self.trainers.items():
            label = effective_labels.get(child_id, child_id)
            resolved_selection = resolve_default_seed_selection(trainer, seed_selection)
            if resolved_selection == "best":
                model = self._models.get(child_id)
                if model is None and load_missing:
                    load_kwargs: Dict[str, Any] = {}
                    if self._shared_base_model is not None:
                        load_kwargs["base_model"] = self._shared_base_model
                    if self._shared_tokenizer is not None:
                        load_kwargs["tokenizer"] = self._shared_tokenizer
                    model = trainer.get_model(use_cache=use_cache, **load_kwargs)
                    if self.retain_models_in_memory:
                        try:
                            model = self._retain_model(child_id, trainer)
                        except (RuntimeError, MemoryError) as e:
                            raise RuntimeError(
                                "TrainerSuite failed to keep all models in memory while loading them. "
                                "Retry with retain_models_in_memory=False."
                            ) from e
                    elif self.model_device == "cpu" and hasattr(model, "cpu"):
                        model = model.cpu()
                if model is not None:
                    models[label] = model
                continue

            grouped_models, self._shared_base_model, self._shared_tokenizer = load_seed_model_group(
                trainer,
                selection=resolved_selection,
                shared_base_model=self._shared_base_model,
                shared_tokenizer=self._shared_tokenizer,
            )
            if self.model_device == "cpu":
                grouped_models = [m.cpu() if hasattr(m, "cpu") else m for m in grouped_models]
            elif self.model_device == "cuda":
                grouped_models = [m.cuda() if hasattr(m, "cuda") else m for m in grouped_models]
            if grouped_models:
                models[label] = grouped_models
        return models

    def compute_similarity_matrix(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute pairwise similarity across child GRADIEND models.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            use_cache: Forwarded to child model loading.
            **kwargs: Forwarded to ``compute_similarity_matrix``.
        """
        seed_selection = kwargs.pop("seed_selection", None)
        models = self.get_models(
            label_mapping=label_mapping,
            use_cache=use_cache,
            seed_selection=seed_selection,
        )
        return compute_similarity_matrix(models, **kwargs)

    def compute_grouped_similarity_matrices(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute pairwise similarity matrices grouped by model component.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            use_cache: Forwarded to child model loading.
            **kwargs: Forwarded to grouped similarity computation.
        """
        models = self.get_models(
            label_mapping=label_mapping,
            use_cache=use_cache,
            seed_selection=kwargs.pop("seed_selection", None),
        )
        return compute_grouped_similarity_matrices(models, **kwargs)

    def compute_cross_encoding_matrix(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        full_eval: bool = True,
        run_evaluation: bool = True,
        allow_incomplete: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute cross-encoding matrix for positive-pair suites.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            full_eval: Whether child encoder evaluation should include all
                available transitions.
            run_evaluation: Whether to run missing encoder evaluation first.
            allow_incomplete: If True, tolerate missing child encoder results.
            **kwargs: Forwarded by subclasses to cross-encoding computation.
        """
        raise NotImplementedError(
            "compute_cross_encoding_matrix is only available for positive-pair suites. "
            "Use PositiveTrainerSuite for true/false-style pair semantics."
        )

    def plot_cross_encoding_heatmap(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        pretty_groups: Optional[Dict[str, List[str]]] = None,
        split: str = "test",
        max_size: Optional[int] = None,
        use_cache: bool = True,
        metric: str = "positive_mean",
        full_eval: bool = True,
        run_evaluation: bool = True,
        allow_incomplete: bool = False,
        seed_selection: Optional[str] = None,
        seed_aggregate: str = "mean",
        dispersion: Optional[str] = None,
        normalize: bool = False,
        order: Any = "input",
        cluster: bool = False,
        row_label_mapping: Optional[Dict[str, str]] = None,
        column_label_mapping: Optional[Dict[str, str]] = None,
        **plot_kwargs: Any,
    ) -> Dict[str, Any]:
        """Plot positive-pair cross-encoding heatmap.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            pretty_groups: Optional display groups for rows/columns.
            split: Encoder split used when running evaluation.
            max_size: Optional encoder-evaluation cap.
            use_cache: Whether to use child evaluation/model caches.
            metric: Cross-encoding metric to plot.
            full_eval: Whether child encoder evaluation includes all transitions.
            run_evaluation: Whether to run missing encoder evaluation first.
            allow_incomplete: If True, tolerate missing child encoder results.
            seed_selection: Optional seed selection for multi-seed children.
            seed_aggregate: Aggregate used for seed-level cross encoding.
            dispersion: Optional seed dispersion statistic.
            normalize: If True, normalize rows by diagonal values.
            order: Heatmap ordering strategy or explicit order.
            cluster: If True, cluster heatmap rows/columns.
            row_label_mapping: Optional row display labels.
            column_label_mapping: Optional column display labels.
            **plot_kwargs: Forwarded to comparison heatmap plotting.
        """
        raise NotImplementedError(
            "plot_cross_encoding_heatmap is only available for positive-pair suites. "
            "Use PositiveTrainerSuite for true/false-style pair semantics."
        )

    def plot_similarity_heatmap(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        pretty_groups: Optional[Dict[str, List[str]]] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Plot pairwise GRADIEND similarity heatmap for suite children.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            pretty_groups: Optional display groups.
            use_cache: Forwarded to child model loading.
            **kwargs: Forwarded to ``plot_similarity_heatmap``.
        """
        effective_labels = self._effective_label_mapping(label_mapping, include_defaults=True)
        kwargs.setdefault("order", "input")
        models = self.get_models(
            label_mapping=effective_labels,
            use_cache=use_cache,
            seed_selection=kwargs.pop("seed_selection", None),
        )
        import gradiend.trainer.suite as suite_api

        return suite_api.plot_similarity_heatmap(
            models,
            pretty_groups=self._transform_pretty_groups(pretty_groups, effective_labels),
            **kwargs,
        )

    def plot_topk_overlap_heatmap(
        self,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        pretty_groups: Optional[Dict[str, List[str]]] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Plot top-k overlap heatmap for suite child GRADIEND models.

        Args:
            label_mapping: Optional child-id to display-label mapping.
            pretty_groups: Optional display groups.
            use_cache: Forwarded to child model loading.
            **kwargs: Forwarded to ``plot_topk_overlap_heatmap``.
        """
        effective_labels = self._effective_label_mapping(label_mapping, include_defaults=True)
        kwargs.setdefault("order", "input")
        models = self.get_models(
            label_mapping=effective_labels,
            use_cache=use_cache,
            seed_selection=kwargs.pop("seed_selection", None),
        )
        return plot_topk_overlap_heatmap(
            models,
            pretty_groups=self._transform_pretty_groups(pretty_groups, effective_labels),
            **kwargs,
        )


