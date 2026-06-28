"""
Training arguments for GRADIEND Trainer (HF-like API).
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional, Callable, Union, Any, List, Dict

import torch
import torch.nn as nn

from gradiend.trainer.core.config import validate_source_target
from gradiend.trainer.core.pruning import PostPruneConfig, PrePruneConfig, _validate_topk


@dataclass
class TrainingArguments:
    """
    Arguments for GRADIEND training (HF Trainer–style, single training_args class).

    Pass to Trainer at construction: Trainer(model=..., training_args=TrainingArguments(...)).
    Used directly by the core training loop.
    """

    # ----- Output -----
    experiment_dir: Optional[str] = None
    """Root directory for this experiment. When set, default paths use subpaths under it (model, encoded_values, etc.). One experiment dir holds one model. Trainer.run_id (when set) is used as subdir under this."""

    output_dir: Optional[str] = None
    """Directory to save the trained model. If None and experiment_dir is set, uses experiment_dir/model (or experiment_dir/run_id/model when Trainer.run_id is set). Otherwise must be set explicitly."""

    use_cache: Union[bool, Literal["always", "only_convergent"]] = False
    """Training checkpoint reuse policy.

    - ``False``: always retrain even when a saved model exists.
    - ``True``: reuse when a saved model exists and matches the training cache fingerprint.
    - ``"always"``: reuse any saved model at the output path (skip fingerprint matching).
    - ``"only_convergent"``: same fingerprint check as ``True``, but only when the saved run
      meets ``min_convergent_seeds`` (per-seed convergence for individual seed dirs;
      aggregate count for the selected model).
    """

    add_identity_for_other_classes: bool = False
    """If True, add identity (factual==alternative) examples for classes not in the target classes used for training."""

    # ----- GRADIEND interpretation -----
    source: str = "alternative"
    """Source for GRADIEND input: 'factual', 'alternative', or 'diff'."""

    target: str = "diff"
    """Target for GRADIEND output: 'factual', 'alternative', or 'diff'."""

    # ----- Training loop -----
    train_batch_size: int = 32
    """Alias/default for base_gradient_batch_size. Does not affect gradiend_batch_size."""

    base_gradient_batch_size: Optional[int] = None
    """Number of raw training examples merged into one base-model loss/backward call, producing one base-gradient vector."""

    gradiend_batch_size: Optional[int] = None
    """Number of base-gradient vectors stacked into one GRADIEND optimizer step. Defaults to 1."""

    precompute_gradient_batches: Optional[bool] = None
    """Whether to precompute the next gradient row asynchronously.
    None (default): auto-enable only when the base model is sharded and multiple CUDA devices
    are available. False: never precompute. True: always precompute (thread-safe via
    ModelWithGradiend.exclusive_base_gradient_access during base forward/backward)."""

    precompute_gradient_buffer_size: int = 1
    """Number of already-computed gradient rows to keep in the asynchronous precompute queue."""

    gradient_timing_steps: int = 0
    """If > 0, log timing for gradient-row creation every N rows."""

    runtime_monitor: bool = False
    """If True, write persistent JSONL runtime diagnostics under the training output directory."""

    runtime_monitor_interval: float = 5.0
    """Seconds between runtime monitor heartbeat samples."""

    runtime_monitor_system_stats: bool = True
    """If True, runtime monitor heartbeats include CPU/GPU memory stats."""

    train_max_size: Optional[int] = None
    """If set, cap training samples per feature_class_id (downsampling). 
    
    Note: Balancing is handled automatically by the dataset scheduler via oversampling (cycling through 
    balance groups). This parameter primarily reduces total dataset size for memory/performance. 
    None = use all data."""

    learning_rate: float = 1e-5
    """Peak learning rate."""

    num_train_epochs: int = 3
    """Number of training epochs."""

    max_steps: int = -1
    """If > 0, total number of steps; overrides num_train_epochs. -1 = use epochs."""

    weight_decay: float = 1e-2
    """Weight decay for the optimizer."""

    adam_epsilon: float = 1e-8
    """Epsilon for Adam/AdamW."""

    optim: str = "adamw"
    """Optimizer: 'adamw' or 'adam'."""

    criterion: Optional[Union[nn.Module, Any]] = field(default=None, repr=False)
    """Loss function; None = MSELoss()."""

    # ----- Evaluation -----
    eval_strategy: str = "steps"
    """When to run evaluation: 'steps' (every eval_steps) or 'no'."""

    eval_steps: int = 250
    """Run evaluation every eval_steps (when eval_strategy == 'steps')."""

    encoder_eval_train_max_size: Optional[int] = None
    """Max samples for encoder evaluation during training (fast estimate; per-feature_class when available). None = use encoder_eval_max_size."""

    encoder_eval_max_size: Optional[int] = None
    """Max samples for encoder evaluation outside training (e.g. analysis, manual evaluate_encoder). None = use all available."""

    encoder_eval_balance: bool = True
    """If True, balance encoder evaluation data per feature_class_id. If False, use natural class distribution."""

    include_other_classes: bool = False
    """If True, encoder evaluation includes all class transitions in the split, not just the trained target pair. Applies when ``all_classes`` has more than two entries. Affects encoder metrics, encoder plots, and cross-encoding matrices that use ``full_eval=True``. Set on [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] or pass ``include_other_classes=True`` to ``evaluate_encoder()``."""

    seed_selection_eval_max_size: Optional[int] = None
    """Max samples for encoder evaluation when selecting the best seed. None = use encoder_eval_max_size."""

    decoder_eval_max_size_training_like: Optional[int] = None
    """Max samples for decoder training-like evaluation data. None = use default behavior."""

    decoder_eval_max_size_neutral: Optional[int] = None
    """Max samples for decoder neutral evaluation data (also LMS text cap). None = use default behavior."""

    decoder_eval_lrs: Optional[List[float]] = None
    """Learning rates for decoder grid search. None = DecoderEvaluator defaults (1/2/5 grid from 100 to 1e-3)."""

    decoder_eval_feature_factors: Optional[List[float]] = None
    """Feature factors for decoder grid search. None = derive from trainer target classes."""

    eval_batch_size: int = 32
    """Batch size for evaluation."""

    do_eval: bool = True
    """Whether to run evaluation during training."""

    evaluate_fn: Optional[Callable] = field(default=None, repr=False)
    """Custom evaluation function; None = default (encoder correlation on eval data)."""

    # ----- Checkpointing / saving -----
    save_strategy: str = "best"
    """'best' (default): keep only best checkpoint by correlation. 'steps': also save periodic checkpoints every save_steps. 'no': no checkpointing."""

    save_steps: int = 5000
    """Save checkpoint every save_steps when save_strategy == 'steps'."""

    save_only_best: bool = True
    """If True, keep only the best checkpoint (by evaluation correlation)."""

    delete_models: bool = False
    """If True, delete intermediate model files at end (e.g. .bin). This can be used to save disk space if you only care about metrics and not the model itself. Does not delete the whole model directory, which may contain other files (e.g. config, pre/post-prune results)."""

    # ----- Model / GRADIEND -----
    trust_remote_code: bool = False
    """If True, pass trust_remote_code=True when loading models/tokenizers from Hugging Face (e.g. for EuroBERT)."""

    dataset_trust_remote_code: Optional[bool] = None
    """Optional trust_remote_code value for HuggingFace datasets.load_dataset. None means do not pass the keyword."""

    model_use_cache: bool = False
    """When False (default), pass use_cache=False to decoder model forward during training (KV cache disabled).
    Use True only for inference/generation. Decoder-only MLM head training respects this via train_decoder_only_mlm_head."""

    prediction_objective: str = "auto"
    """Prediction objective for text-gradient training and decoder probability scoring.
    Supported: ``auto``, ``mlm_mask_token``, ``clm_next_token``, ``clm_mlm_head``,
    ``clm_sequence_cloze``, ``seq2seq_decoder`` (experimental), ``seq2seq_decoder_sequence_cloze`` (experimental),
    ``seq2seq_encoder_mlm``.
    ``auto``: seq2seq models → ``seq2seq_encoder_mlm``; decoder-only → ``clm_next_token`` (or cached
    ``clm_mlm_head`` when a saved head exists); else ``mlm_mask_token``."""

    decoder_mlm_head_epochs: int = 5
    """Epochs used when prediction_objective="clm_mlm_head" has to train the auxiliary head."""

    decoder_mlm_head_batch_size: int = 4
    """Batch size used when prediction_objective="clm_mlm_head" trains the auxiliary head."""

    decoder_mlm_head_lr: float = 1e-4
    """Learning rate used when prediction_objective="clm_mlm_head" trains the auxiliary head."""

    decoder_mlm_head_max_size: Optional[int] = None
    """Optional per-label cap for auxiliary decoder MLM-head training data."""

    decoder_sequence_cloze_rhs_window: int = -1
    """Right-context token window for clm_sequence_cloze / seq2seq_decoder_sequence_cloze scoring and training. -1 uses the full RHS."""

    params: Optional[List[str]] = None
    """If set, only these parameter names or wildcards are included in the GRADIEND param map when building from a base model. None = include all backbone parameters (default). Enables future params selection processes."""

    activation_encoder: Optional[str] = None
    """Encoder activation name (e.g. 'tanh', 'gelu', 'relu'). None = model default ('tanh')."""

    activation_decoder: Optional[str] = None
    """Decoder activation name (e.g. 'id', 'tanh'). None = model default ('id')."""

    bias_decoder: Optional[bool] = None
    """Whether the decoder linear layer uses a bias term. None = model default (True)."""

    latent_dim: Optional[int] = None
    """GRADIEND latent dimension (number of features). None = model default (1)."""

    normalize_gradiend: bool = True
    """Whether to normalize GRADIEND encodings during training, i.e., first target class is encoded to +1 and second to -1. This is recommended for enhanced comparability between runs."""

    positive_class: Optional[str] = None
    """Optional canonical positive feature class used for binary cross-encoding comparisons.
    When None, comparison utilities may infer it conservatively from target classes via
    the non_/non- prefix heuristic. Ignored for normal training."""

    torch_dtype: Optional[torch.dtype] = None
    """dtype for model; None = torch.float32."""

    base_model_device_map: Optional[Union[bool, str, Dict[str, Any]]] = None
    """Hugging Face device_map for the base model. None auto-detects large models on >3 GPUs,
    False disables device_map, and strings/dicts are passed through explicitly."""

    base_model_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None
    """Optional Hugging Face max_memory map for base-model device placement.
    When unset and base_model_device_map='auto' on multiple GPUs, GPU0 is reserved for GRADIEND automatically."""

    encoder_decoder_same_device: bool = False
    """If True, place encoder and decoder on the same GPU (cuda:0), giving the base model the rest.
    Useful for large base models with pre-pruning: encoder+decoder are small and can share GPU 0;
    base model can use cuda:1 (2 GPUs) or cuda:2 (3+ GPUs). If False (default),
    encoder and decoder are split across cuda:0 and cuda:1 when 2+ GPUs are available."""

    # ----- Multi-seed training -----
    max_seeds: int = 3
    """Maximum number of seeds to try."""

    min_convergent_seeds: Optional[int] = 1
    """Stop once this many seeds have converged. None = run max_seeds. 0 is invalid."""

    convergent_metric: Optional[str] = None
    """Metric for convergence: "correlation" or "loss". Defaults to correlation unless supervised_decoder."""

    convergent_score_threshold: Optional[float] = None
    """Threshold for convergence. Defaults to 0.6 for correlation; required for loss."""

    convergent_mean_by_class_threshold: Optional[float] = None
    """Optional additional convergence criterion: minimum absolute mean encoded value per target class.

    Default: 0.5 when convergent_metric='correlation'. Set to None to disable the mean-based check and use only
    convergent_score_threshold. When set, convergence requires BOTH |correlation| >= convergent_score_threshold AND
    min(|mean|) over non-zero target classes >= convergent_mean_by_class_threshold at the best checkpoint step. For
    correlation-based convergence, the two non-zero target classes must also have opposite-sign mean encodings
    at the best checkpoint step (their product must be negative)."""

    split_resplit_per_seed: bool = False
    """When split_col is None, re-draw vocabulary-held-out splits per training seed.
    False keeps the same split assignment across multi-seed runs (using TrainingArguments.seed)."""

    split_resplit_strategy: Literal["random", "balanced_cycle"] = "random"
    """Strategy used when split_resplit_per_seed=True.
    ``"random"`` redraws splits from each seed. ``"balanced_cycle"`` rotates
    canonical target groups through train/validation/test across seed indices
    so ratios such as 60/20/20 over five seed slots place each word in train
    three times, validation once, and test once."""

    seed: Optional[int] = 0
    """Random seed for reproducible runs (default 0). The Trainer sets PyTorch/numpy/Python RNG, CUDA determinism, and CUBLAS/OMP env vars; data pipelines use this as random_state. Also the base for multi-seed runs (seed+i). Pass seed=None for non-deterministic runs. If results still vary, call set_seed(42) at the very start of your script or set env CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP_NUM_THREADS=1 before starting Python."""

    seed_runs_dir: Optional[str] = None
    """Directory for per-seed runs. Defaults to experiment_dir/seeds when experiment_dir is set."""

    saved_seed_runs: str = "all_convergent"
    """Multi-seed retention policy: 'best_only', 'all_convergent', or 'all_tried'.
    """

    seed_stability_topk: Optional[Union[int, float]] = 1000
    """Top-k selection used for stability summaries across convergent seeds. None disables the report."""

    seed_stability_part: str = "decoder-weight"
    """Importance part used for convergent-seed top-k stability summaries."""

    analyze_seed_stability: bool = False
    """If True, require at least min_convergent_seeds convergent seeds after multi-seed training
    and forbid saved_seed_runs='best_only'. Multi-seed evaluation uses trainer.multi_seed()."""

    # ----- Advanced -----
    supervised_encoder: bool = False
    """If True, train only the GRADIEND encoder: encode(source) vs labels (MSE). Baseline mode."""

    supervised_decoder: bool = False
    """If True, train only the GRADIEND decoder: decoder(labels) vs target gradients (MSE). Baseline mode. Cannot be True together with supervised_encoder."""

    use_cached_gradients: bool = False
    """Whether to use cached gradients if available. Using cached gradients speeds up training and evaluation, but leads to exhaustive memory usage (in memory) and/or on disk (in cached files)."""

    # ----- Pre-prune -----
    pre_prune_config: Optional["PrePruneConfig"] = None
    """If set, pre-prune is run automatically before training. The pruned model is kept in memory; training then uses it. No disk save unless you save explicitly."""

    reuse_pre_prune: bool = False
    """If True, cache pre-prune keep_idx under experiment_dir/cache/pre_prune for reuse across seeds in one train() call. Cache is removed when training finishes."""

    fail_on_non_convergence: bool = False
    """If True, raise when training finishes and convergent_count < min_convergent_seeds (requires min_convergent_seeds > 0)."""

    highlight_non_convergence: bool = True
    """If True, append a non-convergence marker (✝) to plot/tick labels for non-converged runs."""

    # ----- Post-prune -----
    post_prune_config: Optional["PostPruneConfig"] = None
    """If set, post-prune is run automatically after training. The pruned model is kept in memory for subsequent evaluation. No disk save unless you save explicitly."""

    # ----- Extra -----
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Type checks for key scalar parameters
        if self.experiment_dir is not None and not isinstance(self.experiment_dir, str):
            raise TypeError(f"experiment_dir must be str or None, got {type(self.experiment_dir).__name__}")
        if self.output_dir is not None and not isinstance(self.output_dir, str):
            raise TypeError(f"output_dir must be str or None, got {type(self.output_dir).__name__}")
        from gradiend.trainer.core.cache_policy import normalize_use_cache

        normalize_use_cache(self.use_cache)
        if not isinstance(self.reuse_pre_prune, bool):
            raise TypeError(f"reuse_pre_prune must be bool, got {type(self.reuse_pre_prune).__name__}")
        if not isinstance(self.fail_on_non_convergence, bool):
            raise TypeError(f"fail_on_non_convergence must be bool, got {type(self.fail_on_non_convergence).__name__}")
        if not isinstance(self.highlight_non_convergence, bool):
            raise TypeError(
                f"highlight_non_convergence must be bool, got {type(self.highlight_non_convergence).__name__}"
            )
        if not isinstance(self.analyze_seed_stability, bool):
            raise TypeError(
                f"analyze_seed_stability must be bool, got {type(self.analyze_seed_stability).__name__}"
            )
        if not isinstance(self.split_resplit_per_seed, bool):
            raise TypeError(
                f"split_resplit_per_seed must be bool, got {type(self.split_resplit_per_seed).__name__}"
            )
        if self.split_resplit_strategy not in {"random", "balanced_cycle"}:
            raise ValueError(
                "split_resplit_strategy must be 'random' or 'balanced_cycle', "
                f"got {self.split_resplit_strategy!r}"
            )
        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")
        if not isinstance(self.target, str):
            raise TypeError(f"target must be str, got {type(self.target).__name__}")
        if not isinstance(self.train_batch_size, int):
            raise TypeError(f"train_batch_size must be int, got {type(self.train_batch_size).__name__}")
        if self.train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {self.train_batch_size}")
        if self.base_gradient_batch_size is None:
            self.base_gradient_batch_size = self.train_batch_size
        if self.gradiend_batch_size is None:
            self.gradiend_batch_size = 1
        if not isinstance(self.base_gradient_batch_size, int):
            raise TypeError(
                f"base_gradient_batch_size must be int or None, got {type(self.base_gradient_batch_size).__name__}"
            )
        if self.base_gradient_batch_size < 1:
            raise ValueError(f"base_gradient_batch_size must be >= 1, got {self.base_gradient_batch_size}")
        if not isinstance(self.gradiend_batch_size, int):
            raise TypeError(
                f"gradiend_batch_size must be int or None, got {type(self.gradiend_batch_size).__name__}"
            )
        if self.gradiend_batch_size < 1:
            raise ValueError(f"gradiend_batch_size must be >= 1, got {self.gradiend_batch_size}")
        if self.precompute_gradient_batches is not None and not isinstance(self.precompute_gradient_batches, bool):
            raise TypeError(
                "precompute_gradient_batches must be bool or None, "
                f"got {type(self.precompute_gradient_batches).__name__}"
            )
        if not isinstance(self.precompute_gradient_buffer_size, int):
            raise TypeError(
                "precompute_gradient_buffer_size must be int, "
                f"got {type(self.precompute_gradient_buffer_size).__name__}"
            )
        if self.precompute_gradient_buffer_size < 1:
            raise ValueError(
                f"precompute_gradient_buffer_size must be >= 1, got {self.precompute_gradient_buffer_size}"
            )
        if not isinstance(self.gradient_timing_steps, int):
            raise TypeError(f"gradient_timing_steps must be int, got {type(self.gradient_timing_steps).__name__}")
        if self.gradient_timing_steps < 0:
            raise ValueError(f"gradient_timing_steps must be >= 0, got {self.gradient_timing_steps}")
        if not isinstance(self.runtime_monitor, bool):
            raise TypeError(f"runtime_monitor must be bool, got {type(self.runtime_monitor).__name__}")
        if not isinstance(self.runtime_monitor_interval, (int, float)):
            raise TypeError(
                f"runtime_monitor_interval must be int or float, got {type(self.runtime_monitor_interval).__name__}"
            )
        if float(self.runtime_monitor_interval) < 0:
            raise ValueError(f"runtime_monitor_interval must be >= 0, got {self.runtime_monitor_interval}")
        self.runtime_monitor_interval = float(self.runtime_monitor_interval)
        if not isinstance(self.runtime_monitor_system_stats, bool):
            raise TypeError(
                f"runtime_monitor_system_stats must be bool, got {type(self.runtime_monitor_system_stats).__name__}"
            )
        if self.train_max_size is not None and not isinstance(self.train_max_size, int):
            raise TypeError(f"train_max_size must be int or None, got {type(self.train_max_size).__name__}")
        if self.train_max_size is not None and self.train_max_size < 0:
            raise ValueError(f"train_max_size must be >= 0, got {self.train_max_size}")
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(f"learning_rate must be float, got {type(self.learning_rate).__name__}")
        if not isinstance(self.num_train_epochs, int):
            raise TypeError(f"num_train_epochs must be int, got {type(self.num_train_epochs).__name__}")
        if not isinstance(self.max_steps, int):
            raise TypeError(f"max_steps must be int, got {type(self.max_steps).__name__}")
        if not isinstance(self.eval_steps, int):
            raise TypeError(f"eval_steps must be int, got {type(self.eval_steps).__name__}")
        if not isinstance(self.eval_batch_size, int):
            raise TypeError(f"eval_batch_size must be int, got {type(self.eval_batch_size).__name__}")
        if self.eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if not isinstance(self.do_eval, bool):
            raise TypeError(f"do_eval must be bool, got {type(self.do_eval).__name__}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError(f"seed must be int or None, got {type(self.seed).__name__}")

        validate_source_target("source", self.source)
        validate_source_target("target", self.target)
        if self.torch_dtype is None:
            self.torch_dtype = torch.float32
        if self.base_model_device_map is not None and self.base_model_device_map is not False and not isinstance(self.base_model_device_map, (str, dict)):
            raise TypeError(
                "base_model_device_map must be None, False, a string such as 'auto', or a device-map dict; "
                f"got {type(self.base_model_device_map).__name__}"
            )
        if self.base_model_max_memory is not None and not isinstance(self.base_model_max_memory, dict):
            raise TypeError(
                "base_model_max_memory must be None or a max-memory dict; "
                f"got {type(self.base_model_max_memory).__name__}"
            )
        if self.criterion is None:
            self.criterion = nn.MSELoss()
        if self.supervised_encoder and self.supervised_decoder:
            raise ValueError(
                "Cannot set both supervised_encoder and supervised_decoder. "
                "Run two separate train() calls: train(supervised_encoder=True) then train(supervised_decoder=True)."
            )
        if self.supervised_encoder and self.target is not None:
            self.target = None  # encoder baseline doesn't use target
        if self.max_seeds is None:
            self.max_seeds = 3
        if not isinstance(self.max_seeds, int) or self.max_seeds < 1:
            raise ValueError(f"max_seeds must be a positive int, got {self.max_seeds!r}")
        if self.min_convergent_seeds == 0:
            raise ValueError("min_convergent_seeds=0 is not allowed. Use None to run max_seeds.")
        if self.min_convergent_seeds is not None:
            if not isinstance(self.min_convergent_seeds, int) or self.min_convergent_seeds < 0:
                raise ValueError("min_convergent_seeds must be a positive int or None.")
            if self.min_convergent_seeds > self.max_seeds:
                raise ValueError("min_convergent_seeds cannot exceed max_seeds.")
        if not isinstance(self.saved_seed_runs, str):
            raise TypeError(f"saved_seed_runs must be str, got {type(self.saved_seed_runs).__name__}")
        self.saved_seed_runs = str(self.saved_seed_runs).strip().lower()
        supported_saved_seed_runs = {"best_only", "all_convergent", "all_tried"}
        if self.saved_seed_runs not in supported_saved_seed_runs:
            raise ValueError(
                f"saved_seed_runs must be one of {sorted(supported_saved_seed_runs)}, got {self.saved_seed_runs!r}"
            )
        if self.analyze_seed_stability and self.saved_seed_runs == "best_only":
            raise ValueError(
                "analyze_seed_stability=True requires convergent seed checkpoints on disk; "
                "saved_seed_runs='best_only' deletes non-selected seed runs."
            )
        if self.seed_stability_topk is not None:
            _validate_topk(self.seed_stability_topk, "seed_stability_topk")
        if not isinstance(self.seed_stability_part, str) or not self.seed_stability_part.strip():
            raise ValueError("seed_stability_part must be a non-empty string.")

        metric = (self.convergent_metric or ("loss" if self.supervised_decoder else "correlation")).lower()
        if metric not in ("correlation", "loss"):
            raise ValueError(f"convergent_metric must be 'correlation' or 'loss', got {metric!r}")
        if metric == "correlation" and self.convergent_score_threshold is None:
            self.convergent_score_threshold = 0.5
        if metric == "correlation" and self.convergent_mean_by_class_threshold is None:
            self.convergent_mean_by_class_threshold = 0.5
        if metric == "loss" and self.convergent_score_threshold is None:
            raise ValueError("convergent_score_threshold is required when convergent_metric='loss'.")

    def to_dict(self) -> dict:
        """Dict for serialization (excludes callables and nn.Module). Canonical keys only."""
        fields = getattr(type(self), "__dataclass_fields__", {})
        result = {}
        for k in fields:
            if k in ("evaluate_fn", "criterion"):
                continue
            v = getattr(self, k, None)
            if callable(v):
                continue
            if isinstance(v, (nn.Module, torch.dtype)):
                result[k] = str(v) if v is not None else None
            elif k == "pre_prune_config" and v is not None:
                cfg = dataclasses.asdict(v)
                cfg["dataset"] = None  # do not serialize dataset reference
                result[k] = cfg
            elif k == "post_prune_config" and v is not None:
                cfg = dataclasses.asdict(v)
                cfg["mask"] = None  # do not serialize tensor
                result[k] = cfg
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingArguments":
        """Create from dict (e.g. loaded from JSON). Canonical keys only."""
        d = dict(d)
        if "torch_dtype" in d and isinstance(d.get("torch_dtype"), str):
            d["torch_dtype"] = getattr(torch, d["torch_dtype"], torch.float32)
        if "pre_prune_config" in d and isinstance(d.get("pre_prune_config"), dict):
            d["pre_prune_config"] = PrePruneConfig(**d["pre_prune_config"])
        if "post_prune_config" in d and isinstance(d.get("post_prune_config"), dict):
            d["post_prune_config"] = PostPruneConfig(**d["post_prune_config"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        return getattr(self, key, default)

    def __str__(self) -> str:
        parts = [
            f"experiment_dir={self.experiment_dir!r}",
            f"output_dir={self.output_dir!r}",
            f"source={self.source!r}",
            f"target={self.target!r}",
            f"learning_rate={self.learning_rate}",
            f"num_train_epochs={self.num_train_epochs}",
            f"max_steps={self.max_steps}",
            f"seed={self.seed}",
            f"max_seeds={self.max_seeds}",
        ]
        return f"TrainingArguments({', '.join(parts)})"
