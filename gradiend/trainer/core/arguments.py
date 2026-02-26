"""
Training arguments for GRADIEND Trainer (HF-like API).
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Any, List, Dict

import torch
import torch.nn as nn

from gradiend.trainer.core.pruning import PostPruneConfig, PrePruneConfig


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

    use_cache: bool = False
    """If True, skip when output path exists (training: model dir; encoder: CSV; etc.). Use False to recompute/retrain."""

    add_identity_for_other_classes: bool = False
    """If True, add identity (factual==alternative) examples for classes not in the target classes used for training."""

    # ----- GRADIEND interpretation -----
    source: str = "alternative"
    """Source for GRADIEND input: 'factual', 'alternative', or 'diff'."""

    target: str = "diff"
    """Target for GRADIEND output: 'factual', 'alternative', or 'diff'."""

    # ----- Training loop -----
    train_batch_size: int = 32
    """Batch size for training (single-device)."""

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

    seed_selection_eval_max_size: Optional[int] = None
    """Max samples for encoder evaluation when selecting the best seed. None = use encoder_eval_max_size."""

    decoder_eval_max_size_training_like: Optional[int] = None
    """Max samples for decoder training-like evaluation data. None = use default behavior."""

    decoder_eval_max_size_neutral: Optional[int] = None
    """Max samples for decoder neutral evaluation data (also LMS text cap). None = use default behavior."""

    decoder_eval_lrs: Optional[List[float]] = None
    """Learning rates for decoder grid search. None = DecoderEvaluator defaults ([1e-2, 1e-3, 1e-4, 1e-5])."""

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

    model_use_cache: bool = False
    """When False (default), pass use_cache=False to decoder model forward during training (KV cache disabled).
    Use True only for inference/generation. Decoder-only MLM head training respects this via train_decoder_only_mlm_head."""

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

    torch_dtype: Optional[torch.dtype] = None
    """dtype for model; None = torch.float32."""

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
    """Optional additional convergence criterion: minimum absolute mean encoded value (e.g. abs_mean_by_type['training']). When None, only convergent_score_threshold is used. Set to e.g. 0.5 to require strong separation in addition to correlation."""

    seed: Optional[int] = 0
    """Random seed for reproducible runs (default 0). The Trainer sets PyTorch/numpy/Python RNG, CUDA determinism, and CUBLAS/OMP env vars; data pipelines use this as random_state. Also the base for multi-seed runs (seed+i). Pass seed=None for non-deterministic runs. If results still vary, call set_seed(42) at the very start of your script or set env CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP_NUM_THREADS=1 before starting Python."""

    seed_runs_dir: Optional[str] = None
    """Directory for per-seed runs. Defaults to experiment_dir/seeds when experiment_dir is set."""

    keep_seed_runs: bool = False
    """If True, keep all per-seed model directories; otherwise delete model files and keep only metrics."""

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
        if not isinstance(self.use_cache, bool):
            raise TypeError(f"use_cache must be bool, got {type(self.use_cache).__name__}")
        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")
        if not isinstance(self.target, str):
            raise TypeError(f"target must be str, got {type(self.target).__name__}")
        if not isinstance(self.train_batch_size, int):
            raise TypeError(f"train_batch_size must be int, got {type(self.train_batch_size).__name__}")
        if self.train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {self.train_batch_size}")
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

        supported = {"factual", "alternative", "diff"}
        if self.source not in supported:
            raise ValueError(f"source must be one of {supported}, got {self.source!r}")
        if self.target not in supported:
            raise ValueError(f"target must be one of {supported}, got {self.target!r}")
        if self.torch_dtype is None:
            self.torch_dtype = torch.float32
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

        metric = (self.convergent_metric or ("loss" if self.supervised_decoder else "correlation")).lower()
        if metric not in ("correlation", "loss"):
            raise ValueError(f"convergent_metric must be 'correlation' or 'loss', got {metric!r}")
        if metric == "correlation" and self.convergent_score_threshold is None:
            self.convergent_score_threshold = 0.5
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
