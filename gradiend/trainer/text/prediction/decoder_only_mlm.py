"""
Decoder-only MLM Model: Add MLM heads to decoder-only models.

This module provides DecoderModelWithMLMHead, which wraps decoder-only models
(e.g., GPT, Llama) with a custom MLM head. This is useful for using GRADIEND
with decoder-only models, as it allows them to work with masked language modeling
tasks by adding a classification head over target tokens.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from gradiend.model.utils import load_model_weights, save_model_weights
from gradiend.util.device import validate_cuda_usable_if_visible
from gradiend.util.logging import get_logger
from gradiend.util.paths import ensure_writable_dir
from gradiend.util.tqdm_utils import gradiend_tqdm

logger = get_logger(__name__)

PoolingLengthSpec = Union[int, Sequence[int]]


def load_decoder_mlm_head_meta(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    meta_path = os.path.join(path, "config_mlm_head.json")
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_pooling_lengths(pooling_length: PoolingLengthSpec) -> List[int]:
    if isinstance(pooling_length, bool):
        raise TypeError("pooling_length must be int or a sequence of ints, not bool")
    if isinstance(pooling_length, int):
        if pooling_length < 1:
            raise ValueError("pooling_length must be >= 1")
        return [pooling_length]
    values = [int(v) for v in pooling_length]
    if not values:
        raise ValueError("pooling_length sequence must not be empty")
    if any(v < 1 for v in values):
        raise ValueError("pooling_length values must be >= 1")
    return values


def _split_train_val_df(train_df: pd.DataFrame, *, val_frac: float = 0.1, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_frac <= 0.0 or len(train_df) < 4:
        return train_df, train_df.iloc[0:0].copy()
    shuffled = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    if n_val >= len(shuffled):
        n_val = max(1, len(shuffled) // 5)
    val_df = shuffled.iloc[:n_val].reset_index(drop=True)
    train_part = shuffled.iloc[n_val:].reset_index(drop=True)
    if train_part.empty:
        train_part = shuffled.iloc[1:].reset_index(drop=True)
        val_df = shuffled.iloc[:1].reset_index(drop=True)
    return train_part, val_df


def _evaluate_mlm_head_val_loss(
    model: "DecoderModelWithMLMHead",
    tokenizer,
    val_df: pd.DataFrame,
    *,
    batch_size: int,
    max_length: int,
    use_cache: Optional[bool],
    label_class_map: Optional[Dict[str, int]] = None,
) -> float:
    if val_df is None or len(val_df) == 0:
        return float("inf")
    if label_class_map is None and model.target_labels:
        label_class_map = {lab: idx for idx, lab in enumerate(model.target_labels)}
    cuda_available, _ = validate_cuda_usable_if_visible()
    device = "cuda" if cuda_available else "cpu"
    dataset = DataFrameMLMDataset(
        tokenizer, val_df, max_length=max_length, label_class_map=label_class_map
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    count = 0
    use_cache_val = use_cache if use_cache is not None else False
    with torch.no_grad():
        for batch in loader:
            input_ids, attn_mask, labels_b = [b.to(device) for b in batch]
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels_b,
                use_cache=use_cache_val,
            )
            if out.loss is None:
                continue
            total_loss += float(out.loss.item())
            count += 1
    if count == 0:
        return float("inf")
    return total_loss / count


def _save_pooling_length_ablation(
    results: List[Dict[str, float]],
    output_dir: str,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "pooling_length_ablation.json")
    csv_path = os.path.join(output_dir, "pooling_length_ablation.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)
    plot_path = os.path.join(output_dir, "pooling_length_ablation.png")
    try:
        from gradiend.visualizer.plot_optional import _require_matplotlib

        plt = _require_matplotlib()
        lengths = [int(r["pooling_length"]) for r in results]
        losses = [float(r["val_loss"]) for r in results]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(lengths, losses, marker="o")
        ax.set_xlabel("pooling_length")
        ax.set_ylabel("validation loss")
        ax.set_title("Decoder-only MLM head pooling_length selection")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except ImportError:
        plot_path = ""
        logger.warning("matplotlib not available; skipping pooling_length ablation plot")
    return json_path, plot_path or csv_path


_HF_MODEL_LOAD_KWARGS = {
    "device_map",
    "max_memory",
    "low_cpu_mem_usage",
    "torch_dtype",
    "dtype",
    "trust_remote_code",
    "token",
    "revision",
    "cache_dir",
    "local_files_only",
}

_WRAPPER_ONLY_KWARGS = frozenset(
    {
        "target_token_ids",
        "target_labels",
        "mask_token_id",
        "pooling_length",
    }
)


def _config_kwargs(kwargs: Dict) -> Dict:
    return {
        k: v
        for k, v in kwargs.items()
        if k not in {"device_map", "max_memory", "low_cpu_mem_usage", "token"}
        and k not in _WRAPPER_ONLY_KWARGS
    }


def _hf_model_load_kwargs(kwargs: Dict) -> Dict:
    return {
        k: v
        for k, v in kwargs.items()
        if k in _HF_MODEL_LOAD_KWARGS and k not in _WRAPPER_ONLY_KWARGS and v is not False
    }


def _resolve_wrapper_init_kwargs(
    kwargs: Dict,
    *,
    target_token_ids: Optional[List[int]],
    target_labels: Optional[List[str]],
    mask_token_id: Optional[int],
    pooling_length: int,
) -> Tuple[Optional[List[int]], Optional[List[str]], Optional[int], int]:
    if target_labels is None:
        target_labels = kwargs.pop("target_labels", None)
    else:
        kwargs.pop("target_labels", None)
    if target_token_ids is None:
        target_token_ids = kwargs.pop("target_token_ids", None)
    else:
        kwargs.pop("target_token_ids", None)
    if mask_token_id is None:
        mask_token_id = kwargs.pop("mask_token_id", None)
    else:
        kwargs.pop("mask_token_id", None)
    if "pooling_length" in kwargs:
        pooling_length = int(kwargs.pop("pooling_length"))
    return target_token_ids, target_labels, mask_token_id, pooling_length


def _load_selected_checkpoint_tensors(directory: Union[str, os.PathLike], names: List[str]) -> Dict[str, torch.Tensor]:
    st_path = os.path.join(directory, "model.safetensors")
    bin_path = os.path.join(directory, "pytorch_model.bin")
    if not os.path.exists(st_path) and not os.path.exists(bin_path):
        nested = os.path.join(directory, "model")
        st_path = os.path.join(nested, "model.safetensors")
        bin_path = os.path.join(nested, "pytorch_model.bin")

    if os.path.exists(st_path):
        from safetensors import safe_open

        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for name in names:
                if name in available:
                    tensors[name] = f.get_tensor(name)
        return tensors

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        logger.warning(
            "Loading selected tensors from %s requires reading the full .bin checkpoint into CPU memory. "
            "Use safetensors for large decoder-only MLM-head checkpoints.",
            bin_path,
        )
        return {name: state_dict[name] for name in names if name in state_dict}

    raise FileNotFoundError(f"No model weights found in {directory}")


def _decoder_embedding_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return [
        key
        for key in (
            "decoder.transformer.wte.weight",
            "decoder.model.embed_tokens.weight",
        )
        if key in state_dict
    ]


def _custom_head_state_dict(model: "DecoderModelWithMLMHead") -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    keep = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("classifier.") or key in _decoder_embedding_keys(state_dict)
    }
    if not any(key.startswith("classifier.") for key in keep):
        raise ValueError("Could not find classifier weights to save for decoder-only MLM head.")
    return keep


def _resolve_decoder_hidden_size(decoder: nn.Module) -> int:
    configs = []
    config = getattr(decoder, "config", None)
    if config is not None:
        configs.append(config)
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)

    for cfg in configs:
        for attr in ("hidden_size", "n_embd", "d_model", "model_dim", "embed_dim", "embedding_size"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    get_input_embeddings = getattr(decoder, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 2 and int(weight.shape[1]) > 0:
            return int(weight.shape[1])

    raise AttributeError(
        "Could not determine decoder hidden size from config or input embeddings. "
        "Expected one of hidden_size/n_embd/d_model/model_dim/embed_dim/embedding_size, "
        "or a 2D get_input_embeddings().weight tensor."
    )


class DataFrameMLMDataset(Dataset):
    """Dataset from a DataFrame with 'masked' and 'label' columns for decoder-only MLM. Expects [MASK] in masked text."""

    def __init__(
        self,
        tokenizer,
        df: pd.DataFrame,
        max_length: int = 128,
        label_class_map: Optional[Dict[str, int]] = None,
    ):
        self.tokenizer = tokenizer
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.label_class_map = label_class_map or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        masked = str(row["masked"]).strip()
        label = str(row["label"]).strip()
        mask_token = getattr(self.tokenizer, "mask_token", None)
        if mask_token and "[MASK]" in masked and mask_token != "[MASK]":
            masked = masked.replace("[MASK]", mask_token, 1)
        enc = self.tokenizer(
            masked,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        attn_mask = enc.attention_mask.squeeze(0)
        if label not in self.label_class_map:
            raise ValueError(
                f"Label {label!r} is not in the MLM head class map. "
                f"Known labels: {sorted(self.label_class_map)}"
            )
        class_idx = self.label_class_map[label]
        label_ids = torch.tensor([class_idx], dtype=torch.long)
        return input_ids, attn_mask, label_ids



def train_mlm_head(
    base_model: str,
    train_df: pd.DataFrame,
    output_path: str,
    *,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 1e-4,
    pooling_length: PoolingLengthSpec = 3,
    max_length: int = 128,
    trust_remote_code: bool = False,
    use_cache: Optional[bool] = None,
    ablation_dir: Optional[str] = None,
) -> str:
    """
    Train a DecoderModelWithMLMHead on (masked, label) data and save to output_path.

    When ``pooling_length`` is a sequence (e.g. ``range(1, 7)``), run a small validation
  grid search, save ablation JSON/CSV/plot under ``ablation_dir`` (or next to output_path),
    then train the final head on the full dataset with the best length.
    """
    candidates = _normalize_pooling_lengths(pooling_length)
    if len(candidates) == 1:
        _train_mlm_head_single(
            base_model=base_model,
            train_df=train_df,
            output_path=output_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            pooling_length=candidates[0],
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            use_cache=use_cache,
            save=True,
        )
        return output_path

    ablation_root = ablation_dir
    if ablation_root is None:
        ablation_root = os.path.join(os.path.dirname(os.path.normpath(output_path)), "pooling_length_ablation")
    train_part, val_df = _split_train_val_df(train_df)
    results: List[Dict[str, float]] = []
    for pl in candidates:
        model, tokenizer = _train_mlm_head_single(
            base_model=base_model,
            train_df=train_part,
            output_path=None,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            pooling_length=pl,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            use_cache=use_cache,
            save=False,
        )
        val_loss = _evaluate_mlm_head_val_loss(
            model,
            tokenizer,
            val_df,
            batch_size=batch_size,
            max_length=max_length,
            use_cache=use_cache,
            label_class_map={lab: idx for idx, lab in enumerate(model.target_labels or [])},
        )
        results.append({"pooling_length": int(pl), "val_loss": float(val_loss)})
        logger.info("pooling_length=%s validation loss=%.4f", pl, val_loss)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = min(results, key=lambda row: row["val_loss"])
    best_pl = int(best["pooling_length"])
    logger.info("Selected pooling_length=%s (val_loss=%.4f)", best_pl, best["val_loss"])
    _save_pooling_length_ablation(results, ablation_root)

    _train_mlm_head_single(
        base_model=base_model,
        train_df=train_df,
        output_path=output_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        pooling_length=best_pl,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        use_cache=use_cache,
        save=True,
    )
    return output_path


def _train_mlm_head_single(
    base_model: str,
    train_df: pd.DataFrame,
    output_path: Optional[str],
    *,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 1e-4,
    pooling_length: int = 3,
    max_length: int = 128,
    trust_remote_code: bool = False,
    use_cache: Optional[bool] = None,
    save: bool = True,
) -> Union[str, Tuple["DecoderModelWithMLMHead", Any]]:
    """
    Train a DecoderModelWithMLMHead on (masked, label) data and save to output_path.

    Same data source as GRADIEND training. train_df must have columns 'masked' and
    'label'; masked must contain [MASK]. Each unique label string maps to one
    classifier output index (multi-token strings are one class).

    Args:
        use_cache: If False, disable KV cache in model forward (recommended for training).
            If None, defaults to False. Manual override for inference or special cases.

    Returns:
        output_path (saved model + tokenizer).
    """
    if "masked" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train_df must have columns 'masked' and 'label'")
    cuda_available, _ = validate_cuda_usable_if_visible()
    device = "cuda" if cuda_available else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.pad_token = tokenizer.eos_token

    target_labels = train_df["label"].astype(str).str.strip().unique().tolist()
    if len(target_labels) < 2:
        raise ValueError(f"At least two unique labels required; got {len(target_labels)} ({target_labels}).")
    label_class_map = {lab: idx for idx, lab in enumerate(target_labels)}

    logger.info("MLM head target labels (class indices): %s", target_labels)

    dataset = DataFrameMLMDataset(
        tokenizer, train_df, max_length=max_length, label_class_map=label_class_map
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecoderModelWithMLMHead.from_pretrained(
        base_model,
        mask_token_id=tokenizer.mask_token_id,
        target_labels=target_labels,
        pooling_length=pooling_length,
        trust_remote_code=trust_remote_code,
    )
    model.decoder.resize_token_embeddings(len(tokenizer))
    for p in model.decoder.parameters():
        p.requires_grad = False
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        use_cache_val = use_cache if use_cache is not None else False
        for batch in gradiend_tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids, attn_mask, labels_b = [b.to(device) for b in batch]
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels_b,
                use_cache=use_cache_val,
            )
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

    if save:
        if output_path is None:
            raise ValueError("output_path is required when save=True")
        save_path = ensure_writable_dir(output_path)
        model.save_pretrained(save_path, base_model=base_model)
        tokenizer.save_pretrained(save_path)
        return save_path
    return model, tokenizer


@dataclass
class DecoderWithMLMHeadOutput(ModelOutput):
    """Output class for DecoderModelWithMLMHead forward pass."""
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class DecoderModelWithMLMHead(PreTrainedModel):
    """
    Wrapper for decoder-only models with a custom MLM head.

    The auxiliary classifier has one output per target label string. Class index ``i``
  corresponds to ``target_labels[i]`` and is not tied to a single vocabulary token id.
    """

    config_class = PretrainedConfig

    @property
    def uses_label_classes(self) -> bool:
        return bool(self.target_labels)

    def _restricted_head_size(self) -> int:
        if self.target_labels is not None:
            return len(self.target_labels)
        if self.target_token_ids is not None:
            return len(self.target_token_ids)
        return int(self.config.vocab_size)

    def _align_classifier_to_decoder(self):
        if self.target_labels is None and self.target_token_ids is None:
            self.classifier = self.decoder.lm_head
            return

        decoder_param = next(self.decoder.parameters(), None)
        if decoder_param is None:
            return
        self.classifier.to(device=decoder_param.device, dtype=decoder_param.dtype)

    def _align_classifier_to_hidden_states(self, hidden_states: torch.Tensor):
        if self.target_labels is None and self.target_token_ids is None:
            self.classifier = self.decoder.lm_head
            return
        self.classifier.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def __init__(
            self,
            config: PretrainedConfig,
            target_token_ids: Optional[List[int]] = None,
            target_labels: Optional[List[str]] = None,
            pooling_length: int = 3,
            decoder: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        self.decoder = decoder if decoder is not None else AutoModelForCausalLM.from_config(config)

        self.config.model_type = f'{self.decoder.config.model_type}-with-mlm-head'
        self.pooling_length = pooling_length
        self.config.pooling_length = pooling_length
        self.target_labels = list(target_labels) if target_labels else None
        self.target_token_ids = target_token_ids

        if self.target_labels is None and self.target_token_ids is None:
            self.classifier = self.decoder.lm_head
        else:
            hidden_size = _resolve_decoder_hidden_size(self.decoder)
            self.classifier = nn.Linear(hidden_size, self._restricted_head_size())
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_token_ids: Optional[List[int]] = None,
        target_labels: Optional[List[str]] = None,
        mask_token_id: int = None,
        pooling_length: int = 3,
        *model_args,
        **kwargs
    ):
        target_token_ids, target_labels, mask_token_id, pooling_length = _resolve_wrapper_init_kwargs(
            kwargs,
            target_token_ids=target_token_ids,
            target_labels=target_labels,
            mask_token_id=mask_token_id,
            pooling_length=pooling_length,
        )
        meta_path = os.path.join(pretrained_model_name_or_path, "config_mlm_head.json")
        is_custom_checkpoint = os.path.exists(meta_path)

        if is_custom_checkpoint:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **_config_kwargs(kwargs))
            config.mask_token_id = mask_token_id or getattr(config, "mask_token_id", None)

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            if target_labels is None and "target_labels" in meta:
                target_labels = meta["target_labels"]
            if target_token_ids is None and "target_token_ids" in meta:
                target_token_ids = meta["target_token_ids"]

            pooling_length = getattr(config, "pooling_length", pooling_length)
            requested_device_map = kwargs.get("device_map")
            base_model_id = meta.get("base_model") or meta.get("base_model_id")
            init_kwargs = dict(
                target_token_ids=target_token_ids,
                target_labels=target_labels,
                pooling_length=pooling_length,
            )

            if base_model_id:
                hf_kwargs = _hf_model_load_kwargs(kwargs)
                decoder = AutoModelForCausalLM.from_pretrained(
                    base_model_id, *model_args, **hf_kwargs
                )
                model = cls(config, decoder=decoder, **init_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
                if len(tokenizer) != model.decoder.get_input_embeddings().weight.shape[0]:
                    model.decoder.resize_token_embeddings(len(tokenizer))

                selected = _load_selected_checkpoint_tensors(
                    pretrained_model_name_or_path,
                    [
                        "classifier.weight",
                        "classifier.bias",
                        "decoder.transformer.wte.weight",
                        "decoder.model.embed_tokens.weight",
                    ],
                )
                incompatible = model.load_state_dict(selected, strict=False)
                if incompatible.unexpected_keys:
                    raise RuntimeError(
                        "Unexpected tensor(s) while loading decoder-only MLM head: "
                        f"{incompatible.unexpected_keys}"
                    )
                model._align_classifier_to_decoder()
            elif requested_device_map not in (None, False):
                if not base_model_id:
                    raise ValueError(
                        f"Decoder-only MLM-head checkpoint {pretrained_model_name_or_path!r} does not record "
                        "the original base model id, so it cannot be loaded with device_map. Retrain the MLM "
                        "head with a current GRADIEND version, or load with base_model_device_map=False."
                    )
            else:
                state_dict = load_model_weights(pretrained_model_name_or_path)

                if 'decoder.transformer.wte.weight' in state_dict:
                    wte_size = state_dict['decoder.transformer.wte.weight'].size(0)
                elif 'decoder.model.embed_tokens.weight' in state_dict:
                    wte_size = state_dict['decoder.model.embed_tokens.weight'].size(0)
                else:
                    raise ValueError("Unknown model architecture for loading embeddings.")
                model = cls(config, **init_kwargs)
                current_vocab_size = model.decoder.config.vocab_size
                if current_vocab_size != wte_size:
                    model.decoder.resize_token_embeddings(wte_size)
                model.load_state_dict(state_dict, strict=True)
                model._align_classifier_to_decoder()

        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **_config_kwargs(kwargs))
            config.mask_token_id = mask_token_id or getattr(config, "mask_token_id", None)

            model = cls(
                config,
                target_token_ids=target_token_ids,
                target_labels=target_labels,
                pooling_length=pooling_length,
            )

            hf_kwargs = _hf_model_load_kwargs(kwargs)
            model.decoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, *model_args, **hf_kwargs
            )

            if target_labels is None and target_token_ids is None:
                model.classifier = model.decoder.lm_head
            else:
                model._align_classifier_to_decoder()

        return model


    def save_pretrained(self, save_directory: Union[str, os.PathLike], use_safetensors: Optional[bool] = None, **kwargs):
        """Save model and config. Uses model.safetensors when available, else pytorch_model.bin.

        Args:
            save_directory: Directory to save to.
            use_safetensors: If True, require safetensors. If False, force bin. If None, prefer safetensors.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save wrapper config
        self.config.save_pretrained(save_directory)

        # Save wrapper-specific meta
        meta: Dict[str, Any] = {}
        if self.target_labels is not None:
            meta["target_labels"] = self.target_labels
        if self.target_token_ids is not None:
            meta["target_token_ids"] = self.target_token_ids
        base_model = kwargs.pop("base_model", None)
        if base_model is not None:
            meta["base_model"] = str(base_model)
        with open(os.path.join(save_directory, "config_mlm_head.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save model weights (safetensors preferred when available)
        state_dict = _custom_head_state_dict(self) if base_model is not None else self.state_dict()
        save_model_weights(save_directory, state_dict, use_safetensors=use_safetensors)

    def forward(
            self,
            input_ids: Union[torch.LongTensor, List[int], List[List[int]]],
            attention_mask: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            loss_weights=None,
            use_cache: Optional[bool] = None,
    ):
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=self.device)

        # Ensure 2D (batch, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)

        # Check for exactly one [MASK] per sequence
        mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("mask_token_id must be set in the training_args for MLM forward pass.")

        if use_cache is None:
            use_cache = False
        outputs = self.decoder.base_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache
        )
        hidden_states = outputs.last_hidden_state
        self._align_classifier_to_hidden_states(hidden_states)

        mask_pos = (input_ids == mask_token_id).nonzero(as_tuple=False)
        if labels is not None and mask_pos.numel() == 0:
            raise ValueError(
                f"No mask token id {mask_token_id} found in decoder-only MLM input_ids. "
                "Ensure the masked text uses tokenizer.mask_token, or normalize '[MASK]' placeholders before tokenization."
            )

        logits_list = []
        for batch_idx, seq_idx in mask_pos:
            h = hidden_states[batch_idx, seq_idx:seq_idx + self.pooling_length, :].mean(dim=0)
            logits = self.classifier(h)
            logits_list.append(logits)

        if logits_list:
            logits = torch.stack(logits_list, dim=0)
        else:
            logits = torch.empty(
                (0, self._restricted_head_size()),
                device=hidden_states.device
            )

        loss = None
        if labels is not None and logits.numel() > 0:
            loss_fct = nn.CrossEntropyLoss(weight=loss_weights)

            if self.target_labels is None and self.target_token_ids is None:
                selected_labels = labels[mask_pos[:, 0]]
                loss = loss_fct(logits, selected_labels)
            elif self.target_labels is not None:
                if labels.shape[1] == 1:
                    selected_labels = torch.tensor(
                        [
                            int(labels[i, 0].item())
                            for i in range(labels.size(0))
                            for _ in range(int((mask_pos[:, 0] == i).sum().item()))
                        ],
                        device=logits.device,
                    )
                else:
                    selected_labels = labels[mask_pos[:, 0], mask_pos[:, 1]].to(logits.device)
                loss = loss_fct(logits, selected_labels.long())
            else:
                label_map = {tid: idx for idx, tid in enumerate(self.target_token_ids)}
                if labels.shape[1] == 1:
                    selected_labels = torch.tensor([l  for i, l in enumerate(labels) for _ in range((mask_pos[:, 0] == i).sum())])
                else:
                    selected_labels = labels[mask_pos[:, 0], mask_pos[:, 1]]

                mapped_labels = torch.tensor(
                    [label_map.get(l.item(), -1) for l in selected_labels],
                    device=logits.device
                )

                valid_mask = mapped_labels != -1
                if valid_mask.any():
                    logits = logits[valid_mask]
                    mapped_labels = mapped_labels[valid_mask]
                    loss = loss_fct(logits, mapped_labels)
                else:
                    # No valid labels (e.g. [MASK] outside context); zero loss in graph so backward() still fills .grad
                    loss = logits.sum() * 0.0

        return DecoderWithMLMHeadOutput(logits=logits, loss=loss)


    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """
        Return only the decoder backbone parameters.

        The auxiliary MLM classifier and the decoder LM head are prediction heads,
        not GRADIEND parameters. This matters for untied-head models such as Qwen,
        where decoder.lm_head.weight is a separate parameter but is not used by
        this wrapper's MLM loss.
        """
        backbone_param_ids = {
            id(param)
            for param in self.get_gradiend_backbone_module().parameters(recurse=True)
        }
        seen = set()
        for name, param in self.decoder.named_parameters(
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):
            if id(param) not in backbone_param_ids:
                continue
            if remove_duplicate and id(param) in seen:
                continue
            seen.add(id(param))
            yield (f"{prefix}.{name}" if prefix else name), param

    def get_gradiend_backbone_module(self):
        """Backbone used for GRADIEND parameter discovery."""
        return self.decoder.base_model

    def to_original_model(self):
        """
        Return the original model to use for evaluation.

        The specialized MLM head is used for GRADIEND training (bidirectional context),
        but evaluation should use the original decoder-only model (CLM) to measure
        real model change.
        """
        return self.decoder


__all__ = [
    "DataFrameMLMDataset",
    "DecoderModelWithMLMHead",
    "DecoderWithMLMHeadOutput",
    "load_decoder_mlm_head_meta",
    "train_mlm_head",
]
