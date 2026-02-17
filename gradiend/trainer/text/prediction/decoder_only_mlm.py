"""
Decoder-only MLM Model: Add MLM heads to decoder-only models.

This module provides DecoderModelWithMLMHead, which wraps decoder-only models
(e.g., GPT, Llama) with a custom MLM head. This is useful for using GRADIEND
with decoder-only models, as it allows them to work with masked language modeling
tasks by adding a classification head over target tokens.
"""

import json
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import ModelOutput
from typing import Optional, Union, List, Iterator, Tuple
import os

from gradiend.model.utils import load_model_weights, save_model_weights
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

class DataFrameMLMDataset(Dataset):
    """Dataset from a DataFrame with 'masked' and 'label' columns for decoder-only MLM. Expects [MASK] in masked text."""

    def __init__(
        self,
        tokenizer,
        df: pd.DataFrame,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.df = df.reset_index(drop=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        masked = str(row["masked"]).strip()
        label = str(row["label"]).strip()
        enc = self.tokenizer(
            masked,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        attn_mask = enc.attention_mask.squeeze(0)
        label_ids = self.tokenizer(
            label, add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)
        return input_ids, attn_mask, label_ids



def train_mlm_head(
    base_model: str,
    train_df: pd.DataFrame,
    output_path: str,
    *,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 1e-4,
    pooling_length: int = 3,
    max_length: int = 128,
    trust_remote_code: bool = False,
    use_cache: Optional[bool] = None,
) -> str:
    """
    Train a DecoderModelWithMLMHead on (masked, label) data and save to output_path.

    Same data source as GRADIEND training. train_df must have columns 'masked' and
    'label'; masked must contain [MASK], label must be a single token per row.

    Args:
        use_cache: If False, disable KV cache in model forward (recommended for training).
            If None, defaults to False. Manual override for inference or special cases.

    Returns:
        output_path (saved model + tokenizer).
    """
    if "masked" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train_df must have columns 'masked' and 'label'")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.pad_token = tokenizer.eos_token

    # Resolve target token IDs from unique values in the 'label' column (one token per label)
    labels = train_df["label"].astype(str).str.strip().unique().tolist()
    target_token_ids = []
    for lab in labels:
        ids = tokenizer(lab, add_special_tokens=False).get("input_ids", [])
        if len(ids) != 1:
            raise ValueError(
                f"Label '{lab}' must tokenize to a single token (got {len(ids)}). "
                "Decoder-only MLM requires single-token labels."
            )
        target_token_ids.append(ids[0])
    if len(target_token_ids) < 2:
        raise ValueError(f"At least two unique labels required; got {len(target_token_ids)} ({labels}).")

    logger.info("Target token IDs: %s", target_token_ids)

    dataset = DataFrameMLMDataset(tokenizer, train_df, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecoderModelWithMLMHead.from_pretrained(
        base_model,
        mask_token_id=tokenizer.mask_token_id,
        target_token_ids=target_token_ids,
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
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
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

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path



@dataclass
class DecoderWithMLMHeadOutput(ModelOutput):
    """Output class for DecoderModelWithMLMHead forward pass."""
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class DecoderModelWithMLMHead(PreTrainedModel):
    """
    Wrapper for decoder-only models with a custom MLM head.

    This class adds a masked language modeling (MLM) head to decoder-only models,
    allowing them to be used with GRADIEND. The MLM head can be restricted to
    specific target tokens (e.g., ["he", "she"]) or use the full vocabulary.

    Args:
        config: PretrainedConfig for the base decoder model
        target_token_ids: Optional list of token IDs to restrict the MLM head to.
                         If None, uses the full vocabulary via the decoder's lm_head.

    Example:
        >>> from gradiend.trainer.text import DecoderModelWithMLMHead
        >>> model = DecoderModelWithMLMHead.from_pretrained(
        ...     "gpt2",
        ...     target_token_ids=[1234, 5678]  # token IDs for "he" and "she"
        ... )
    """
    config_class = PretrainedConfig

    def __init__(
            self,
            config: PretrainedConfig,
            target_token_ids: Optional[List[int]] = None,
            pooling_length: int = 3,
    ):
        super().__init__(config)
        # Base decoder model
        self.decoder = AutoModelForCausalLM.from_config(config)

        self.config.model_type = f'{self.decoder.config.model_type}-with-mlm-head'
        self.pooling_length = pooling_length
        self.config.pooling_length = pooling_length
        self.target_token_ids = target_token_ids

        if self.target_token_ids is None:
            self.classifier = self.decoder.lm_head
        else:
            hidden_size = self.decoder.config.hidden_size
            self.classifier = nn.Linear(hidden_size, len(self.target_token_ids))
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

        self.init_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_token_ids: Optional[List[int]] = None,
        mask_token_id: int = None,
        pooling_length: int = 3,
        *model_args,
        **kwargs
    ):
        # Detect if this is a custom checkpoint (by presence of our special meta file)
        meta_path = os.path.join(pretrained_model_name_or_path, "config_mlm_head.json")
        is_custom_checkpoint = os.path.exists(meta_path)

        if is_custom_checkpoint:
            # --- Load from custom checkpoint ---
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config.mask_token_id = mask_token_id or getattr(config, "mask_token_id", None)

            # Restore target_token_ids from meta file if not provided
            with open(meta_path) as f:
                meta = json.load(f)

            if target_token_ids is None and "target_token_ids" in meta:
                target_token_ids = meta["target_token_ids"]

            pooling_length = getattr(config, "pooling_length", pooling_length)

            # Init wrapper
            model = cls(config, target_token_ids=target_token_ids, pooling_length=pooling_length)

            # Load saved weights (safetensors preferred when available)
            state_dict = load_model_weights(pretrained_model_name_or_path)

            if 'decoder.transformer.wte.weight' in state_dict:
                wte_size = state_dict['decoder.transformer.wte.weight'].size(0)
            elif 'decoder.model.embed_tokens.weight' in state_dict:
                wte_size = state_dict['decoder.model.embed_tokens.weight'].size(0)
            else:
                raise ValueError("Unknown model architecture for loading embeddings.")
            # resize embeddings if needed
            current_vocab_size = model.decoder.config.vocab_size
            if current_vocab_size != wte_size:
                model.decoder.resize_token_embeddings(wte_size)
            model.load_state_dict(state_dict, strict=True)

        else:
            # --- Load from a standard LM checkpoint (e.g., "gpt2") ---
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config.mask_token_id = mask_token_id or getattr(config, "mask_token_id", None)

            # Init wrapper — this will internally freeze decoder and init head
            model = cls(config, target_token_ids=target_token_ids, pooling_length=pooling_length)

            # Load decoder from the base LM checkpoint (use dtype for HF; torch_dtype is deprecated)
            hf_kwargs = {k if k != "torch_dtype" else "dtype": v for k, v in kwargs.items()}
            model.decoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, *model_args, **hf_kwargs
            )

            # Replace classifier if in full vocab mode
            if target_token_ids is None:
                model.classifier = model.decoder.lm_head

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
        meta = {"target_token_ids": self.target_token_ids}
        with open(os.path.join(save_directory, "config_mlm_head.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save model weights (safetensors preferred when available)
        save_model_weights(save_directory, self.state_dict(), use_safetensors=use_safetensors)

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

        mask_pos = (input_ids == mask_token_id).nonzero(as_tuple=False)

        logits_list = []
        for batch_idx, seq_idx in mask_pos:
            h = hidden_states[batch_idx, seq_idx:seq_idx + self.pooling_length, :].mean(dim=0)
            logits = self.classifier(h)
            logits_list.append(logits)

        if logits_list:
            logits = torch.stack(logits_list, dim=0)
        else:
            logits = torch.empty(
                (0, len(self.target_token_ids) if self.target_token_ids else self.config.vocab_size),
                device=hidden_states.device
            )

        loss = None
        if labels is not None and logits.numel() > 0:
            loss_fct = nn.CrossEntropyLoss(weight=loss_weights)

            if self.target_token_ids is None:
                selected_labels = labels[mask_pos[:, 0]]
                loss = loss_fct(logits, selected_labels)
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
        Override to ensure we return only the decoder parameters and not the head.
        This ensures that this model implementations mimics the parameter structure of
        the underlying decoder-only model.
        """
        return self.decoder.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate
        )

    def to_original_model(self):
        """
        Return the original model to use for evaluation.

        The specialized MLM head is used for GRADIEND training (bidirectional context),
        but evaluation should use the original decoder-only model (CLM) to measure
        real model change.
        """
        return self.decoder


__all__ = ["DataFrameMLMDataset", "DecoderModelWithMLMHead", "DecoderWithMLMHeadOutput", "train_mlm_head"]
