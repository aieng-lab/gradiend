"""
Train a fresh MLM head on top of a frozen encoder-style backbone.

This script is designed for checkpoints such as ``AnReu/math_pretrained_bert``
that provide a pretrained encoder but no MLM head. It creates a standard
``AutoModelForMaskedLM`` checkpoint, copies the encoder weights into it, freezes
the backbone, and trains only the MLM head.

The data pipeline is self-contained:
- default corpus: ``open-web-math/open-web-math``
- optional mixture with ``ddrg/math_text``
- math positions are detected from LaTeX/math spans in the raw text
- masking is span-aware instead of using a heuristic math-token list

Example:
    python scripts/train_frozen_mlm_head.py ^
        --model AnReu/math_pretrained_bert ^
        --output-dir runs/math_pretrained_bert_mlm ^
        --train-token-budget 10000000 ^
        --validation-token-budget 500000 ^
        --math-text-proportion 0.2
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from gradiend import set_seed


INLINE_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)
BRACKET_MATH_RE = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
ENV_MATH_RE = re.compile(
    r"\\begin\{(equation|equation\*|align|align\*|gather|gather\*|multline|multline\*|cases)\}(.+?)\\end\{\1\}",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-backbone MLM head from math corpora.")
    parser.add_argument("--model", type=str, required=True, help="Base Hugging Face model id or path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument(
        "--mask-strategy",
        type=str,
        choices=("all", "math-biased", "math-only"),
        default="math-biased",
        help="Mask all positions uniformly, bias toward detected math spans, or mask only math spans.",
    )
    parser.add_argument(
        "--math-target-ratio",
        type=float,
        default=0.8,
        help="For math-biased masking, target share of masked positions sampled from detected math spans.",
    )
    parser.add_argument(
        "--train-token-budget",
        type=int,
        default=10_000_000,
        help="Approximate number of non-special tokens to sample for training.",
    )
    parser.add_argument(
        "--validation-token-budget",
        type=int,
        default=500_000,
        help="Approximate number of non-special tokens to sample for validation.",
    )
    parser.add_argument(
        "--open-web-math-dataset",
        type=str,
        default="open-web-math/open-web-math",
        help="Primary default training corpus.",
    )
    parser.add_argument(
        "--math-text-dataset",
        type=str,
        default="ddrg/math_text",
        help="Optional secondary corpus mixed in via --math-text-proportion.",
    )
    parser.add_argument(
        "--math-text-proportion",
        type=float,
        default=0.0,
        help="Fraction of sampled tokens drawn from math_text. Default keeps the run open-web-math only.",
    )
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the optional 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc
    return load_dataset


def _detect_math_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pattern in (INLINE_DISPLAY_MATH_RE, BRACKET_MATH_RE, ENV_MATH_RE):
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _offset_overlaps_span(offset: Tuple[int, int], spans: Sequence[Tuple[int, int]]) -> bool:
    start, end = offset
    if end <= start:
        return False
    for span_start, span_end in spans:
        if span_end <= start:
            continue
        if span_start >= end:
            break
        return True
    return False


@dataclass
class PreparedExample:
    input_ids: List[int]
    attention_mask: List[int]
    special_tokens_mask: List[int]
    math_token_mask: List[int]

    @property
    def non_special_token_count(self) -> int:
        return int(sum(1 for is_special in self.special_tokens_mask if not is_special))


def _chunk_document(
    text: str,
    tokenizer: Any,
    max_length: int,
) -> List[PreparedExample]:
    if max_length < 8:
        raise ValueError("--max-length must be at least 8.")

    spans = _detect_math_spans(text)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        verbose=False,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not token_ids:
        return []

    chunk_size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if chunk_size <= 0:
        raise ValueError("--max-length is too small once special tokens are added.")

    examples: List[PreparedExample] = []
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start : start + chunk_size]
        chunk_offsets = offsets[start : start + chunk_size]
        input_ids = list(tokenizer.build_inputs_with_special_tokens(chunk_ids))
        special_tokens_mask = _build_special_tokens_mask_from_input_ids(tokenizer, input_ids)
        attention_mask = [1] * len(input_ids)
        chunk_math_mask: List[int] = []
        offset_idx = 0
        for is_special in special_tokens_mask:
            if is_special:
                chunk_math_mask.append(0)
            else:
                chunk_math_mask.append(1 if _offset_overlaps_span(chunk_offsets[offset_idx], spans) else 0)
                offset_idx += 1
        examples.append(
            PreparedExample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                math_token_mask=chunk_math_mask,
            )
        )
    return examples


class PreparedDataset(Dataset):
    def __init__(self, examples: Sequence[PreparedExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.long),
            "special_tokens_mask": torch.tensor(example.special_tokens_mask, dtype=torch.long),
            "math_token_mask": torch.tensor(example.math_token_mask, dtype=torch.long),
        }


def _build_special_tokens_mask_from_input_ids(tokenizer: Any, input_ids: Sequence[int]) -> List[int]:
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [1 if int(token_id) in special_token_ids else 0 for token_id in input_ids]


class MathSpanMLMCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        mlm_probability: float,
        mask_strategy: str,
        math_target_ratio: float,
    ):
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no mask token. This script expects an encoder-style MLM tokenizer.")
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_strategy = mask_strategy
        self.math_target_ratio = math_target_ratio
        self.vocab_size = len(tokenizer)

    def _sample_positions(
        self,
        valid_positions: torch.Tensor,
        math_positions: torch.Tensor,
    ) -> torch.Tensor:
        valid_list = valid_positions.tolist()
        if not valid_list:
            return torch.empty(0, dtype=torch.long)
        n_to_mask = max(1, int(round(len(valid_list) * self.mlm_probability)))
        math_list = math_positions.tolist()

        if self.mask_strategy == "all":
            chosen = valid_list[torch.randperm(len(valid_list))[: min(n_to_mask, len(valid_list))].tolist()]
            return torch.tensor(sorted(chosen), dtype=torch.long)

        if self.mask_strategy == "math-only":
            if math_list:
                chosen = math_list[torch.randperm(len(math_list))[: min(n_to_mask, len(math_list))].tolist()]
            else:
                chosen = valid_list[torch.randperm(len(valid_list))[: min(n_to_mask, len(valid_list))].tolist()]
            return torch.tensor(sorted(chosen), dtype=torch.long)

        chosen: List[int] = []
        if math_list:
            n_math = min(len(math_list), int(round(n_to_mask * self.math_target_ratio)))
            if n_math > 0:
                perm = torch.randperm(len(math_list))[:n_math].tolist()
                chosen.extend(math_list[idx] for idx in perm)
        remaining = n_to_mask - len(chosen)
        if remaining > 0:
            remaining_pool = [pos for pos in valid_list if pos not in chosen]
            if remaining_pool:
                perm = torch.randperm(len(remaining_pool))[: min(remaining, len(remaining_pool))].tolist()
                chosen.extend(remaining_pool[idx] for idx in perm)
        return torch.tensor(sorted(chosen), dtype=torch.long)

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max(int(example["input_ids"].size(0)) for example in examples)

        def _pad_tensor(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
            if tensor.size(0) == max_len:
                return tensor
            pad = torch.full((max_len - tensor.size(0),), pad_value, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)

        input_ids = torch.stack([_pad_tensor(example["input_ids"], pad_id) for example in examples])
        attention_mask = torch.stack([_pad_tensor(example["attention_mask"], 0) for example in examples])
        special_tokens_mask = torch.stack(
            [_pad_tensor(example["special_tokens_mask"], 1) for example in examples]
        ).bool()
        math_token_mask = torch.stack(
            [_pad_tensor(example["math_token_mask"], 0) for example in examples]
        ).bool()
        labels = torch.full_like(input_ids, fill_value=-100)

        for row_idx in range(input_ids.size(0)):
            valid_positions = torch.where(attention_mask[row_idx].bool() & ~special_tokens_mask[row_idx])[0]
            math_positions = torch.where(
                attention_mask[row_idx].bool() & ~special_tokens_mask[row_idx] & math_token_mask[row_idx]
            )[0]
            selected = self._sample_positions(valid_positions, math_positions)
            if selected.numel() == 0:
                continue

            labels[row_idx, selected] = input_ids[row_idx, selected]
            replace_probs = torch.rand(selected.size(0))
            mask_positions = selected[replace_probs < 0.8]
            random_positions = selected[(replace_probs >= 0.8) & (replace_probs < 0.9)]
            input_ids[row_idx, mask_positions] = self.tokenizer.mask_token_id
            if random_positions.numel() > 0:
                input_ids[row_idx, random_positions] = torch.randint(
                    0,
                    self.vocab_size,
                    (random_positions.size(0),),
                    dtype=torch.long,
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _load_mlm_model(base_model_name: str, trust_remote_code: bool) -> AutoModelForMaskedLM:
    return AutoModelForMaskedLM.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
    )


def _untie_output_embeddings_if_needed(model: AutoModelForMaskedLM) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None or getattr(output_embeddings, "weight", None) is None:
        return False
    if output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr():
        return False
    output_embeddings.weight = torch.nn.Parameter(output_embeddings.weight.detach().clone())
    return True


def _freeze_backbone_only(model: AutoModelForMaskedLM) -> Tuple[int, int]:
    prefix = getattr(model, "base_model_prefix", None)
    if not prefix:
        raise ValueError("Model has no base_model_prefix; cannot separate backbone from head.")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        is_backbone = name.startswith(f"{prefix}.")
        param.requires_grad = not is_backbone
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, total_params


def _iter_dataset_texts(
    dataset_name: str,
    text_column: str,
    split: str,
    seed: int,
    shuffle_buffer_size: int,
) -> Iterator[str]:
    load_dataset = _require_datasets()
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    for row in dataset:
        value = row.get(text_column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            yield text


def _prepare_examples_from_text_stream(
    text_iter: Iterable[str],
    tokenizer: Any,
    token_budget: int,
    max_length: int,
) -> Tuple[List[PreparedExample], int]:
    examples: List[PreparedExample] = []
    seen_tokens = 0
    for text in text_iter:
        doc_examples = _chunk_document(text, tokenizer, max_length=max_length)
        for example in doc_examples:
            if example.non_special_token_count == 0:
                continue
            examples.append(example)
            seen_tokens += example.non_special_token_count
            if seen_tokens >= token_budget:
                return examples, seen_tokens
    return examples, seen_tokens


def _split_token_budget(total_budget: int, secondary_proportion: float) -> Tuple[int, int]:
    secondary_budget = int(round(total_budget * secondary_proportion))
    primary_budget = max(total_budget - secondary_budget, 0)
    return primary_budget, secondary_budget


def _build_dataset(
    *,
    tokenizer: Any,
    primary_dataset_name: str,
    secondary_dataset_name: str,
    secondary_proportion: float,
    text_column: str,
    split: str,
    token_budget: int,
    max_length: int,
    seed: int,
    shuffle_buffer_size: int,
) -> Tuple[PreparedDataset, Dict[str, int]]:
    if not 0.0 <= secondary_proportion <= 1.0:
        raise ValueError("--math-text-proportion must be in [0, 1].")

    primary_budget, secondary_budget = _split_token_budget(token_budget, secondary_proportion)
    examples: List[PreparedExample] = []
    usage: Dict[str, int] = {}

    if primary_budget > 0:
        primary_iter = _iter_dataset_texts(
            primary_dataset_name,
            text_column=text_column,
            split=split,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        primary_examples, primary_tokens = _prepare_examples_from_text_stream(
            primary_iter,
            tokenizer,
            token_budget=primary_budget,
            max_length=max_length,
        )
        examples.extend(primary_examples)
        usage[primary_dataset_name] = primary_tokens

    if secondary_budget > 0:
        secondary_iter = _iter_dataset_texts(
            secondary_dataset_name,
            text_column=text_column,
            split=split,
            seed=seed + 1,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        secondary_examples, secondary_tokens = _prepare_examples_from_text_stream(
            secondary_iter,
            tokenizer,
            token_budget=secondary_budget,
            max_length=max_length,
        )
        examples.extend(secondary_examples)
        usage[secondary_dataset_name] = secondary_tokens

    if not examples:
        raise RuntimeError("No training examples were created from the selected corpora.")

    return PreparedDataset(examples), usage


def _evaluate(model: AutoModelForMaskedLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.mask_token is None:
        raise ValueError(
            f"Tokenizer for {args.model!r} has no mask token. "
            "This script is for encoder-style MLM checkpoints."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.mask_token or tokenizer.eos_token or tokenizer.unk_token

    train_dataset, train_usage = _build_dataset(
        tokenizer=tokenizer,
        primary_dataset_name=args.open_web_math_dataset,
        secondary_dataset_name=args.math_text_dataset,
        secondary_proportion=args.math_text_proportion,
        text_column=args.text_column,
        split="train",
        token_budget=args.train_token_budget,
        max_length=args.max_length,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    val_dataset, val_usage = _build_dataset(
        tokenizer=tokenizer,
        primary_dataset_name=args.open_web_math_dataset,
        secondary_dataset_name=args.math_text_dataset,
        secondary_proportion=args.math_text_proportion,
        text_column=args.text_column,
        split="train",
        token_budget=args.validation_token_budget,
        max_length=args.max_length,
        seed=args.seed + 10_000,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    collator = MathSpanMLMCollator(
        tokenizer,
        mlm_probability=args.mlm_probability,
        mask_strategy=args.mask_strategy,
        math_target_ratio=args.math_target_ratio,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = _load_mlm_model(args.model, args.trust_remote_code)
    untied = _untie_output_embeddings_if_needed(model)
    trainable_params, total_params = _freeze_backbone_only(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training sources: {train_usage}")
    print(f"Validation sources: {val_usage}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Untied output embeddings from frozen input embeddings: {untied}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": args.model,
        "train_sources": train_usage,
        "validation_sources": val_usage,
        "mask_strategy": args.mask_strategy,
        "math_target_ratio": args.math_target_ratio,
        "mlm_probability": args.mlm_probability,
        "train_token_budget": args.train_token_budget,
        "validation_token_budget": args.validation_token_budget,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "untied_output_embeddings": untied,
        "math_span_detection": {
            "inline_display_math": r"\$\$(.+?)\$\$|\$(.+?)\$",
            "bracket_math": r"\\\((.+?)\\\)|\\\[(.+?)\\\]",
            "environment_math": r"\\begin\{equation|equation\*|align|align\*|gather|gather\*|multline|multline\*|cases\}...\n",
        },
    }
    (args.output_dir / "frozen_mlm_head_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved frozen-backbone MLM model to {args.output_dir}")


if __name__ == "__main__":
    main()
