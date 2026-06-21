"""
Probe decoder-only "MLM-style" cloze probabilities.

By default this script keeps the whole cloze sentence by prompting a causal
model to answer the blank, then scores candidate answers such as "he" vs.
"she". It is not bidirectional MLM, but it lets the decoder-only model see the
right-hand context inside the prompt.

Example:
    python scripts/probe_decoder_only_cloze.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradiend.util.device import validate_cuda_usable_if_visible


DEFAULT_MODELS = [
    "sshleifer/tiny-gpt2",
]
DEFAULT_MODE = "prompted-cloze"

DEFAULT_EXAMPLES = [
    {
        "text": "The nurse said that [MASK] was exhausted after the shift.",
        "candidates": ["he", "she", "they"],
    },
    {
        "text": "The engineer said that [MASK] fixed the server.",
        "candidates": ["he", "she", "they"],
    },
    {
        "text": "Der Arzt sagte, dass [MASK] heute spaet kommt.",
        "candidates": ["er", "sie"],
    },
    {
        "text": "Die Lehrerin sagte, dass [MASK] heute spaet kommt.",
        "candidates": ["er", "sie"],
    },
]

DEFAULT_PROMPT_TEMPLATE = (
    "Fill in the blank with exactly one of the candidate words.\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


@dataclass
class CandidateScore:
    candidate: str
    continuation_ids: List[int]
    continuation_tokens: List[str]
    logprob: float
    raw_probability: float
    candidate_probability: float


def split_cloze(text: str, mask: str = "[MASK]") -> str:
    if mask not in text:
        raise ValueError(f"Example must contain {mask!r}: {text!r}")
    return text.split(mask, 1)[0]


def cloze_sentence(text: str, mask: str = "[MASK]", blank: str = "___") -> str:
    if mask not in text:
        raise ValueError(f"Example must contain {mask!r}: {text!r}")
    return text.replace(mask, blank, 1)


def ids_for_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def continuation_ids(tokenizer, prefix: str, candidate: str) -> List[int]:
    prefix_ids = ids_for_text(tokenizer, prefix)
    full_ids = ids_for_text(tokenizer, prefix + candidate)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]

    # Some tokenizers merge across the prompt/continuation boundary. Adding a
    # space is a common fallback when users write "[MASK]" without surrounding
    # whitespace.
    full_with_space = ids_for_text(tokenizer, prefix + " " + candidate)
    if full_with_space[: len(prefix_ids)] == prefix_ids:
        return full_with_space[len(prefix_ids) :]

    candidate_ids = ids_for_text(tokenizer, candidate)
    if candidate_ids:
        return candidate_ids
    raise ValueError(f"Could not tokenize candidate {candidate!r}.")


def score_continuation(
    model,
    tokenizer,
    prefix: str,
    candidate_ids: Sequence[int],
    device: torch.device,
) -> float:
    if not candidate_ids:
        return float("-inf")

    prefix_ids = ids_for_text(tokenizer, prefix)
    input_ids = torch.tensor([prefix_ids + list(candidate_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)

    logprob = 0.0
    for offset, token_id in enumerate(candidate_ids):
        pred_pos = len(prefix_ids) + offset - 1
        if pred_pos < 0:
            raise ValueError("Causal scoring requires a non-empty prefix before [MASK].")
        logprob += float(log_probs[0, pred_pos, token_id].item())
    return logprob


def score_example(
    model,
    tokenizer,
    text: str,
    candidates: Sequence[str],
    device: torch.device,
    *,
    mode: str,
    prompt_template: str,
) -> List[CandidateScore]:
    if mode == "prefix-next-token":
        prefix = split_cloze(text)
    elif mode == "prompted-cloze":
        prefix = prompt_template.format(sentence=cloze_sentence(text))
        if not prefix.endswith((" ", "\n", "\t")):
            prefix += " "
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    scored: List[CandidateScore] = []
    for candidate in candidates:
        ids = continuation_ids(tokenizer, prefix, candidate)
        logprob = score_continuation(model, tokenizer, prefix, ids, device)
        scored.append(
            CandidateScore(
                candidate=candidate,
                continuation_ids=list(ids),
                continuation_tokens=tokenizer.convert_ids_to_tokens(list(ids)),
                logprob=logprob,
                raw_probability=math.exp(logprob),
                candidate_probability=0.0,
            )
        )

    norm = torch.softmax(torch.tensor([item.logprob for item in scored]), dim=0).tolist()
    for item, prob in zip(scored, norm):
        item.candidate_probability = float(prob)
    return scored


def parse_examples(values: Iterable[str]) -> List[Dict[str, List[str]]]:
    examples = []
    for value in values:
        if "::" not in value:
            raise ValueError(
                "Custom examples must be formatted as 'sentence with [MASK]::candidate1,candidate2'."
            )
        text, candidates = value.split("::", 1)
        examples.append(
            {
                "text": text.strip(),
                "candidates": [candidate.strip() for candidate in candidates.split(",") if candidate.strip()],
            }
        )
    return examples


def print_scores(model_name: str, example: Dict[str, List[str]], scores: Sequence[CandidateScore]) -> None:
    print(f"\n[{model_name}] {example['text']}")
    print("candidate  cand_prob  raw_prob    logprob    ids        tokens")
    print("-" * 78)
    for score in sorted(scores, key=lambda item: item.candidate_probability, reverse=True):
        ids = ",".join(str(token_id) for token_id in score.continuation_ids)
        tokens = " ".join(token.encode("ascii", "backslashreplace").decode("ascii") for token in score.continuation_tokens)
        print(
            f"{score.candidate:<10} "
            f"{score.candidate_probability:>8.4f}  "
            f"{score.raw_probability:>8.2e}  "
            f"{score.logprob:>8.3f}  "
            f"{ids:<9} {tokens}"
        )


def load_model(model_name: str, device: torch.device, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    cuda_available, _ = validate_cuda_usable_if_visible()
    device = torch.device("cuda" if cuda_available else "cpu")

    print(f"device={device}")
    print("Interpretation: cand_prob is normalized only over the listed candidates.")
    if DEFAULT_MODE == "prompted-cloze":
        print("Mode=prompted-cloze: the full sentence, including text after [MASK], is part of the prompt.")
    else:
        print("Mode=prefix-next-token: text after [MASK] is ignored by the causal model.")

    for model_name in DEFAULT_MODELS:
        try:
            model, tokenizer = load_model(model_name, device, local_files_only=False)
        except Exception as exc:
            print(f"\n[{model_name}] skipped: {exc}")
            continue

        for example in DEFAULT_EXAMPLES:
            scores = score_example(
                model,
                tokenizer,
                example["text"],
                example["candidates"],
                device,
                mode=DEFAULT_MODE,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
            )
            print_scores(model_name, example, scores)


if __name__ == "__main__":
    main()
