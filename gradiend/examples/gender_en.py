"""
Gender English demo using TextPredictionTrainer.

This demonstrates how to use TextPredictionTrainer for gender debiasing with name augmentation.
Names are augmented once before training/evaluation (10 names per template per gender)
instead of using complicated data augmentation at training time.

This example showcases custom decoder evaluation: it overrides evaluate_base_model
to use GENTypes-based name-prediction metrics (paper-style BPI, FPI, MPI; see
https://arxiv.org/abs/2502.01406). Custom metrics are exposed via a summary_extractor
(that adds bpi/fpi/mpi to candidates from raw results) and summary_metrics; the
default SelectionPolicy in compute_metric_summaries then selects the best candidate
per metric. Use target_class='bpi' (etc.) with rewrite_base_model(output_dir=...).
"""

import json
import os
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch import softmax

from gradiend import TextPredictionTrainer, TextPredictionConfig
from gradiend.evaluator.decoder import default_extract_candidates, LMSTimesMetricPolicy
from gradiend.trainer.text import TextModelWithGradiend
from gradiend.trainer.text.prediction.unified_data import UNIFIED_SPLIT
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_lms


def read_gentypes():
    return load_dataset('aieng-lab/gentypes')['train'].to_pandas()


def read_namextend():
    return load_dataset('aieng-lab/namextend')['train'].to_pandas()


def read_namexact(split=None):
    return load_dataset('aieng-lab/namexact', split=split).to_pandas()


def read_geneutral(max_size=None):
    df = load_dataset('aieng-lab/geneutral', trust_remote_code=True, split='train').to_pandas()

    if max_size:
        df = df.head(n=max_size)
    return df


def calculate_average_probability_difference(fairness_dict):
    """Calculate average probability difference between M and F."""
    total_diff = 0.0
    num_texts = len(fairness_dict)
    for text, probs in fairness_dict.items():
        prob_m = probs['M']
        prob_f = probs['F']
        total_diff += abs(prob_m - prob_f)
    total_diff /= num_texts
    return total_diff


def calculate_average_prediction_quality(gender_probabilities):
    """Calculate average prediction quality."""
    num_texts = len(gender_probabilities)
    keys = list(gender_probabilities.values())[0]['M']
    if isinstance(keys, dict):
        total_sums = {key: sum([sum([v[key] for v in probs.values() if isinstance(v, dict)]) for probs in gender_probabilities.values()]) for key in keys}
        averages = {key: total_sums[key] / num_texts for key in keys}
    else:
        total_sum = sum([sum(v for v in probs.values() if isinstance(v, float)) for probs in gender_probabilities.values()])
        averages = total_sum / num_texts
    return averages


def compute_gender_preference_accuracy(current_model: dict) -> float:
    """Compute the accuracy of how often the current model prefers the male gender over the female gender."""
    male_preference_count = 0
    total_texts = len(current_model)
    for text in current_model:
        if isinstance(current_model[text]['M'], dict):
            current_M = current_model[text]['M']['total']
            current_F = current_model[text]['F']['total']
        else:
            current_M = current_model[text]['M']
            current_F = current_model[text]['F']
        if current_M > current_F:
            male_preference_count += 1
    return male_preference_count / total_texts


_token_indices_cache = {}
_gender_mapping_cache = {}


def evaluate_gender_bias_name_predictions(model, tokenizer, text_prefix=None, batch_size=64):
    """
    Evaluate gender bias in name predictions (GENTypes; paper-style BPI/FPI/MPI).
    Showcases custom decoder evaluation: use this in _evaluate_model_for_decoder override.
    """
    data = read_gentypes()
    names_df = read_namextend()

    def preprocess(text):
        return text.lower().replace('ġ', '').replace('Ġ', '').strip()

    tokenizer_names_id = (tokenizer.name_or_path, hash(tuple(names_df['name'].tolist())))
    if tokenizer_names_id in _gender_mapping_cache:
        gender_mapping_he, gender_mapping_she = _gender_mapping_cache[tokenizer_names_id]
    else:
        names_df = names_df.copy()
        names_df['name_lower'] = names_df['name'].str.lower()
        gender_mapping_he = names_df[names_df['gender'] == 'M'].set_index('name_lower')['prob_M'].to_dict()
        gender_mapping_she = names_df[names_df['gender'] == 'F'].set_index('name_lower')['prob_F'].to_dict()
        tokenizer_vocab_lower = [preprocess(k) for k in tokenizer.vocab.keys()]
        gender_mapping_he = {k: v for k, v in gender_mapping_he.items() if k.lower() in tokenizer_vocab_lower}
        gender_mapping_she = {k: v for k, v in gender_mapping_she.items() if k.lower() in tokenizer_vocab_lower}
        _gender_mapping_cache[tokenizer_names_id] = (gender_mapping_he, gender_mapping_she)

    if tokenizer.name_or_path in _token_indices_cache:
        he_token_indices, she_token_indices, he_token_factors, she_token_factors = _token_indices_cache[tokenizer.name_or_path]
    else:
        he_tokens = [name for name in tokenizer.vocab if preprocess(name) in gender_mapping_he]
        he_token_factors = np.array([gender_mapping_he[preprocess(name)] for name in he_tokens])
        she_tokens = [name for name in tokenizer.vocab if preprocess(name) in gender_mapping_she]
        she_token_factors = np.array([gender_mapping_she[preprocess(name.lower())] for name in she_tokens])
        he_token_indices = [tokenizer.vocab[name] for name in he_tokens]
        she_token_indices = [tokenizer.vocab[name] for name in she_tokens]
        _token_indices_cache[tokenizer.name_or_path] = (he_token_indices, she_token_indices, he_token_factors, she_token_factors)

    gender_probabilities = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_texts = []
    for _, record in data.iterrows():
        text = record['text']
        if text_prefix:
            if text.startswith('My friend, [NAME],'):
                text = f'{text_prefix.strip()}, [NAME], {text.removeprefix("My friend, [NAME],").strip()}'
            elif text.startswith('[NAME]'):
                text = f'{text_prefix.strip()}, [NAME], {text.removeprefix("[NAME]").strip()}'
            else:
                text = f'{text_prefix.strip()} {text}'
        masked_text = text.replace("[NAME]", tokenizer.mask_token)
        all_texts.append(masked_text)

    for start_idx in range(0, len(all_texts), batch_size):
        end_idx = min(start_idx + batch_size, len(all_texts))
        batch_texts = all_texts[start_idx:end_idx]
        batch_tokenized_text = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True)
        input_ids = batch_tokenized_text["input_ids"].to(device)
        attention_mask = batch_tokenized_text["attention_mask"].to(device)
        mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        for i in range(len(batch_texts)):
            text = batch_texts[i]
            text_dict = {key: {} for key in {'M', 'F'}}
            masked_index = mask_token_index[i].item()
            predictions = logits[i, masked_index]
            softmax_probs = softmax(predictions, dim=-1).cpu()

            for gender, token_indices, token_factors in [('M', he_token_indices, he_token_factors), ('F', she_token_indices, she_token_factors)]:
                gender_probs = softmax_probs[token_indices] * token_factors
                gender_prob = gender_probs.sum().item()
                text_dict[gender] = gender_prob

            total_sum = sum(text_dict.values())
            factor_M = text_dict['M'] / total_sum if total_sum > 0 else 0
            factor_F = text_dict['F'] / total_sum if total_sum > 0 else 0
            sum_M = text_dict['M']
            sum_F = text_dict['F']
            text_dict['factor_M'] = factor_M
            text_dict['factor_F'] = factor_F
            text_apd = min(1.0, max(0.0, abs(sum_M - sum_F)))
            text_dict['text_bpi'] = (1 - text_apd) * (sum_M + sum_F)
            text_dict['text_mpi'] = (1 - sum_F) * sum_M
            text_dict['text_fpi'] = (1 - sum_M) * sum_F
            gender_probabilities[text] = text_dict

    apd = calculate_average_probability_difference(gender_probabilities)
    _bpi = np.mean([prob['text_bpi'] for prob in gender_probabilities.values()]).item()
    _mpi = np.mean([prob['text_mpi'] for prob in gender_probabilities.values()]).item()
    _fpi = np.mean([prob['text_fpi'] for prob in gender_probabilities.values()]).item()
    prediction_quality = calculate_average_prediction_quality(gender_probabilities)

    avg_prob_m = sum(probs['M'] for probs in gender_probabilities.values()) / len(gender_probabilities)
    avg_prob_f = sum(probs['F'] for probs in gender_probabilities.values()) / len(gender_probabilities)
    he_prob = compute_gender_preference_accuracy(gender_probabilities)
    print(f'P(M)= {avg_prob_m:.4f}, P(F)={avg_prob_f:.4f}, APD={apd:.4f}, BPI={_bpi:.4f}, MPI={_mpi:.4f}, FPI={_fpi:.4f}')

    result = {
        'apd': apd,
        'pq': prediction_quality,
        '_bpi': _bpi,
        '_mpi': _mpi,
        '_fpi': _fpi,
        'avg_prob_m': avg_prob_m,
        'avg_prob_f': avg_prob_f,
        'preference_score': abs(avg_prob_m - avg_prob_f),
        'he_prob': he_prob,
    }
    return result


def augment_gender_data_with_names(
    templates_df: pd.DataFrame,
    names_per_template: int = 10,
) -> pd.DataFrame:
    """
    Augment gender templates with names.
    
    For each template, creates variants with male and female names.
    Replaces [NAME] placeholder with actual names.
    
    Args:
        templates_df: DataFrame with 'text' column containing [NAME] placeholder
        names_per_template: Number of names to use per template per gender
    
    Returns:
        Augmented DataFrame with 'masked', 'label', 'label_class', 'split' columns
    """
    # Load names
    names_df = read_namexact(split='train')
    male_names = names_df[names_df['gender'] == 'M']['name'].unique().tolist()
    female_names = names_df[names_df['gender'] == 'F']['name'].unique().tolist()
    
    # Limit names
    male_names = male_names[:names_per_template * 100]  # More than needed
    female_names = female_names[:names_per_template * 100]
    
    rows = []
    
    for _, template_row in templates_df.iterrows():
        template_text = template_row.get('masked', template_row.get('masked', ''))
        split = template_row.get('split', 'train')

        # Create male variants
        for i, name in enumerate(male_names[:names_per_template]):
            augmented_text = template_text.replace('[NAME]', name).replace('[PRONOUN]', '[MASK]')
            rows.append({
                'masked': augmented_text,
                'label': 'he',
                'label_class': 'M',
                'split': split,
            })
        
        # Create female variants
        for i, name in enumerate(female_names[:names_per_template]):
            augmented_text = template_text.replace('[NAME]', name).replace('[PRONOUN]', '[MASK]')
            rows.append({
                'masked': augmented_text,
                'label': 'she',
                'label_class': 'F',
                'split': split,
            })
    
    return pd.DataFrame(rows)


def build_gender_trainer(
    model: str,
    names_per_template: int = 10,
    args: Optional[Any] = None,
) -> TextPredictionTrainer:
    """
    Build gender TextPredictionTrainer with name augmentation.

    Args:
        model: Base model name or path (e.g. "roberta-base", "distilbert-base-cased").
        names_per_template: Number of names per template per gender
        args: Optional TrainingArguments (HF-like); pass at construction for Trainer.train().

    Returns:
        TextPredictionTrainer configured for gender debiasing (model at creation time).
    """
    # Load gender templates
    genter_df = load_dataset('aieng-lab/genter', trust_remote_code=True)
    
    # Convert to pandas if needed
    if hasattr(genter_df, 'to_pandas'):
        genter_df = genter_df.to_pandas()
    elif isinstance(genter_df, dict):
        # Multiple splits
        dfs = []
        for split_name, split_ds in genter_df.items():
            df = split_ds.to_pandas()
            df[UNIFIED_SPLIT] = split_name
            dfs.append(df)
        genter_df = pd.concat(dfs, ignore_index=True)
    
    # Augment with names
    augmented_df = augment_gender_data_with_names(
        genter_df,
        names_per_template=names_per_template,
    )
    # Per-class format: one DataFrame per class, column = token (class name or "label").
    # Alternative = other class's token for the pair (M,F); inferred by unified path.
    data_per_class = {
        "M": augmented_df[augmented_df["label_class"] == "M"][["masked", "split", "label"]].copy(),
        "F": augmented_df[augmented_df["label_class"] == "F"][["masked", "split", "label"]].copy(),
    }
    data_per_class["M"].rename(columns={"label": "M"}, inplace=True)
    data_per_class["F"].rename(columns={"label": "F"}, inplace=True)

    neutral_df = read_geneutral(max_size=1000)

    config = TextPredictionConfig(
        run_id="gender_en",
        data=data_per_class,
        target_classes=["M", "F"],
        masked_col="masked",
        split_col="split",
        eval_neutral_data=neutral_df,
    )
    trainer = TextPredictionTrainer(model=model, config=config, args=args)

    # Showcase customizability: override decoder evaluation to use GENTypes + paper-style BPI/FPI/MPI
    def _evaluate_model_for_decoder_gender(
        self,
        model,
        tokenizer,
        training_like_df=None,
        neutral_df=None,
        max_size_training_like=None,
        max_size_neutral=None,
        eval_batch_size=None,
        **kwargs
    ):
        if max_size_training_like is None:
            max_size_training_like = self.config.decoder_eval_lms_max_samples
        if max_size_neutral is None:
            max_size_neutral = self.config.decoder_eval_lms_max_samples
        if training_like_df is None or neutral_df is None:
            training_like_df, neutral_df = self._get_decoder_eval_dataframe(
                tokenizer,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                cached_training_like_df=training_like_df,
                cached_neutral_df=neutral_df,
            )
        gender_results = evaluate_gender_bias_name_predictions(model, tokenizer, text_prefix='My friend, ', batch_size=64)
        ignore_tokens = self.config.decoder_eval_ignore_tokens or []
        if eval_batch_size is None:
            eval_batch_size = self._default_from_training_args(
                None,
                "eval_batch_size",
                fallback=32,
            )
        lms = compute_lms(
            model,
            tokenizer,
            neutral_df['text'].tolist(),
            ignore=ignore_tokens,
            max_texts=max_size_neutral,
            batch_size=eval_batch_size,
        )
        return {'gender_bias_names': gender_results, 'lms': lms}

    trainer.evaluate_base_model = _evaluate_model_for_decoder_gender.__get__(trainer, TextPredictionTrainer)
    return trainer


def compute_gender_metrics(results):
    """Add _bpi, _fpi, _mpi (raw) and bpi, fpi, mpi (= metric * lms) to each entry for extractor/selector."""
    processed = {}
    for k, entry in results.items():
        if k == 'base':
            processed[k] = entry
            continue
        gender_bias = entry['gender_bias_names']
        lms = entry['lms']
        lms_val = lms['lms'] if isinstance(lms, dict) else lms
        _bpi = gender_bias['_bpi']
        _mpi = gender_bias['_mpi']
        _fpi = gender_bias['_fpi']
        processed[k] = {
            **entry,
            '_bpi': _bpi, '_fpi': _fpi, '_mpi': _mpi,
            'bpi': lms_val * _bpi, 'mpi': lms_val * _mpi, 'fpi': lms_val * _fpi,
        }
    return processed


if __name__ == "__main__":
    from gradiend import TrainingArguments

    model_name = "distilbert-base-cased"
    args = TrainingArguments(
        experiment_dir="runs/examples/gender_en",
        train_batch_size=32,
        encoder_eval_max_size=10,
        eval_steps=25,
        num_train_epochs=1,
        max_steps=100,
        source="factual",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-4,
        use_cache=True,
    )

    print("=== Gender English Demo ===")
    trainer = build_gender_trainer(model=model_name, names_per_template=4, args=args)
    print(f"Run: {trainer.run_id}, samples: {len(trainer.combined_data)}")

    trainer.train(model_class=TextModelWithGradiend)
    print(f"Using cached model at {trainer.model_path}" if getattr(trainer, "_last_train_used_cache", False) else f"Model saved at {trainer.model_path}")

    ts = (trainer.get_training_stats() or {}).get("training_stats", {})
    print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")

    enc_eval = trainer.evaluate_encoder(split="test", max_size=100, use_cache=False, return_df=True)
    enc_df = enc_eval.get("encoder_df")
    trainer.plot_encoder_distributions(encoder_df=enc_df, show=True)
    print(f"  encoder: {len(enc_df)} samples")
    enc_metrics = trainer.get_encoder_metrics(encoder_df=enc_df)
    print(f"  encoder metrics: {enc_metrics}")

    # Custom decoder: select by argmax(_bpi * lms), argmax(_fpi * lms), argmax(_mpi * lms) via LMSTimesMetricPolicy
    dec = trainer.evaluate_decoder(
        selector=LMSTimesMetricPolicy(),
        summary_extractor=lambda results: default_extract_candidates(compute_gender_metrics(results)),
        summary_metrics=["_bpi", "_fpi", "_mpi"],
    )
    summary = dec.get("summary", {})
    for key in ("_bpi", "_fpi", "_mpi"):
        if key in summary and isinstance(summary[key], dict):
            b = summary[key]
            print(f"  best {key} (argmax({key}*lms)): value={b.get('value')}, feature_factor={b.get('feature_factor')}, lr={b.get('learning_rate')}")

    # Save model that maximizes _bpi * lms
    changed_path = trainer.rewrite_base_model(
        decoder_results=dec,
        target_class="_bpi",
        output_dir="./output",
    )
    print(f"Changed model: {changed_path}")
