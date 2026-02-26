"""
Core training loop for GRADIEND models.
"""

import gc
import time
import dataclasses
from typing import List, Optional

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from gradiend.util.logging import get_logger
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.callbacks import (
    TrainingCallback,
    EvaluationCallback,
    NormalizationCallback,
    CheckpointCallback,
    LoggingCallback,
    get_default_callbacks,
)
from gradiend.trainer.core.stats import write_training_stats, load_training_stats, _best_step_abs_mean_by_type

logger = get_logger(__name__)


def train(
    model_with_gradiend,
    data: DataLoader,
    training_args: Optional[TrainingArguments] = None,
    callbacks: Optional[List[TrainingCallback]] = None,
    **kwargs
) -> str:
    """
    Train a GRADIEND model.

    Uses a list of callbacks (default: evaluation, normalization, checkpoint, logging).
    Pass callbacks=[] to add extra callbacks (e.g. EarlyStoppingCallback, TensorBoardCallback).

    Args:
        model_with_gradiend: ModelWithGradiend instance to train
        data: DataLoader providing training batches
        training_args: TrainingArguments (single config class for all training parameters)
        callbacks: Optional list of extra callbacks (appended to default callbacks)
        **kwargs: Additional parameters (will override config values)

    Returns:
        Path to saved model
    """
    if training_args is None:
        training_args = TrainingArguments.from_dict(kwargs)
    else:
        # Create a copy to avoid modifying the original object
        if kwargs:
            # Build dict of fields to update
            updates = {}
            for key, value in kwargs.items():
                if hasattr(training_args, key) and key in getattr(TrainingArguments, "__dataclass_fields__", {}):
                    updates[key] = value
                else:
                    raise ValueError(f"Invalid training argument: {key} with value {value}. All train() kwargs must be a field in TrainingArguments.")
            if updates:
                training_args = dataclasses.replace(training_args, **updates)

    training_args.__post_init__()

    # Re-apply seed so training loop (forward/backward, dropout, etc.) starts from a known RNG state
    if getattr(training_args, "seed", None) is not None:
        from gradiend.trainer.trainer import _apply_seed
        _apply_seed(int(training_args.seed))

    # Log training start
    if model_with_gradiend.gradiend is not None:
        logger.info(
            f'Training GRADIEND model over {len(model_with_gradiend):,} base model weights with {model_with_gradiend.gradiend.latent_dim} feature neurons'
        )
        logger.info(f'Output: {training_args.output_dir or ""}')

    if len(data) == 0:
        raise ValueError('Dataloader is empty! Please provide a dataloader with training data.')

    # Ensure model is on correct dtype
    if model_with_gradiend.base_model.dtype != training_args.torch_dtype:
        model_with_gradiend = model_with_gradiend.to(dtype=training_args.torch_dtype)

    # Initialize training state
    last_losses = []
    losses = []
    max_losses = 100
    global_step = 0
    total_training_time_start = time.time()

    training_stats = {
        'global_step': 0,
        'correlation': -1.0,
        'scores': {},
        'mean_by_class': {},  # step -> dict of label -> mean encoded value
        'mean_by_feature_class': {},  # step -> dict of feature_class (e.g. masc_nom) -> mean encoded value
        'encoder_norms': [],
        'decoder_norms': [],
    }

    time_stats = {
        'data_preparation': 0.0,
        'model_with_gradiend': 0.0,
        'eval': 0.0,
        'total': 0.0,
    }

    # Build callback list: defaults + optional extra. Deduplicate by default type (keep first).
    default_types = (EvaluationCallback, NormalizationCallback, CheckpointCallback, LoggingCallback)
    raw_callbacks = get_default_callbacks(training_args) + (callbacks or [])
    seen_types = set()
    all_callbacks = []
    for c in raw_callbacks:
        ctype = type(c)
        if ctype in default_types:
            if ctype in seen_types:
                logger.warning(
                    f"Duplicate {ctype.__name__} skipped; only the first instance is used. "
                    "Prefer get_default_callbacks() plus extra callbacks (e.g. EarlyStoppingCallback)."
                )
                continue
            seen_types.add(ctype)
        all_callbacks.append(c)
    control: dict = {"should_stop": False}
    config_dict = training_args.to_dict()

    for cb in all_callbacks:
        cb.on_train_begin(config=config_dict, training_stats=training_stats)
    
    # Ensure encoder/decoder built (lazy init: build with full input_dim if not yet built)
    if hasattr(model_with_gradiend.gradiend, "_ensure_built"):
        model_with_gradiend.gradiend._ensure_built()

    # Setup optimizer (encoder-only, decoder-only, or full GRADIEND)
    if training_args.supervised_encoder:
        train_params = list(model_with_gradiend.gradiend.encoder.parameters())
        logger.info("Supervised encoder: optimizing encoder parameters only.")
    elif training_args.supervised_decoder:
        train_params = list(model_with_gradiend.gradiend.decoder.parameters())
        logger.info("Supervised decoder: optimizing decoder parameters only.")
    else:
        train_params = list(model_with_gradiend.gradiend.parameters())
    if training_args.optim.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            train_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            eps=training_args.adam_epsilon
        )
    else:
        optimizer = torch.optim.Adam(
            train_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            eps=training_args.adam_epsilon
        )

    # Initial evaluation before training starts (at step 0)
    if training_args.do_eval and training_args.evaluate_fn is not None:
        logger.info('Running initial evaluation before training...')
        eval_start = time.time()
        eval_result = None
        _corr0 = training_stats.get('correlation', -1.0)
        step_kwargs_0 = dict(
            step=0,
            loss=0.0,
            model=model_with_gradiend,
            config=config_dict,
            training_stats=training_stats,
            eval_result=eval_result,
            control=control,
            last_iteration=False,
            last_losses=last_losses,
            losses=losses,
            epoch=0,
            best_score_checkpoint={
                'correlation': _corr0,
                'global_step': 0,
                'epoch': 0,
                'loss': 0.0,
            },
        )
        for cb in all_callbacks:
            out = cb.on_step_end(**step_kwargs_0)
            if out is not None:
                eval_result = out
                step_kwargs_0['eval_result'] = eval_result
        time_stats['eval'] += time.time() - eval_start
    
    # Training loop
    max_iter = training_args.max_steps if training_args.max_steps > 0 else None
    for epoch in range(training_args.num_train_epochs):
        if max_iter is not None and global_step >= max_iter:
            logger.info(f'Max steps {max_iter} reached, stopping training.')
            break
        
        for cb in all_callbacks:
            cb.on_epoch_begin(epoch=epoch, config=config_dict, training_stats=training_stats)

        # Set tqdm total to remaining steps when max_steps will cause early stop this epoch
        if max_iter is not None:
            steps_remaining = max(0, max_iter - global_step)
            epoch_total = min(len(data), steps_remaining) if steps_remaining > 0 else 0
        else:
            epoch_total = len(data)
        dataloader_iterator = tqdm(
            data,
            desc=f'Epoch {epoch + 1}/{training_args.num_train_epochs}',
            total=epoch_total,
            leave=True,
        )

        data_prep_start = time.time()
        for i, batch in enumerate(dataloader_iterator):
            # Data preparation
            source_tensor = batch['source']
            target_tensor = batch['target']

            if training_args.supervised_encoder or training_args.supervised_decoder:
                labels = torch.tensor(batch['label'], dtype=torch.float32)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # (batch, 1)
                label_dim = labels.shape[1]
                gradiend_n_features = model_with_gradiend.gradiend.latent_dim
                if label_dim != gradiend_n_features:
                    raise ValueError(
                        f'Label dimension {label_dim} does not match GRADIEND latent dimension {gradiend_n_features}! '
                        'Supervised encoder/decoder training requires equal feature and label dimensions.'
                    )

            time_stats['data_preparation'] += time.time() - data_prep_start

            step_begin_kwargs = dict(
                step=global_step + 1,
                config=training_args.to_dict(),
                model=model_with_gradiend,
                epoch=epoch,
                training_stats=training_stats,
            )
            for cb in all_callbacks:
                cb.on_step_begin(**step_begin_kwargs)

            # Forward pass
            gradiend_start = time.time()

            if training_args.supervised_decoder:
                # Supervised decoder: decoder(labels) vs target gradients; do not use source
                if target_tensor is None:
                    raise ValueError(
                        'Supervised decoder training requires target gradients. '
                        'Ensure target is set (e.g. "diff") and dataset provides target.'
                    )
                dev_dec = model_with_gradiend.gradiend.device_decoder
                labels = labels.to(device=dev_dec, dtype=model_with_gradiend.gradiend.torch_dtype)
                decoder_output = model_with_gradiend.gradiend.decoder(labels)
                if target_tensor.device != decoder_output.device:
                    target_tensor = target_tensor.to(decoder_output.device)
                loss = training_args.criterion(decoder_output, target_tensor)
                del target_tensor
            elif training_args.supervised_encoder:
                # Supervised encoder: encode(source) vs labels
                if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                    source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)
                encoded_value = model_with_gradiend.gradiend.encoder(source_tensor)
                del source_tensor
                if labels.device != encoded_value.device:
                    labels = labels.to(encoded_value.device)
                labels = labels.to(dtype=encoded_value.dtype)
                if training_args.train_batch_size > 1:
                    labels = labels.mean(dim=0, keepdim=True)
                else:
                    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
                loss = training_args.criterion(encoded_value, labels)
            else:
                # Standard GRADIEND: encode and decode
                if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                    source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)
                outputs_gradiend, encoded_value = model_with_gradiend.gradiend(
                    source_tensor, return_encoded=True
                )
                del source_tensor
                if target_tensor.device != outputs_gradiend.device:
                    outputs_gradiend = outputs_gradiend.to(target_tensor.device)
                loss = training_args.criterion(outputs_gradiend, target_tensor)
                del target_tensor
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            time_stats['model_with_gradiend'] += time.time() - gradiend_start
            
            # Update loss tracking
            loss_value = loss.item()
            last_losses.append(loss_value)
            if len(last_losses) > max_losses:
                last_losses.pop(0)
            losses.append(loss_value)
            
            global_step += 1
            
            # Update training stats
            training_stats['global_step'] = global_step
            training_stats['encoder_norms'].append(model_with_gradiend.gradiend.encoder_norm)
            training_stats['decoder_norms'].append(model_with_gradiend.gradiend.decoder_norm)
            
            # Call callbacks in order; eval_result flows from EvaluationCallback to later callbacks
            config_dict = training_args.to_dict()
            last_iteration = (
                (max_iter is not None and global_step >= max_iter) or
                (i == len(data) - 1 and epoch == training_args.num_train_epochs - 1)
            )
            _corr = training_stats.get('correlation', -1.0)
            step_kwargs = dict(
                step=global_step,
                loss=loss_value,
                model=model_with_gradiend,
                config=config_dict,
                training_stats=training_stats,
                eval_result=None,
                control=control,
                last_iteration=last_iteration,
                epoch=epoch,
                losses=losses,
                last_losses=last_losses,
                best_score_checkpoint={
                    'correlation': _corr,
                    'global_step': global_step,
                    'epoch': epoch,
                    'loss': loss_value,
                },
            )
            eval_result = None
            step_start = time.time()
            for cb in all_callbacks:
                out = cb.on_step_end(**step_kwargs)
                if out is not None:
                    eval_result = out
                    step_kwargs['eval_result'] = eval_result
            time_stats['eval'] += time.time() - step_start
            if max_iter is not None and global_step >= max_iter:
                break
            
            # Restart timer
            data_prep_start = time.time()
        
        # End of epoch
        time_stats['total'] = time.time() - total_training_time_start

        # Epoch-end callbacks (CheckpointCallback saves final model)
        _corr = training_stats.get('correlation', -1.0)
        training_info = {
            'losses': losses,
            'best_score_checkpoint': {
                'correlation': _corr,
                'global_step': global_step,
                'epoch': epoch,
            },
            'training_stats': training_stats,
            'training_args': config_dict,
            'time': time_stats,
        }
        epoch_end_kwargs = dict(
            epoch=epoch,
            model=model_with_gradiend,
            config=config_dict,
            time_stats=time_stats,
            **training_info,
        )
        for cb in all_callbacks:
            cb.on_epoch_end(**epoch_end_kwargs)

        if control.get("should_stop"):
            break

        if training_args.num_train_epochs > 1 and not (training_args.save_only_best or training_args.delete_models):
            output_epoch = f'{training_args.output_dir or ""}_epoch_{epoch + 1}'
            model_with_gradiend.save_pretrained(output_epoch, training=training_info)
    
    # Train-end callbacks
    for cb in all_callbacks:
        cb.on_train_end(config=config_dict, training_stats=training_stats)

    # Best checkpoint info (from callback; full training_stats stay in memory)
    best_score_checkpoint = {"correlation": training_stats.get("correlation", -1.0), "global_step": None, "epoch": None}
    for cb in all_callbacks:
        if isinstance(cb, CheckpointCallback):
            if getattr(cb, "use_loss_for_best", False):
                best_score_checkpoint = {
                    "correlation": None,
                    "loss": cb.best_score,
                    "global_step": cb.best_step,
                    "epoch": cb.best_epoch,
                }
                logger.info(f"Training completed (supervised_decoder). Best loss: {cb.best_score:.6f}")
            else:
                best_score_checkpoint = {
                    "correlation": cb.best_score,
                    "global_step": cb.best_step,
                    "epoch": cb.best_epoch,
                }
                logger.info(f"Training completed. Best correlation: {cb.best_score:.6f}")
            break
    else:
        best_corr = best_score_checkpoint.get("correlation") or training_stats.get("correlation", -1.0)
        logger.info(f"Training completed. Best correlation: {best_corr:.6f}")

    # Check convergence status and warn if non-convergent
    convergent_metric = (training_args.convergent_metric or ("loss" if training_args.supervised_decoder else "correlation")).lower()
    threshold = training_args.convergent_score_threshold
    min_convergent_seeds = training_args.min_convergent_seeds
    
    converged = True
    convergent_count = None
    if threshold is not None and min_convergent_seeds is not None and min_convergent_seeds > 0:
        # Check if this single-seed run converged
        if convergent_metric == "loss":
            metric_val = best_score_checkpoint.get("loss")
            if metric_val is None:
                metric_val = training_stats.get("loss")
            converged = metric_val is not None and metric_val <= threshold
        else:
            metric_val = best_score_checkpoint.get("correlation")
            if metric_val is None:
                metric_val = training_stats.get("correlation")
            abs_training_mean = None
            if training_args.convergent_mean_by_class_threshold is not None:
                best_abs = _best_step_abs_mean_by_type(training_stats, best_score_checkpoint)
                abs_training_mean = (best_abs or {}).get("training")
            mean_ok = (
                training_args.convergent_mean_by_class_threshold is None
                or abs_training_mean is None
                or (isinstance(abs_training_mean, (int, float)) and abs_training_mean >= training_args.convergent_mean_by_class_threshold)
            )
            converged = metric_val is not None and abs(metric_val) >= threshold and mean_ok
        
        if not converged:
            logger.warning(
                "Training completed but model did not converge: "
                "%s=%.4f (threshold=%.4f, required: %s convergent seeds). "
                "Model may not have reached the convergence threshold.",
                convergent_metric,
                metric_val if metric_val is not None else float('nan'),
                threshold,
                min_convergent_seeds,
            )
        convergent_count = 1 if converged else 0
    
    convergence_info = None
    if threshold is not None:
        convergence_info = {
            "converged": converged,
            "convergent_count": convergent_count,
            "min_convergent_seeds": min_convergent_seeds,
            "convergence_metric": convergent_metric,
            "threshold": threshold,
        }
    
    # Handle keep_only_best
    output_dir = training_args.output_dir or ""
    if training_args.save_only_best:
        _handle_keep_only_best(output_dir)
        # Overwrite training.json with full run history (best checkpoint was saved with truncated stats)
        write_training_stats(
            output_dir,
            training_stats=training_stats,
            best_score_checkpoint=best_score_checkpoint,
            training_args=config_dict,
            time_stats=time_stats,
            losses=losses,
            convergence_info=convergence_info,
        )
    else:
        # When save_only_best is False, CheckpointCallback saves training.json via save_pretrained.
        # We need to update it with convergence_info. Load existing training.json and rewrite with convergence_info.
        try:
            existing_stats = load_training_stats(output_dir)
            if existing_stats:
                write_training_stats(
                    output_dir,
                    training_stats=existing_stats.get("training_stats", training_stats),
                    best_score_checkpoint=existing_stats.get("best_score_checkpoint", best_score_checkpoint),
                    training_args=existing_stats.get("training_args", config_dict),
                    time_stats=existing_stats.get("time", time_stats),
                    losses=existing_stats.get("losses", losses),
                    convergence_info=convergence_info,
                )
        except Exception as e:
            logger.debug("Could not update convergence_info in training.json: %s", e)

    # Handle delete_models
    if training_args.delete_models:
        _delete_model_files(training_args.output_dir or "")
    
    # Release memory
    del model_with_gradiend
    gc.collect()
    torch.cuda.empty_cache()
    
    return training_args.output_dir or ""


def _handle_keep_only_best(output: str):
    """Handle keeping only the best model."""
    import os
    import shutil
    
    best_output = f'{output}_best'
    
    # If best model exists, use it; otherwise keep the final model
    if os.path.exists(best_output):
        output_temp = f'{output}_temp'
        if os.path.exists(output):
            os.rename(output, output_temp)
        
        os.rename(best_output, output)
        
        # Copy PDF and PNG files from temp
        if os.path.exists(output_temp):
            for file in os.listdir(output_temp):
                if file.endswith(('.pdf', '.png')):
                    shutil.copy(os.path.join(output_temp, file), output)
            
            # Copy subdirectories
            for folder in os.listdir(output_temp):
                folder_path = os.path.join(output_temp, folder)
                if os.path.isdir(folder_path):
                    shutil.copytree(folder_path, os.path.join(output, folder), dirs_exist_ok=True)
            
            shutil.rmtree(output_temp)
    # If no best model was saved, keep the final model (already at output dir)


def _delete_model_files(output: str):
    """Delete model files from output directory (.bin and .safetensors)."""
    import os

    if os.path.exists(output):
        for file in os.listdir(output):
            path = os.path.join(output, file)
            if file.endswith(".bin") or file.endswith(".safetensors"):
                os.remove(path)
