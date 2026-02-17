"""
Training callbacks for GRADIEND models (HF Trainer–style).

Callbacks handle evaluation, checkpointing, normalization, logging,
early stopping, and optional TensorBoard/Wandb logging.

Provided callbacks:
- EvaluationCallback: periodic evaluation (correlation, mean_by_class).
- NormalizationCallback: invert encoding when correlation < -0.5 (single latent).
- CheckpointCallback: save best model and optional periodic checkpoints.
- LoggingCallback: log loss, correlation, encoder/decoder norms.
- EarlyStoppingCallback: stop when metric has not improved for N steps.
- TensorBoardCallback: log metrics to TensorBoard (optional).
"""

import os
import time
from typing import Optional, Callable, Dict, Any, Union, List
from abc import ABC

from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


class TrainingCallback(ABC):
    """
    Base class for training callbacks (HF Trainer–style lifecycle).

    Subclass and override the hooks you need. All lifecycle hooks are optional
    except on_step_end and on_epoch_end (default no-op for compatibility).
    """

    def on_train_begin(self, config: Dict[str, Any], **kwargs) -> None:
        """Called once at the start of training."""
        pass

    def on_train_end(self, config: Dict[str, Any], **kwargs) -> None:
        """Called once at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, config: Dict[str, Any], **kwargs) -> None:
        """Called at the start of each epoch."""
        pass

    def on_step_begin(self, step: int, config: Dict[str, Any], **kwargs) -> None:
        """Called at the start of each step (before forward)."""
        pass

    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Called at the end of each step. Return value can be used by other callbacks
        (e.g. EvaluationCallback returns eval result for NormalizationCallback).
        """
        pass

    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], **kwargs) -> None:
        """Called at the end of each epoch."""
        pass


class EvaluationCallback(TrainingCallback):
    """Callback for periodic evaluation during training."""
    
    def __init__(self, evaluate_fn: Callable, n_evaluation: int = 250, do_eval: bool = True):
        """
        Args:
            evaluate_fn: Function to call for evaluation
            n_evaluation: Evaluate every N steps
            do_eval: Whether to perform evaluation
        """
        self.evaluate_fn = evaluate_fn
        self.n_evaluation = n_evaluation
        self.do_eval = do_eval
        self.last_eval_step = -1
    
    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any], training_stats: Dict[str, Any], **kwargs):
        """Perform evaluation if needed."""
        if not self.do_eval or self.evaluate_fn is None:
            return None
        
        # Check if we should evaluate at this step
        # Evaluate at step 0 (initial), every n_evaluation steps, and at last iteration
        should_eval = (
            step % self.n_evaluation == 0 or 
            kwargs.get('last_iteration', False)
        )
        
        # Skip if we already evaluated at this step
        if not should_eval or step == self.last_eval_step:
            return None
        
        self.last_eval_step = step
        
        logger.debug(f'Evaluating at step {step}...')
        
        # Set eval mode only on base_model (not the whole model) to avoid recursion
        # We need gradients enabled for GRADIEND evaluation, so don't use torch.no_grad()
        base_model_was_training = model.base_model.training
        gradiend_was_training = model.gradiend.training
        
        try:
            # Set eval mode on submodules (not the whole model to avoid recursion)
            model.base_model.eval()
            model.gradiend.eval()

            eval_start = time.time()
            eval_result = self.evaluate_fn(config=config, training_stats=training_stats)
            
            if eval_result:
                # Store label_value_to_class_name once (same for all steps); used for display
                if "label_value_to_class_name" in eval_result:
                    training_stats["label_value_to_class_name"] = eval_result["label_value_to_class_name"]

                # Update training_stats with evaluation results (tracked over time)
                for key, value in eval_result.items():
                    if key == "label_value_to_class_name":
                        continue  # already stored above
                    if key not in training_stats:
                        training_stats[key] = {}
                    
                    # Store results by step for tracking over time
                    if isinstance(training_stats[key], dict):
                        training_stats[key][step] = value
                    elif isinstance(training_stats[key], list):
                        training_stats[key].append(value)
                    elif isinstance(training_stats[key], (int, float)):
                        # Convert to dict to track over time
                        training_stats[key] = {step: value}
                
                # Ensure correlation is set (and per-step history)
                if 'correlation' in eval_result:
                    val = eval_result['correlation']
                    training_stats['correlation'] = val
                    if 'scores' not in training_stats:
                        training_stats['scores'] = {}
                    training_stats['scores'][step] = val
                
                # Per-feature-class means at DEBUG, 4 decimals
                if 'mean_by_feature_class' in eval_result:
                    mbfc = eval_result['mean_by_feature_class']
                    compact = ', '.join(f'{k}={v:.4f}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in sorted(mbfc.items()))
                    logger.debug(f'Mean by feature class: {compact}')
                
                return eval_result
            else:
                logger.warning(f'Evaluation at step {step} returned no results')
                return None
        except Exception as e:
            logger.error(f"Error during evaluation at step {step}: {e}", exc_info=True)
            return None
        finally:
            # Restore training mode
            if base_model_was_training:
                model.base_model.train()
            if gradiend_was_training:
                model.gradiend.train()
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], **kwargs):
        """No action needed at epoch end."""
        pass


class NormalizationCallback(TrainingCallback):
    """Callback for GRADIEND normalization during training."""
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to perform normalization
        """
        self.normalize = normalize
    
    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any], 
                   eval_result: Optional[Dict], training_stats: Dict[str, Any], **kwargs):
        """Perform normalization if needed."""
        if not self.normalize or eval_result is None:
            return
        
        if model.gradiend.latent_dim == 1 and 'mean_by_class' in eval_result:
            # For normalization, if the correlation is strongly negative, invert the encoding.
            try:
                corr = float(eval_result.get('correlation', 0.0))
                if corr < -0.6:
                    logger.info(f'Inverting encoding since correlation is {corr} < -0.5')
                    model.invert_encoding(update_direction=False)

                    # After inversion, flip the sign of the recorded correlation so that
                    # logs, checkpoints, and subsequent callbacks see the normalized
                    # (positive) orientation for this step.
                    new_corr = -corr
                    eval_result['correlation'] = new_corr
                    training_stats['correlation'] = new_corr

                    # Update per-step history if present
                    scores_hist = training_stats.get('scores')
                    if isinstance(scores_hist, dict):
                        scores_hist[step] = new_corr

                    # Flip mean encodings per class and per feature class to match the new orientation
                    mean_by_class = eval_result.get('mean_by_class')
                    if isinstance(mean_by_class, dict):
                        flipped_means = {
                            k: (-float(v) if isinstance(v, (int, float)) else v)
                            for k, v in mean_by_class.items()
                        }
                        eval_result['mean_by_class'] = flipped_means

                        # If we track mean_by_class over steps, update the current step
                        means_hist = training_stats.get('mean_by_class')
                        if isinstance(means_hist, dict):
                            means_hist[step] = flipped_means

                    mean_by_feature_class = eval_result.get('mean_by_feature_class')
                    if isinstance(mean_by_feature_class, dict):
                        flipped_mbfc = {
                            k: (-float(v) if isinstance(v, (int, float)) else v)
                            for k, v in mean_by_feature_class.items()
                        }
                        eval_result['mean_by_feature_class'] = flipped_mbfc

                        mbfc_hist = training_stats.get('mean_by_feature_class')
                        if isinstance(mbfc_hist, dict):
                            mbfc_hist[step] = flipped_mbfc
            except Exception as e:
                logger.warning(f"Error during normalization: {e}")
        elif model.gradiend.latent_dim > 1:
            logger.warning('Normalization is only implemented for single feature GRADIENDs')
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], **kwargs):
        """No action needed at epoch end."""
        pass


class CheckpointCallback(TrainingCallback):
    """
    Callback for saving model checkpoints during training.
    
    Behavior:
    - Saves the best model based on evaluation correlation (or loss when use_loss_for_best=True)
    - When use_loss_for_best=True (e.g. supervised_decoder), correlation is meaningless; best = lowest loss
    - Saves periodic checkpoints every checkpoint_interval steps (if checkpoints enabled)
    
    Args:
        output: Directory to save checkpoints
        checkpoints: If True, saves checkpoints every checkpoint_interval steps.
        keep_only_best: If True, keeps only the best model checkpoint (default: True)
        checkpoint_interval: Interval in steps for saving checkpoints (default: 5000)
        use_loss_for_best: If True, best checkpoint = lowest loss (e.g. for supervised_decoder where correlation is N/A)
    """

    def __init__(self, output: str, checkpoints: Union[bool, int] = False, keep_only_best: bool = True, checkpoint_interval: int = 5000, use_loss_for_best: bool = False):
        self.output = output
        self.checkpoints = checkpoints
        self.keep_only_best = keep_only_best
        self.use_loss_for_best = use_loss_for_best
        if checkpoints is True:
            self.checkpoint_step = checkpoint_interval
        elif isinstance(checkpoints, int):
            self.checkpoint_step = checkpoints
        else:
            self.checkpoint_step = 0
        self.best_score = None
        self.best_step = None
        self.best_epoch = None

    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any],
                    training_stats: Dict[str, Any], **kwargs):
        """Save checkpoint if needed."""
        if self.use_loss_for_best:
            # Best = lowest loss (e.g. supervised_decoder; correlation not meaningful)
            is_better = self.best_score is None or loss < self.best_score
            score_for_log = loss
        else:
            corr = training_stats.get('correlation', -1.0)
            is_better = self.best_score is None or abs(corr) > abs(self.best_score)
            score_for_log = corr

        if is_better:
            was_first = self.best_score is None
            old_score = self.best_score
            self.best_score = loss if self.use_loss_for_best else training_stats.get('correlation', -1.0)
            self.best_step = step
            self.best_epoch = kwargs.get('epoch', 0)

            if was_first:
                logger.debug(f'First {"loss" if self.use_loss_for_best else "correlation"}: {score_for_log:.4f} at step {step}')
            else:
                logger.debug(f'New best {"loss" if self.use_loss_for_best else "correlation"}: {score_for_log:.4f} at step {step} (previous: {old_score:.4f})')

            if step > 1:
                best_output = f'{self.output}_best'
                training_info = {
                    'losses': kwargs.get('losses', []),
                    'best_score_checkpoint': {
                        'correlation': None if self.use_loss_for_best else self.best_score,
                        'loss': loss,
                        'global_step': step,
                        'epoch': self.best_epoch,
                    },
                    'training_stats': training_stats,
                    'training_args': config,
                }
                model.save_pretrained(best_output, training=training_info)
                logger.debug(f'Saved best model checkpoint at step {step} ({"loss" if self.use_loss_for_best else "correlation"}: {score_for_log:.4f})')
        
        # Save periodic checkpoint
        if self.checkpoint_step > 0 and step % self.checkpoint_step == 0 and step > 0:
            checkpoint_output = f'{self.output}_step_{step}'
            model.save_pretrained(checkpoint_output)
            logger.info(f'Saved checkpoint at step {step}')
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], 
                    training_stats: Dict[str, Any], losses: list, **kwargs):
        """Save final model at end of epoch."""
        best_score_checkpoint = {
            'global_step': self.best_step,
            'epoch': epoch,
        }
        if self.use_loss_for_best:
            best_score_checkpoint['loss'] = self.best_score
            best_score_checkpoint['correlation'] = None
        else:
            best_score_checkpoint['correlation'] = self.best_score
        training_info = {
            'losses': losses,
            'best_score_checkpoint': best_score_checkpoint,
            'training_stats': training_stats,
            'training_args': config,
        }
        model.gradiend.save_pretrained(self.output, training=training_info)
        logger.info(f'Saved model after epoch {epoch + 1} to {self.output}')


class LoggingCallback(TrainingCallback):
    """Callback for logging training progress."""
    
    def __init__(self, n_loss_report: int = 100, loss_only: bool = False):
        """
        Args:
            n_loss_report: Log every N steps
            loss_only: If True (e.g. supervised_decoder), correlation is N/A; log loss only.
        """
        self.n_loss_report = n_loss_report
        self.loss_only = loss_only
    
    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any],
                    training_stats: Dict[str, Any], **kwargs):
        """Log training progress."""
        last_losses = kwargs.get("last_losses") or ()
        eval_result = kwargs.get("eval_result")
        should_log = (
            step % self.n_loss_report == 0 or
            step == 0 or
            kwargs.get('last_iteration', False)
        )
        # Log if we should log AND (have losses OR have eval results for step 0)
        if should_log and (last_losses or (step == 0 and eval_result is not None)):
            corr = training_stats.get('correlation', -1.0)

            # Try to get mean encoded values per class for the current step
            mean_by_class_hist = training_stats.get('mean_by_class', {})
            current_means = None
            if isinstance(mean_by_class_hist, dict):
                current_means = mean_by_class_hist.get(step)
            # Label value -> display name (stored once; fallback: step->dict from older runs)
            _lv_to_name_raw = training_stats.get('label_value_to_class_name') or {}
            if not _lv_to_name_raw:
                label_value_to_class_name = {}
            else:
                first_val = next(iter(_lv_to_name_raw.values()), None)
                label_value_to_class_name = first_val if isinstance(first_val, dict) else _lv_to_name_raw

            mean_str = ""
            if isinstance(current_means, dict) and current_means:
                def _mean_display_name(label_val):
                    return label_value_to_class_name.get(
                        label_val,
                        label_value_to_class_name.get(float(label_val), "neutral" if label_val in (0, 0.0) else str(label_val))
                    )

                keys = sorted(current_means.keys(), key=lambda x: (0 if x == 0 else (-1 if x < 0 else 1), x))
                parts = [
                    f"{_mean_display_name(k)}: {float(current_means[k]):.4f}"
                    for k in keys if isinstance(current_means.get(k), (int, float))
                ]
                mean_str = ", ".join(parts)

            suffix = ""
            if not self.loss_only and kwargs.get('best_score_checkpoint'):
                bsc = kwargs['best_score_checkpoint']
                best_corr = bsc.get('correlation', -1.0)
                if best_corr is not None and abs(corr) > abs(best_corr):
                    suffix = " (new best)"
            corr_str = "N/A" if self.loss_only else f"{corr:.4f}"
            logger.info(
                f'Step {step}, Correlation: {corr_str}, '
                + (f', mean: {mean_str}' if mean_str else '')
                + suffix
            )
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], 
                    time_stats: Dict[str, float], **kwargs):
        """Log epoch completion."""
        try:
            import humanize
            import datetime
            
            def humanize_time(seconds):
                return humanize.naturaldelta(datetime.timedelta(seconds=seconds))
            
            total = humanize_time(time_stats.get("total", 0))
            total_epochs = config.get("num_train_epochs", config.get("epochs", 1))
            logger.info(
                f'Epoch {epoch + 1}/{total_epochs} finished in {total}. '
            )
        except ImportError:
            t = time_stats.get("total", 0)
            total_epochs = config.get("num_train_epochs", config.get("epochs", 1))
            logger.info(f'Epoch {epoch + 1}/{total_epochs} finished. Total: {t:.2f}s')

def get_default_callbacks(config: Any) -> List[TrainingCallback]:
    """
    Build the default callback list used by the core training loop.

    Order: Evaluation -> Normalization -> Checkpoint -> Logging.
    Accepts TrainingArguments or a dict (e.g. training_args.to_dict()).
    """
    if hasattr(config, "output_dir"):
        output = config.output_dir or ""
    else:
        output = config.get("output_dir", "")
    supervised_decoder = getattr(config, "supervised_decoder", False) or config.get("supervised_decoder", False)
    if hasattr(config, "eval_steps"):
        n_eval = config.eval_steps if config.do_eval and not supervised_decoder else 0
        do_eval = config.do_eval and not supervised_decoder  # skip eval for supervised_decoder (correlation N/A)
        evaluate = config.evaluate_fn
        checkpoints = config.save_strategy == "steps"
        keep_only_best = config.save_only_best
        checkpoint_interval = config.save_steps
        normalize_gradiend = config.normalize_gradiend
    else:
        output = config.get("output_dir", "")
        n_eval = config.get("eval_steps", 250) if (config.get("do_eval", True) and not supervised_decoder) else 0
        do_eval = config.get("do_eval", True) and not supervised_decoder
        evaluate = config.get("evaluate_fn")
        checkpoints = config.get("save_strategy", "best") == "steps"
        keep_only_best = config.get("save_only_best", True)
        checkpoint_interval = config.get("save_steps", 5000)
        normalize_gradiend = config.get("normalize_gradiend", True)

    return [
        EvaluationCallback(evaluate_fn=evaluate, n_evaluation=n_eval, do_eval=do_eval),
        NormalizationCallback(normalize=normalize_gradiend),
        CheckpointCallback(
            output=output,
            checkpoints=checkpoints,
            keep_only_best=keep_only_best,
            checkpoint_interval=checkpoint_interval,
            use_loss_for_best=supervised_decoder,
        ),
        LoggingCallback(n_loss_report=n_eval if n_eval > 0 else 100, loss_only=supervised_decoder),
    ]
