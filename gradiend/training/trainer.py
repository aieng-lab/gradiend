import gc
import random
import shutil

import numpy as np
from matplotlib import pyplot as plt

import time
import torch.nn as nn
from scipy.stats import pearsonr

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from gradiend.model import ModelWithGradiend
from gradiend.training.dataset import create_training_dataset, create_eval_dataset

import datetime
import os

from gradiend.util import hash_it


# Create a unique directory for each run based on the current time
def get_log_dir(base_dir="logs", output=''):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, output + f'_{current_time}')
    return log_dir

# Define the custom loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        return mse + self.alpha * l1

    def __str__(self):
        return f'CombinedLoss(alpha={self.alpha})'

class PolarFeatureLoss(nn.Module):
    def __init__(self, alpha=0.001):
        super(PolarFeatureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target, encoded_value):
        mse_loss = self.mse(output, target)
        reg_term = 1.0 - torch.abs(encoded_value)
        loss = mse_loss + self.alpha * reg_term * mse_loss
        print(f'MSE Loss: {mse_loss.item()}, Regularization Term: {reg_term.item()}, Encoded Value: {encoded_value.item()}')
        return loss


"""
Function: train

Trains a BERT-based autoencoder model using gradient-based input representations with evaluation, and logging functionalities.

Parameters:

model_with_gradiend: ModelWithGradiend instance of the model to train
output (str, default='results/models/gradiend'): Path to save trained model.
checkpoints (bool, default=False): Enables intermediate checkpointing every 5000 steps.
max_iterations (int, optional): Maximum number of training iterations.
criterion_ae (nn.Module, default=nn.MSELoss()): Loss function for autoencoder.
batch_size (int, default=32): Training batch size.
batch_size_data (bool or int, default=True): If the training data is batched, i.e., only a single gender is used for the training. If True, uses batch_size for data loading.
source (str, default='gradient'): Type of input data ('gradient', 'inv_gradient', 'diff').
target (str, default='diff'): Type of target data ('gradient', 'inv_gradient', 'diff').
epochs (int, default=1): Number of training epochs.
neutral_data (bool, default=False): Uses gender-neutral data if True. This gender-neutral data is then also used for the training.
neutral_data_prop (float, default=0.5): Proportion of neutral data when neutral_data=True.
plot (bool, default=False): Enables visualization of autoencoder training.
n_evaluation (int, default=250): Evaluation frequency in training steps. The evaluation is based on the GRADIEND encoder.
lr (float, default=1e-5): Learning rate.
weight_decay (float, default=1e-2): Weight decay for optimizer.
do_eval (bool, default=True): Enables evaluation during training.
keep_only_best (bool, default=True): Retains only the best-performing model.
eval_max_size (int, optional): Maximum size of evaluation dataset.
eval_batch_size (int, default=32): Batch size for evaluation.
eps (float, default=1e-8): Epsilon for numerical stability in optimization.
normalized (bool, default=True): Normalizes encoded values if True.

Saves best model in output_best directory.

Returns:

output (str): Path where the trained model is saved.
"""
def train(model_with_gradiend,
          output='results/models/gradiend',
          checkpoints=False,
          max_iterations=None,
          criterion_ae=nn.MSELoss(),
          batch_size=32,
          batch_size_data=True,
          source='gradient',
          target='diff',
          epochs=1,
          neutral_data=False,
          neutral_data_prop=0.5,
          plot=False,
          n_evaluation=250,
          lr=1e-5,
          weight_decay=1e-2,
          do_eval=True,
          keep_only_best=True,
          eval_max_size=None,
          eval_batch_size=32,
          eps=1e-8,
          normalized=True,
          use_cached_gradients=False,
          torch_dtype=torch.float32,
          ):

    print('Training GRADIEND model')
    print('Output:', output)
    print('Batch size:', batch_size)
    print('Learning rate:', lr)

    if model_with_gradiend.base_model.dtype != torch_dtype:
        model_with_gradiend = model_with_gradiend.to(dtype=torch_dtype)

    # Load pre-trained BERT model and tokenizer
    tokenizer = model_with_gradiend.tokenizer
    is_generative = model_with_gradiend.is_generative

    # Create a dataset and dataloader for BERT inputs

    if batch_size_data is True:
        batch_size_data = batch_size

    dataset = create_training_dataset(tokenizer,
                                      max_size=None,
                                      split='train',
                                      neutral_data=neutral_data,
                                      batch_size=batch_size_data,
                                      neutral_data_prop=neutral_data_prop,
                                      is_generative=is_generative,
                                      dtype=torch_dtype,
                                      )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if do_eval:
        eval_data = create_eval_dataset(model_with_gradiend, split='val', source=source, max_size=eval_max_size, is_generative=is_generative)

        def _evaluate(data):
            start = time.time()
            # even if other source is chosen, the correct gradients are loaded into eval_data
            gradients = data['gradients']
            labels = data['labels']

            device = model_with_gradiend.gradiend.device_encoder
            encoded = []

            grads = list(gradients.values())

            if eval_batch_size > 1:
                for i in range(0, len(grads), eval_batch_size):
                    batch = grads[i:min(i + eval_batch_size, len(grads))]
                    batch_on_device = [g.to(device, dtype=torch_dtype) for g in batch]
                    encoded_values = model_with_gradiend.gradiend.encoder(torch.stack(batch_on_device))
                    encoded.extend(encoded_values.view(-1).tolist())
                    # free memory on device
                    del batch_on_device
                    torch.cuda.empty_cache()  # if using GPU, it helps clear memory
            else:
                for grads in gradients.values():
                    encoded_value = model_with_gradiend.gradiend.encoder(grads.to(device, dtype=torch_dtype))
                    encoded.append(encoded_value.item())

            score = -pearsonr(list(labels.values()), encoded).correlation

            # split the encoded values by label value
            male_encoded_values = [e for e, label in zip(encoded, labels.values()) if label == 1]
            female_encoded_values = [e for e, label in zip(encoded, labels.values()) if label == 0]
            mean_male_encoded_value = np.mean(male_encoded_values)
            mean_female_encoded_value = np.mean(female_encoded_values)

            if normalized and mean_female_encoded_value < -0.5 and mean_male_encoded_value > 0.5:
                print(f'Invert encoding since female encoded value is {mean_female_encoded_value}<-0.5 and male encoded value is {mean_male_encoded_value}>0.5')
                model_with_gradiend.invert_encoding()
                score = -score
                mean_male_encoded_value = -mean_male_encoded_value
                mean_female_encoded_value = -mean_female_encoded_value

            if np.isnan(score):
                score = 0.0

            end = time.time()
            print(f'Evaluated in {(end - start):.2f}s, mean male {mean_male_encoded_value:.6f}, mean female {mean_female_encoded_value:.6f}')
            print('male encoded values', male_encoded_values[:10])
            print('female encoded values', female_encoded_values[:10])
            return score, mean_male_encoded_value, mean_female_encoded_value


        def evaluate():
            score_ = _evaluate(eval_data)
            return score_
    else:
        if normalized:
            raise ValueError('Normalization is only possible if evaluation is enabled!')

        # dummy evaluation function
        def evaluate():
            return None


    # Training loop
    start_time = time.time()
    global_step = 0
    last_losses = []
    last_losses2 = []
    scores = []
    losses = []
    losses2 = []
    encoder_changes = []
    decoder_changes = []
    encoder_norms = []
    decoder_norms = []
    mean_males = []
    mean_females = []
    max_losses = 100
    max_losses2 = 1000 # keep track of a 2nd moving average for compatibility reasons
    convergence = None
    best_score_checkpoint = None
    score = 0.0
    total_training_time_start = time.time()
    training_data_prep_time = 0.0
    training_gradiend_time = 0.0
    eval_time = 0.0
    cache_dir = ''
    if use_cached_gradients:
        model_id = os.path.basename(model_with_gradiend.base_model.name_or_path)
        layers_hash = model_with_gradiend.layers_hash()
        cache_dir = f'results/cache/training/gradiend/{model_id}/{layers_hash}'

    optimizer_ae = torch.optim.AdamW(model_with_gradiend.gradiend.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)

    if plot:
        fig = plt.figure(figsize=(12, 6))
        model_with_gradiend.gradiend.plot(fig=fig)
        plt.pause(0.1)
    else:
        fig = None

    len_dataloader = len(dataloader)
    for epoch in range(epochs):
        dataloader_iterator = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)

        for i, batch in enumerate(dataloader_iterator):

            ####### Data Preparation ########
            data_prep_start = time.time()
            cache_file = ''
            if use_cached_gradients:
                hash = hash_it(batch['text'])
                cache_file = f'{cache_dir}/{hash}.pt'

            if use_cached_gradients and os.path.exists(cache_file):
                factual_gradients, counterfactual_gradients = torch.load(cache_file)
                # todo actually save gradients!

            else:

                factual_inputs = batch[True]
                counterfactual_inputs = batch[False]

                factual_gradients = None
                counterfactual_gradients = None

                gradients_keywords = {'gradient', 'diff'}
                inv_gradients_keywords = {'inv_gradient', 'diff'}

                source = source.strip()
                target = target.strip()

                if source in gradients_keywords or target in gradients_keywords:
                    factual_gradients = model_with_gradiend.forward_pass(factual_inputs)

                if source in inv_gradients_keywords or target in inv_gradients_keywords:
                    counterfactual_gradients = model_with_gradiend.forward_pass(counterfactual_inputs)

                del factual_inputs
                del counterfactual_inputs

            if source == 'gradient':
                source_tensor = factual_gradients
            elif source == 'diff':
                source_tensor = factual_gradients - counterfactual_gradients
            elif source == 'inv_gradient':
                source_tensor = counterfactual_gradients
            else:
                raise ValueError(f'Unknown source: {source}')

            target_tensor = factual_gradients
            if target == 'inv_gradient':
                target_tensor = counterfactual_gradients
            elif target == 'diff':
                target_tensor -= counterfactual_gradients
            elif target == 'gradient':
                break # target_tensor is already set
            else:
                raise ValueError(f'Unknown target: {target}')
            training_data_prep_time += time.time() - data_prep_start


            del factual_gradients
            del counterfactual_gradients

            ######## Gradiend Training ########
            gradiend_start = time.time()

            # Forward pass through autoencoder
            if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)

            outputs_ae, encoded_value = model_with_gradiend.gradiend(source_tensor, return_encoded=True)
            del source_tensor

            # calculate loss
            if target_tensor.device != outputs_ae.device:
                target_tensor = target_tensor.to(outputs_ae.device)

            # release memory
            torch.cuda.empty_cache()
            gc.collect()


            if isinstance(criterion_ae, PolarFeatureLoss):
                loss_ae = criterion_ae(outputs_ae, target_tensor, encoded_value)
            else:
                loss_ae = criterion_ae(outputs_ae, target_tensor)
            del target_tensor

            optimizer_ae.zero_grad()
            loss_ae.backward()
            #gc.collect()  # Run Python garbage collection
            #torch.cuda.empty_cache()  # Release unused memory back to CUDA driver
            optimizer_ae.step()

            loss_ae = loss_ae.item()
            training_gradiend_time += time.time() - gradiend_start

            if len(last_losses) < max_losses:
                last_losses.append(loss_ae)
            else:
                last_losses = last_losses[1:] + [loss_ae]

            if len(last_losses2) < max_losses2:
                last_losses2.append(loss_ae)
            else:
                last_losses2 = last_losses2[1:] + [loss_ae]

            if max_iterations and global_step >= max_iterations:
                convergence = f'max iterations ({max_iterations}) reached'
                break

            global_step += 1
            last_iteration = global_step == max_iterations or (epoch == epochs - 1 and i == len_dataloader - 1)
            if do_eval and ((i+1) % n_evaluation == 0 or i == 0)or last_iteration:
                eval_start = time.time()
                score, mean_male, mean_female = evaluate()
                scores.append(score)
                mean_males.append(mean_male)
                mean_females.append(mean_female)
                eval_time += time.time() - eval_start

            n_loss_report = n_evaluation if n_evaluation > 0 else 100
            if ((i+1) % n_loss_report == 0 or  i == 0) or last_iteration:
                # validate on small validation set
                mean_loss = sum(last_losses) / len(last_losses)
                mean_loss2 = sum(last_losses2) / len(last_losses2)
                encoder_norm = model_with_gradiend.gradiend.encoder_norm
                decoder_norm = model_with_gradiend.gradiend.decoder_norm
                avg_grad_norm = model_with_gradiend.gradiend.avg_gradient_norm
                output_str = f'Epoch [{epoch + 1}/{epochs}], Loss AE: {mean_loss:.10f}, Correlation score: {score:.6f}, encoder {encoder_norm}, decoder {decoder_norm}, avg grad norm {avg_grad_norm}'
                if hasattr(model_with_gradiend.gradiend, 'encoder_change'):
                    encoder_change = model_with_gradiend.gradiend.encoder_change
                    decoder_change = model_with_gradiend.gradiend.decoder_change
                    output_str += f'encoder change {encoder_change}, decoder change {decoder_change}'
                    encoder_changes.append(encoder_change)
                    decoder_changes.append(decoder_change)

                print(output_str)
                losses.append(mean_loss)
                losses2.append(mean_loss2)
                encoder_norms.append(encoder_norm)
                decoder_norms.append(decoder_norm)

                if best_score_checkpoint is None or abs(score) >= abs(best_score_checkpoint['score']):
                    if best_score_checkpoint is None:
                        print('First score:', score, 'at global step', global_step)
                    elif abs(score) > abs(best_score_checkpoint['score']):
                        print('New best score:', score, 'at global step', global_step)
                    else:
                        print('Same score:', score, 'at global step', global_step)
                    best_score_checkpoint = {
                        'score': score,
                        'global_step': global_step,
                        'epoch': epoch,
                        'loss': mean_loss
                    }
                    # save checkpoint
                    training_information = {
                        'max_iterations': max_iterations,
                        'convergence': convergence,
                        'batch_size': batch_size,
                        'batch_size_data': batch_size_data,
                        'criterion_ae': str(criterion_ae),
                        'activation': str(model_with_gradiend.gradiend.activation),
                        'output': output,
                        'base_model': model_with_gradiend.base_model.name_or_path,
                        'layers': model_with_gradiend.gradiend.layers,
                        'score': score,
                        'scores': scores,
                        'mean_males': mean_males,
                        'mean_females': mean_females,
                        'losses': losses,
                        'losses_1000': losses2,
                        'encoder_changes': encoder_changes,
                        'decoder_changes': decoder_changes,
                        'encoder_norms': encoder_norms,
                        'decoder_norms': decoder_norms,
                        'time': time.time() - start_time,
                        'best_score_checkpoint': best_score_checkpoint,
                        'bias_decoder': model_with_gradiend.gradiend.bias_decoder,
                        'epoch': epoch,
                        'n_evaluation': n_evaluation,
                        'lr': lr,
                        'weight_decay': weight_decay,
                        'source': source,
                        'target': target,
                        'global_step':  global_step,
                        'eval_max_size': eval_max_size,
                        'eval_batch_size': eval_batch_size,
                        'eps': eps,
                        'training_data_prep_time': training_data_prep_time,
                        'training_gradiend_time': training_gradiend_time,
                        'eval_time': eval_time,
                        'total_training_time': time.time() - total_training_time_start,
                    }

                    model_with_gradiend.save_pretrained(f'{output}_best', training=training_information)

            if i > 0:
                if plot and i % 1000 == 0:
                    model_with_gradiend.gradiend.plot(fig=fig, n=i)
                    plt.pause(0.1)


                if checkpoints and global_step % 5000 == 0:
                    model_name = f'{output}_{global_step}'
                    model_with_gradiend.save_pretrained(model_name, convergence=convergence)
                    print('Saved intermediate result')

        training_information = {
            'max_iterations': max_iterations,
            'convergence': convergence,
            'batch_size': batch_size,
            'batch_size_data': batch_size_data,
            'criterion_ae': str(criterion_ae),
            'activation': str(model_with_gradiend.gradiend.activation),
            'output': output,
            'base_model': model_with_gradiend.base_model.name_or_path,
            'layers': model_with_gradiend.gradiend.layers,
            'score': score,
            'scores': scores,
            'mean_males': mean_males,
            'mean_females': mean_females,
            'losses': losses,
            'losses_1000': losses2,
            'encoder_changes': encoder_changes,
            'decoder_changes': decoder_changes,
            'encoder_norms': encoder_norms,
            'decoder_norms': decoder_norms,
            'time': time.time() - start_time,
            'best_score_checkpoint': best_score_checkpoint,
            'bias_decoder': model_with_gradiend.gradiend.bias_decoder,
            'epoch': epoch,
            'n_evaluation': n_evaluation,
            'lr': lr,
            'weight_decay': weight_decay,
            'source': source,
            'target': target,
            'global_step': global_step,
            'eval_max_size': eval_max_size,
            'eval_batch_size': eval_batch_size,
            'eps': eps,
            'training_data_prep_time': training_data_prep_time,
            'training_gradiend_time': training_gradiend_time,
            'eval_time': eval_time,
            'total_training_time': time.time() - total_training_time_start,
        }

        model_with_gradiend.gradiend.save_pretrained(output, training=training_information)
        print('Saved the auto encoder model as', output)
        print('Best score:', best_score_checkpoint)
        if epochs > 1:
            output_epoch = f'{output}_epoch_{epoch + 1}'
            model_with_gradiend.save_pretrained(output_epoch, training=training_information)


        try:
            import humanize
            import datetime

            def humanize_time(seconds):
                return humanize.naturaldelta(datetime.timedelta(seconds=seconds))
            print(f'Epoch {epoch + 1}/{epochs} finished')
            print('Total Training time:', humanize_time(training_information['total_training_time']))
            print('Training data preparation time:', humanize_time(training_information['training_data_prep_time']))
            print('Training Evaluation time:', humanize_time(training_information['eval_time']))
            print('Training GRADIEND time:', humanize_time(training_information['training_gradiend_time']))
        except ModuleNotFoundError:
            print('Please install humanize to get a human-readable training time')

    if plot:
        plt.show()

    print('Best score:', best_score_checkpoint)

    # release memory
    del model_with_gradiend

    # Call garbage collector
    gc.collect()

    # Empty the CUDA cache
    torch.cuda.empty_cache()

    if keep_only_best:
        # delete the output folder
        shutil.rmtree(output)

        # rename the output_best folder to output
        os.rename(f'{output}_best', output)


    print('Saved the auto encoder model as', output)
    return output

def create_bert_with_ae(model, layers=None, activation='tanh', activation_decoder=None, bias_decoder=True, grad_iterations=1, decoder_factor=1.0, seed=0, torch_dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs['torch_dtype'] = torch_dtype
    return ModelWithGradiend.from_pretrained(model, layers, activation=activation, activation_decoder=activation_decoder, bias_decoder=bias_decoder, grad_iterations=grad_iterations, decoder_factor=decoder_factor, torch_dtype=torch_dtype), kwargs

def train_single_layer_gradiend(model, layer='base_model.encoder.layer.10.output.dense.weight', **kwargs):
    bert_with_ae, kwargs = create_bert_with_ae(model, [layer], **kwargs)
    return train(bert_with_ae, **kwargs)

def train_multiple_layers_gradiend(model, layers, **kwargs):
    bert_with_ae, kwargs = create_bert_with_ae(model, layers, **kwargs)
    return train(bert_with_ae, **kwargs)

def train_all_layers_gradiend(model='bert-base-cased', **kwargs):
    bert_with_ae, kwargs = create_bert_with_ae(model, **kwargs)
    return train(bert_with_ae, **kwargs)



if __name__ == '__main__':
    train_all_layers_gradiend('bert-base-cased',
                              output='results/models/gradiend',
                              checkpoints=False,
                              max_iterations=1000000,
                              criterion_ae=nn.MSELoss(),
                              batch_size=8,
                              batch_size_data=None,
                              activation='relu',
                              )
