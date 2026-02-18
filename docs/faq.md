# FAQ / troubleshooting

## Which modalities are supported?

Currently **only gradients based on text prediction (MLM/CLM)** are supported. All documentation and examples use `TextPredictionTrainer`. Other modalities may be added in the future.

## My model does not converge (bad correlations). What parameters should I tweak?

You should check the training log and inspect the mean probabilities per label class. 

- **Mean Encoded values of the two target classes are similar and close to +1 or -1**: The encoder is not separating the classes (likely only separates target classes from neutral inputs). Try a smaller *learning rate* and/or different seeds (increase `TrainingArguments.max_seeds`). 
- **Mean Encoded values of the two target classes are near zero and barely moving during training:** Add more data (increase `train_max_size`), increase training length (`max_steps` or `num_train_epochs`), and/or use a larger *learning rate*.
- **Mean Encoded values do not change during training even with larger learning rate:** Likely only zero gradients are computed based on the provided data. Probably your mask identifier (default `[MASK]`) is different in your data? Or the mask occurs outside the model's context (too long texts)?

**More general hints:**

- **Source** — Using `source="alternative"` typically yields simpler conversion than `source='factual'`.
- **Data balance and size** — Ensure both classes in your pair have enough examples; use `train_max_size` per class if needed. See [Data handling](guides/data-handling.md) for balancing.
- **Encoder evaluation** — Run `trainer.evaluate_encoder(return_df=True, plot=True)` and check convergence plots for debugging; use `trainer.plot_training_convergence()` to see correlation over steps.
- **Data Quality** — Check the quality of your data and labels (spaCy misclassifies, etc.).
- **Pruning** — try first to train the full (un-pruned) model to check if pre-pruning may accidently removes actually important weigths that are required for convergence

See [TrainingArguments](api/training/TrainingArguments.md) and the [start here](start.md) / [detailed workflow](tutorials/detailed-workflow.md) tutorials for all training options.


## I get a CUDA out of memory error. How can I reduce memory usage?

The GRADIEND model itself holds 3*n+1 parameters, where n is the number of (considered) parameters of the base model. During training, due to the optimizer and the large input parameter space (n), we require a multiple of the base model's memory.
To reduce memory usage, you can:
- Apply pre-pruning (typically, a top-k of 0.01 (i.e, retaining 1% of the base model's parameters) still yields full GRADIEND performance and reduces GRADIEND size significantly.
- Use a smaller base model (e.g. `bert-base-uncased` instead of `bert-large-uncased`).
- Reduce the batch size (`train_batch_size`) and/or sequence length of your data.
- Use mixed precision training (`TrainingArguments.torch_dtype = torch.bfloat16`), which typically reduces memory usage by about half with minimal impact on convergence. Note: this requires a compatible GPU (e.g. NVIDIA Ampere or later for bfloat16).
- Use multiple GPUs: the library supports currently up to 3 GPUs for training (one for the base model, one for the encoder, and one for the decoder). (The training implementation is not super efficient, but this is a workaround if you have multiple GPUs available.)

## You have a different issue

Write a GitHub issue with a clear description of the problem, steps to reproduce, and any error messages or logs. We will try to help as soon as possible!