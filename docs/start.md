# Start here

This page walks through a **self-contained** workflow: you train and evaluate a GRADIEND model for singular-plural feature based on English third person pronouns.

**Runnable script:** [gradiend/examples/start_workflow.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py). 

```bash
python -m gradiend.examples.start_workflow
```


## 1. Create data with `TextPredictionDataCreator`

To extract the desired feature *singular-plural*, we extract data where feature-related tokens are masked. We use third-person pronouns in English for that, i.e., *he*/*she*/*it* (3SG) and *they* (3PL). We need some base data to filter from and extract masks (e.g., *he* gets replaced by *[MASK]*); for the sake of this simple example, we use an explicit list of 25 texts:

```python
ARTIFICIAL_TEXTS = [
    "The chef tasted the soup, then he added a pinch of pepper and stirred it.",
    "The pianist closed her eyes and played the final chord; she had practised it for weeks.",
    "The dog ran to the door; it wanted to go outside and chase the ball.",
    # ...
]
```
We need to mask the pronouns in the texts, e.g., "The chef tasted the soup, then [MASK] added ..." , with the *factual* label being "he" based on the original text. For GRADIEND, we later automatically consider also the *counterfactual/alternative* target *they* by the chosen feature. Symmetrical, we derive texts with factual *they* and counterfactual *he*/*she*/*it*.

This package provides an easy way to derive such masked texts using `TextPredictionDataCreator`. This basic example uses basic string matching based on `TextFilterConfig` (see [here](guides/data-handling.md) for advanced filtering based on [spacy](https://spacy.io/)).

```python
from gradiend import TextPredictionDataCreator, TextFilterConfig
creator = TextPredictionDataCreator(
    base_data=ARTIFICIAL_TEXTS,
    feature_targets=[
        TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
        TextFilterConfig(targets=["they"], id="3PL"),
    ],
)
training = creator.generate_training_data(max_size_per_class=10)
```
This creates training data for GRADIEND, automatically split into train/validation/test splits.

To evaluate GRADIEND models, a dataset being independant (*neutral*) to the considered feature is useful to enable feature-independant evaluation (e.g., to compute a language modeling score). This is also supported by `TextPredictionDataCreator`. 

```python
NEUTRAL_EXCLUDE = ["i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them"]
neutral = creator.generate_neutral_data(additional_excluded_words=NEUTRAL_EXCLUDE, max_size=15)
```

## 2. Train a GRADIEND model

With the generated data, we can simply train and evaluate a GRADIEND model encoding the targeted feature (*singular-plural*), by using `TextPredictionTrainer`.

```python
from gradiend import TrainingArguments, TextPredictionTrainer

# use minimal training settings
args = TrainingArguments(
    train_batch_size=4,
    eval_steps=5,
    max_steps=25,
    learning_rate=1e-4,
)
trainer = TextPredictionTrainer(
    model="bert-base-uncased", 
    data=training,
    eval_neutral_data=neutral, 
    args=args)
trainer.train()
```
Based on our used toy training configuration, we just train for 25 steps, considering 4 texts of equal feature class (3SG or 3PL), and evaluate every 5 steps. The training stats can be plotted by using `trainer.plot_training_convergence()`. The plot shows training loss and encoder correlation over steps.

The evaluation results during training can be visualized via `trainer.plot_training_convergence()`, which shows the training loss and encoder correlation over steps. The correlation is computed between the encoded gradients and the feature classes (3SG vs 3PL) on the evaluation set, and is expected to increase during training.

![training convergence plot](img/start_workflow_training_convergence.png)


## 3. Evaluate the GRADIEND model

The evaluation of a GRADIEND model consists of two core steps: GRADIEND encoder (to which values are different gradients encoded?) and decoder (can we change the base model's behavior related to the feature?) analysis.

The encoder evaluation uses training data on test split and neutral data, and can be run and plotted using:
```python
enc_result = trainer.evaluate_encoder(plot=True)
print("Correlation:", enc_result.get("correlation"))
```
![Encoded values distribution showing separation of the two feature classes](img/start_workflow_encoder_analysis_split_test.png)

The decoder evaluation evaluates how the base model can be changed by applying a learnt GRADIEND decoder update like $base-model + learning-rate * decoder(feature-factor)$. By default, we evaluate for a range of learning rates and pick the feature factor (+-1) depending on the feature encoding. The selected changed model is chosen with respect to a language model constraint.
```python
dec = trainer.evaluate_decoder()
changed_base_model = trainer.select_changed_model(decoder_results=dec, metric_key="3SG")
```
The `changed_base_model` is expected to be biased towards singular, i.e., assign singular tokens higher probabilities than plural tokens, compared to the (unchanged) base model.


## What you just did

- Used **TextPredictionDataCreator** to build per-class textual training data (3SG: he/she vs 3PL: they) to extract a user-defined feature by feature-related-gradients.
- Trained a GRADIEND model on gradient differences between the two classes defined by the data.
- Ran **encoder evaluation** (correlation, plots) and **decoder evaluation** (probability shifts).
- Called **select_changed_model** to obtain a modified model in memory (best configuration for the chosen metric).

## Next steps

- **[Tutorial: Data generation](tutorials/data-generation.md)** — Build training and neutral data from raw text (syncretism, spaCy, morphology).
- **[Tutorial: Training](tutorials/training.md)** — Experiment layout, pruning, multi-seed, convergence plot, and training options in detail.
- **[Tutorial: Evaluation (intra-model)](tutorials/evaluation-intra-model.md)** — Encoder/decoder evaluation and selecting/saving the changed model.
- **[Tutorial: Evaluation (inter-model)](tutorials/evaluation-inter-model.md)** — Comparing multiple runs (top-k overlap, heatmap).
- **[Detailed workflow (overview)](tutorials/detailed-workflow.md)** — Precomputed vs generated data; how the parts connect in one run.
- **[Data handling](guides/data-handling.md)** — All formats the trainer accepts (HF id, DataFrame, per-class dict, …).
- **[Saving & loading](guides/saving-loading.md)** — Where results are stored and how to reload a model.
- **[Pruning](guides/pruning-guide.md)** — Pre- and post-pruning in depth.
- **[Evaluation & visualization](guides/evaluation-visualization.md)** — Customizing plots.
