# Core classes and use cases

This guide explains the most important classes in GRADIEND and when to use them. For complete API details, see the [API reference](../api-reference.md).

---

## Overview: the GRADIEND workflow

The typical GRADIEND workflow involves three main components:

1. **Data creation** → `TextPredictionDataCreator` and `TextFilterConfig`
2. **Training** → `TextPredictionTrainer` and `TrainingArguments`
3. **Evaluation** → `Evaluator`, `EncoderEvaluator`, `DecoderEvaluator`

Below, we explain each component and the key classes within them.

---

## Data creation

### `TextPredictionDataCreator`

**Purpose:** Build training and neutral datasets from raw text or existing data sources.

**When to use:** 
- You have raw text and want to extract feature-specific training data
- You need to create masked text pairs for a specific feature (e.g., gender, grammatical case)
- You want to generate neutral evaluation data

**Key methods:**
- `generate_training_data()` — Creates training/validation/test splits with masked texts
- `generate_neutral_data()` — Creates feature-neutral evaluation data

**Example:**
```python
from gradiend import TextPredictionDataCreator, TextFilterConfig

creator = TextPredictionDataCreator(
    base_data=["The chef tasted the soup, then he added pepper."],
    feature_targets=[
        TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
        TextFilterConfig(targets=["they"], id="3PL"),
    ],
)
training = creator.generate_training_data(max_size_per_class=10)
neutral = creator.generate_neutral_data(max_size=15)
```

**See also:** [Data generation tutorial](../tutorials/data-generation.md), [Data handling guide](data-handling.md)

---

### `TextFilterConfig`

**Purpose:** Define a feature class by specifying target tokens and optional linguistic constraints.

**When to use:**
- Defining what tokens/patterns belong to a feature class
- Using spaCy tags for grammatical filtering (e.g., gender, case, number)
- Creating multiple classes for complex features (e.g., German gender–case with 12 classes)

**Key attributes:**
- `targets` — List of tokens to match (e.g., `["he", "she", "it"]`)
- `spacy_tags` — Dictionary of spaCy tags for filtering (e.g., `{"Gender": "Masc", "Case": "Nom"}`)
- `id` — Identifier for the class (used as column names and class labels)

**Example:**
```python
from gradiend import TextFilterConfig

# Simple string matching
config = TextFilterConfig(targets=["he", "she"], id="3SG")

# With spaCy tags (German definite articles)
config = TextFilterConfig(
    targets=["der"],
    spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Masc"},
    id="masc_nom"
)
```

**See also:** [Data generation tutorial](../tutorials/data-generation.md)

---

## Training

### `TextPredictionTrainer`

**Purpose:** Train a GRADIEND model to learn a feature from gradient differences.

**When to use:**
- Training a GRADIEND encoder-decoder on your data
- The main entry point for the GRADIEND training pipeline
- You want to train, evaluate, and save a GRADIEND model

**Key methods:**
- `train()` — Start training
- `evaluate_encoder()` — Evaluate encoder correlation and separation
- `evaluate_decoder()` — Evaluate decoder's ability to modify the base model
- `select_changed_model()` — Get the best modified model from decoder evaluation
- `plot_training_convergence()` — Visualize training progress

**Example:**
```python
from gradiend import TextPredictionTrainer, TrainingArguments

args = TrainingArguments(
    train_batch_size=4,
    max_steps=100,
    learning_rate=1e-4,
)

trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data=training_data,
    eval_neutral_data=neutral_data,
    args=args,
)
trainer.train()

# Evaluate
enc_result = trainer.evaluate_encoder(plot=True)
dec_result = trainer.evaluate_decoder()
changed_model = trainer.select_changed_model(decoder_results=dec_result)
```

**See also:** [Start here](../start.md), [Training tutorial](../tutorials/training.md)

---

### `TrainingArguments`

**Purpose:** Configure training hyperparameters and experiment settings.

**When to use:**
- Setting batch sizes, learning rates, training length
- Configuring multi-seed training
- Enabling pruning (pre/post)
- Setting evaluation frequency

**Key attributes:**
- `train_batch_size`, `eval_batch_size` — Batch sizes
- `max_steps`, `num_train_epochs` — Training length
- `learning_rate` — Learning rate for GRADIEND model
- `eval_steps` — How often to evaluate during training
- `max_seeds` — Number of seeds for multi-seed training
- `pre_prune_config`, `post_prune_config` — Pruning settings

**Example:**
```python
from gradiend import TrainingArguments, PrePruneConfig

args = TrainingArguments(
    train_batch_size=8,
    max_steps=200,
    learning_rate=1e-4,
    eval_steps=20,
    max_seeds=3,  # Train 3 seeds and pick best
    pre_prune_config=PrePruneConfig(keep_top_k=1000),
)
```

**See also:** [Training arguments guide](training-arguments.md), [Pruning guide](pruning-guide.md)

---

### `TextPredictionConfig`

**Purpose:** Configure data loading and preprocessing for `TextPredictionTrainer`.

**When to use:**
- Loading data from HuggingFace datasets
- Specifying column names for custom data formats
- Configuring target classes and data splits

**Key attributes:**
- `data` — Training data (DataFrame, dict, HF dataset ID, or file path)
- `hf_dataset` — HuggingFace dataset ID
- `target_classes` — Classes to train on (pair is auto-inferred if len=2)
- `masked_col`, `split_col` — Column names
- `eval_neutral_data` — Neutral evaluation data

**Example:**
```python
from gradiend import TextPredictionConfig

config = TextPredictionConfig(
    data="aieng-lab/de-gender-case-articles",
    target_classes=["masc_nom", "fem_nom"],
    masked_col="masked",
    split_col="split",
    eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
)
```

**See also:** [Data handling guide](data-handling.md)

---

## Model classes

### `GradiendModel`

**Purpose:** The core encoder-decoder model (weights-only, no base model context).

**When to use:**
- Low-level access to GRADIEND encoder/decoder weights
- Saving/loading GRADIEND models independently
- Computing importance scores from weights

**Key methods:**
- `forward()` — Encode gradients to latent space
- `forward_decoder()` — Decode from latent space to gradient space
- `save_pretrained()` — Save model weights and config
- `from_pretrained()` — Load a saved model

**Note:** Most users should use `ModelWithGradiend` or `TextPredictionTrainer` instead, which handle the base model integration.

**See also:** [API reference](../api-reference.md#gradiendmodel)

---

### `ParamMappedGradiendModel`

**Purpose:** GRADIEND model with parameter mapping for base-model gradients.

**When to use:**
- Working with gradient dictionaries (parameter name → gradient tensor)
- Need to map between GRADIEND's parameter space and base model parameters
- Advanced use cases requiring direct gradient manipulation

**Key difference from `GradiendModel`:** Handles parameter name mapping, enabling gradient I/O as dictionaries.

**See also:** [API reference](../api-reference.md#parammappedgradiendmodel)

---

### `ModelWithGradiend`

**Purpose:** Wrapper combining a base language model with a GRADIEND encoder-decoder.

**When to use:**
- Loading a trained GRADIEND model with its base model
- Applying decoder updates to modify the base model
- Evaluating encoder/decoder on a loaded model

**Key methods:**
- `encode()` — Encode gradients to latent feature value
- `modify_model()` — Apply decoder update to create a modified base model
- `from_pretrained()` — Load base model + GRADIEND model

**Example:**
```python
from gradiend import ModelWithGradiend

model = ModelWithGradiend.from_pretrained(
    base_model="bert-base-uncased",
    gradiend_model="path/to/gradiend/model",
)

# Encode a gradient
feature_value = model.encode(gradient_dict)

# Modify the base model
modified_model = model.modify_model(
    learning_rate=1e-4,
    feature_factor=1.0,
    part='decoder'
)
```

**See also:** [API reference](../api-reference.md#modelwithgradiend), [Saving & loading guide](saving-loading.md)

---

## Evaluation

### `Evaluator`

**Purpose:** High-level evaluation coordinator bound to a trainer.

**When to use:**
- Running encoder and decoder evaluation together
- Convenient access to evaluation methods from a trainer
- Default evaluation workflow

**Key methods:**
- `evaluate_encoder()` — Run encoder evaluation (correlation, plots)
- `evaluate_decoder()` — Run decoder evaluation (probability shifts)
- Delegates plotting to `Visualizer` if configured

**Note:** `TextPredictionTrainer` already has an `Evaluator` instance, so you typically call `trainer.evaluate_encoder()` directly.

**See also:** [API reference](../api-reference.md), [Evaluation tutorial](../tutorials/evaluation-intra-model.md)

---

### `EncoderEvaluator`

**Purpose:** Evaluate how well the encoder separates feature classes.

**When to use:**
- Computing correlation between encoded values and feature classes
- Analyzing encoder separation on test/neutral data
- Debugging encoder convergence

**Key outputs:**
- Correlation coefficient (higher = better separation)
- Encoded value distributions per class
- Plots showing class separation

**See also:** [API reference](../api-reference.md), [Evaluation guide](evaluation-visualization.md)

---

### `DecoderEvaluator`

**Purpose:** Evaluate how well the decoder can modify the base model.

**When to use:**
- Testing if decoder updates change model behavior as expected
- Finding optimal learning rate and feature factor
- Measuring probability shifts for target tokens

**Key outputs:**
- Grid search results over learning rates and feature factors
- Probability shift scores
- Language modeling scores (to ensure model quality is maintained)

**See also:** [API reference](../api-reference.md), [Evaluation tutorial](../tutorials/evaluation-intra-model.md)

---

## Utility classes

### `TextPreprocessConfig`

**Purpose:** Configure text preprocessing (lowercasing, tokenization, etc.).

**When to use:**
- Normalizing text before filtering/masking
- Handling case sensitivity
- Configuring tokenization options

**Key attributes:**
- `lowercase` — Whether to lowercase text
- `remove_punctuation` — Whether to remove punctuation
- `normalize_whitespace` — Whether to normalize whitespace

**See also:** [Data generation tutorial](../tutorials/data-generation.md)

---

### `PrePruneConfig` / `PostPruneConfig`

**Purpose:** Configure pruning to reduce model size and focus on important parameters.

**When to use:**
- Reducing computational cost
- Focusing on most important parameters
- Speeding up training

**Key attributes:**
- `keep_top_k` — Number of parameters to keep
- `importance_metric` — How to compute importance (e.g., "gradient_norm", "weight_magnitude")

**Pre-pruning:** Prune before training based on gradient statistics.  
**Post-pruning:** Prune after training based on learned weights.

**See also:** [Pruning guide](pruning-guide.md)

---

## Quick reference: which class to use?

| Task | Primary class | Secondary classes |
|------|---------------|-------------------|
| **Create training data from text** | `TextPredictionDataCreator` | `TextFilterConfig`, `TextPreprocessConfig` |
| **Load precomputed data** | `TextPredictionTrainer` (via `data` parameter) | `TextPredictionConfig` |
| **Train a GRADIEND model** | `TextPredictionTrainer` | `TrainingArguments`, `TextPredictionConfig` |
| **Evaluate encoder** | `TextPredictionTrainer.evaluate_encoder()` | `EncoderEvaluator` |
| **Evaluate decoder** | `TextPredictionTrainer.evaluate_decoder()` | `DecoderEvaluator` |
| **Load a trained model** | `ModelWithGradiend.from_pretrained()` | `GradiendModel.from_pretrained()` |
| **Modify a model** | `ModelWithGradiend.modify_model()` | `TextPredictionTrainer.select_changed_model()` |
| **Configure pruning** | `PrePruneConfig`, `PostPruneConfig` | Used via `TrainingArguments` |

---

## Next steps

- **[Start here](../start.md)** — Run a complete example
- **[API reference](../api-reference.md)** — Complete API documentation
- **[Tutorials](../tutorials/data-generation.md)** — Step-by-step workflows
- **[Guides](data-handling.md)** — Detailed guides for specific topics
