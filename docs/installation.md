# Installation

## Requirements

**Python 3.9 or newer** is required. The test suite is regularly exercised on
current Python 3.10/3.11 environments.

## Basic installation

```bash
pip install gradiend
```

This installs the core package and required dependencies. It is enough when you
train from local files or pandas DataFrames and do not need optional plotting,
Hugging Face dataset loading, or large-model device mapping.

## Recommended

For normal research use, install the recommended extras:

```bash
pip install gradiend[recommended]
```

This adds support for common plotting workflows, Hugging Face datasets,
safetensors checkpoints, tokenizer backends, and large-model loading with
`device_map` / `base_model_device_map`.

| Package | Purpose |
|---------|---------|
| `accelerate` | Hugging Face `device_map` / `base_model_device_map` loading for large models |
| `matplotlib` | Static plots such as convergence curves, encoder distributions, strip plots, and heatmaps |
| `seaborn` | Higher-level statistical visualizations |
| `datasets` | Loading Hugging Face datasets by id |
| `safetensors` | Preferred model serialization format |
| `sentencepiece` | Tokenizer backend required by many T5/LLaMA-style tokenizers |

## Optional: data creation (spaCy)

To create training data from raw text with morphological filtering, install:

```bash
pip install gradiend[data]
```

This adds:

| Package | Purpose |
|---------|---------|
| `spacy` | Morphological filtering via `spacy_tags` in [`TextFilterConfig`][gradiend.data.text.filter_config.TextFilterConfig] |
| `datasets` | Loading Hugging Face datasets as base data for [`TextPredictionDataCreator`][gradiend.data.text.prediction.creator.TextPredictionDataCreator] |

spaCy also needs a language model. For German filtering, for example:

```bash
python -m spacy download de_core_news_sm
```

You can combine extras:

```bash
pip install gradiend[recommended,data]
```

## Interactive encoder scatter (Plotly)

The interactive encoder scatter plot ([`trainer.plot_encoder_scatter()`][gradiend.trainer.trainer.Trainer.plot_encoder_scatter]) uses
Plotly for hover labels, zooming, and notebook exploration. Plotly is installed
by `gradiend[recommended]` and `gradiend[plot]`. Without Plotly, the function
returns `None` and logs a warning.

To enable the interactive scatter in a minimal install:

```bash
pip install gradiend[plot]
```

## Choosing extras

| Task | Suggested install |
|------|-------------------|
| Run the quick start from local data | `pip install gradiend[recommended]` |
| Generate text-prediction data from raw corpora | `pip install gradiend[recommended,data]` |
| Use interactive scatter plots in notebooks | `pip install gradiend[recommended]` |
| Develop GRADIEND itself | `pip install -e ".[recommended,data,dev]"` |

## Dev (contributors)

For building docs and running tests:

```bash
pip install -e ".[recommended,data,dev]"
```

## From source

```bash
git clone https://github.com/aieng-lab/gradiend.git
cd gradiend
pip install -e .
# With recommended extras:
pip install -e ".[recommended]"
```

## Verify install

```bash
python -c "import gradiend; print(gradiend.__version__)"
```
