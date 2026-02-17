# Installation

## Requirements

**Python 3.8 or newer** is required (tested on 3.8–3.11).

## Basic installation

```bash
pip install gradiend
```

This installs the core package and required dependencies. Sufficient for training with DataFrames or local data.

## Recommended (plots, HuggingFace, safetensors)

For a full experience (plots, loading HuggingFace datasets, safetensors), install:

```bash
pip install gradiend[recommended]
```

This adds:

| Package     | Purpose                                                                 |
|-------------|-------------------------------------------------------------------------|
| matplotlib  | Plotting (encoder distributions, convergence plots)                     |
| seaborn     | Visualizations (encoder scatter, heatmaps)                             |
| safetensors | Faster, safer model serialization (preferred over `.bin`)              |
| datasets    | Loading HuggingFace datasets by id |

## Optional: data creation (spaCy)

To **create training data** from raw text with morphological filtering (e.g. German articles by gender/case), install:

```bash
pip install gradiend[data]
```

This adds:

| Package  | Purpose                                                                 |
|----------|-------------------------------------------------------------------------|
| spacy    | Morphological filtering via `spacy_tags` in `TextFilterConfig` (see [Data generation](tutorials/data-generation.md)) |
| datasets | Loading HuggingFace datasets as base data for `TextPredictionDataCreator` |

spaCy also needs a language model, e.g. `de_core_news_sm` for German: `python -m spacy download de_core_news_sm`.

> You can combine extras: `pip install gradiend[recommended,data]` for plots, HF, safetensors, and data creation.

## Optional: interactive encoder scatter (Plotly)

The encoder scatter plot (`trainer.plot_encoder_scatter()`) uses Plotly for interactive hover and zoom. It is **optional** and **not** part of `recommended`. Without Plotly, the function returns `None` and logs a warning.

To enable the interactive scatter (e.g. in Jupyter):

```bash
pip install plotly
```

## Dev (contributors)

For building docs and running tests:

```bash
pip install -e ".[recommended,dev]"
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
