# Documentation

## What is GRADIEND?

GRADIEND is a method for **learning features within neural networks** by training an encoder-decoder architecture on model gradients. With this library you can **find where a language model encodes a feature** (e.g. gender, race, religion) and **rewrite the model** to strengthen or weaken it—for example debias it—while keeping other behaviour.

![GRADIEND overview](img/workflow-diagram.png)

GRADIEND works by:
1. **Training an encoder-decoder network** on gradients computed from masked text predictions
2. **Learning a single latent feature neuron** that encodes the desired interpretation (e.g., gender bias)
3. **Using the decoder** to modify the base model's weights, enabling targeted feature manipulation

The method is described in detail in the paper: **[GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](https://arxiv.org/abs/2502.01406)** (ICLR 2026, Drechsel & Herbold, 2025).

> While GRADIEND is methodologically defined to work with any *gradient-learned* and *weight-based* model, this library currently only supports *text prediction* models (specifically `transformers.AutoModelForMaskedLM` and `transformers.AutoModelForCausalLM`).
> However, the library is designed to be modular and we plan to support more models in the future (e.g., text classification, vision, ...).

Example use cases:
- **[English pronouns](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronouns.ipynb)** — notebook; [script](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronouns.py)
- **[English gender](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_en.py)**
- **[German gender–case](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_detailed.py)**
- **[Race and religion](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/race_religion.py)**


---

## Get started

- **[Installation](installation.md)** — Install the package and optional dependencies.
- **[Train your first GRADIEND model](start.md)** — A minimal runnable example to train and evaluate in a few steps.

---

## Tutorials

Step-by-step workflows, following the 5 steps of above's overview:

0. **[Workflow Overview](tutorials/detailed-workflow.md)** — Tutorial Overview.
1. **[Feature Selection and Data Generation](tutorials/data-generation.md)** — Build training and neutral data from raw text (syncretism, spaCy, one filter per grammatical cell). Part 1 of the detailed workflow.
2. **[GRADIEND Training](tutorials/training.md)** — Experiment layout, pruning (pre/post), multi-seed, convergence plot, and training options in detail.
3. **[Intra-Model Evaluation](tutorials/evaluation-intra-model.md)** — Encoder analysis (are target classes seperated?) and decoder evaluation (determine parameters to update model's feature behavior under a language modeling constraint).
4. **[Model Rewrite](tutorials/model-rewrite.md)** — Using decoder-selected settings to rewrite base-model weights in memory or on disk.
5. **[Inter-Model Evaluation](tutorials/evaluation-inter-model.md)** — Comparing multiple runs: top-k overlap and heatmap.

---

## Guides

When you need to understand a topic or look up options:

- **[Core classes and use cases](guides/core-classes.md)** — Overview of the most important classes and when to use them.
- **[Data handling](guides/data-handling.md)** — Data formats, columns, and balancing (DataFrames, per-class dicts, Hugging Face datasets).
- **[Pruning](guides/pruning-guide.md)** — Pre-pruning (from gradients) and post-pruning (from weights); when and how to use them.
- **[Evaluation & visualization](guides/evaluation-visualization.md)** — Encoder and decoder evaluation, convergence and top-k plots, and how to customize plots.
- **[Saving & loading](guides/saving-loading.md)** — Where results are stored and how to reload a trained model.
- **[Training arguments](guides/training-arguments.md)** — Full parameter reference, including multi-seed training and seed report format.
- **[Decoder-only models](guides/decoder-only.md)** — Use causal (decoder-only) LMs with the same API; optional MLM head for better mask gradients.

---

## Reference

- [Examples](examples.md) — All example scripts with short descriptions.
- [API reference](api/index.md) — **Auto-generated** from docstrings; main classes and entry points.
- [FAQ](faq.md) — Troubleshooting and common pitfalls.
- [Citation](citation.md) — How to cite the paper and library.
