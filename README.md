# GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models
> Jonathan Drechsel, Steffen Herbold
[![arXiv](https://img.shields.io/badge/arXiv-2502.01406-blue.svg)](https://arxiv.org/abs/2502.01406)


This repository contains the official source code for the training and evaluation of [GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models](https://arxiv.org/abs/2502.01406).
Further evaluations of this study can be reproduced using our [expanded version of bias-bench](https://github.com/aieng-lab/bias-bench).

## Quick Links
- [GRADIEND Paper](https://arxiv.org/abs/2502.01406)
- GRADIEND Training and Evaluation Datasets:
  - [GENTER](https://huggingface.co/datasets/aieng-lab/genter)
  - [GENEUTRAL](https://huggingface.co/datasets/aieng-lab/geneutral)
  - [GENTYPES](https://huggingface.co/datasets/aieng-lab/gentypes)
  - [NAMEXACT](https://huggingface.co/datasets/aieng-lab/namexact)
  - [NAMEXTEND](https://huggingface.co/datasets/aieng-lab/namextend)
- GRADIEND Gender Debiased Models:
  - [`aieng-lab/bert-base-cased-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/bert-base-cased-gradiend-gender-debiased)
  - [`aieng-lab/bert-large-cased-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/bert-large-cased-gradiend-gender-debiased)
  - [`aieng-lab/distilbert-base-cased-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/distilbert-base-cased-gradiend-gender-debiased)
  - [`aieng-lab/roberta-large-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/roberta-large-gradiend-gender-debiased)
  - [`aieng-lab/gpt2-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/gpt2-gradiend-gender-debiased)
  - [`aieng-lab/Llama-3.2-3B-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/Llama-3.2-3B-gradiend-gender-debiased)
  - [`aieng-lab/Llama-3.2-3B-Instruct-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/Llama-3.2-3B-Instruct-gradiend-gender-debiased)
- Relevant Repositories:
  - [`aieng-lab/bias-bench`](https://github.com/aieng-lab/bias-bench) for evaluation
  - [`aieng-lab/lm-eval-harness`](https://github.com/aieng-lab/lm-eval-harness) for GLUE zero-shot evaluation


## Install
```bash
git clone https://github.com/aieng-lab/gradiend.git
cd gradiend
conda env create --file environment.yml
conda activate gradiend
```

Download [Gendered Words](https://github.com/ecmonsen/gendered_words) and copy the file into the `data/` directory of this repository.

Optional: Install [`aieng-lab/bias-bench`](https://github.com/aieng-lab/bias-bench) for further evaluations and comparison to other debiasing techniques.

In order to use Llama-based models, you must first accept the Llama 3.2 Community License Agreement (see e.g., [here](https://huggingface.co/meta-llama/Llama-3.2-3B)). Further, you need to export a variable `HF_TOKEN` with a HF access token associated to your HF account (alternatively, but not recommended, you could insert your HF token in `gradiend/model.py#HF_TOKEN`).

## Overview

Package | Description
--------|------------
`gradiend.model` | GRADIEND model implementation
`gradiend.data` | Data generation and access
`gradiend.training` | Training of GRADIEND
`gradiend.evaluation` | Evaluation of GRADIEND
`gradiend.export` | Export functions for results, e.g., printing LaTeX tables and plotting images

> **__NOTE:__** All python files of this repository should be called from the root directory of the project to ensure that the correct (relative) paths are used (e.g., `python gradiend/training/gradiend_training.py`).

See `demo.ipynb` for a quick overview of the GRADIEND model and the evaluation process.

### Data
The `gradiend.data` package provides two purposes:
- Data access: The relevant datasets can be accessed via the `read_[dataset]()` functions, i.e., `read_genter()`, `read_geneutral()`, `read_namexact()`, `read_namextend()`, and `read_gentypes()`.
- Data generation: The generation process of these datasets is not necessary for the GRADIEND training (as the datasets are already generated), but the code is still available in the `data` package (see below *Dataset Generation*).

### Training

The training of the GRADIEND models is done by running the `gradiend.training.gradiend_training` script, which will train three GRADIENDs for each considered base model (`bert-base-cased`, `bert-large-cased`, `distilbert-base-cased`, `roberta-large`), selecting the best model at the end.
Intermediate results are saved in `results/experiments/gradiend`, and the final models are saved in `results/models`.
The `gradiend_training` script relies on:
- `gradiend.training.data`: the `TrainingDataset` class combines several datasets (e.g., GENTER, NAMEXACT, ...) and contains  the logic to create appropriate training data during the training, i.e., matching a GENTER template sentence with a name of a certain gender and computing the tokens. 
- `gradiend.training.trainer`: the `train()` function trains a single GRADIEND model and provides many hyperparameters

### Evaluation

#### Analysis of Encoder
The `gradiend.evaluation.analyze_encoder.analyze()` function analyzes the encoder of a trained GRADIEND model with three dataset:

- GENTER as in the training process
- GENTER with correctly filled template tokens, and with masked tokens that are gender-neutral
- GENEUTRAL

This function can be easily called for multiple models by calling `gradiend.evaluation.analyze_encoder.analyze_models(*models)`. The raw results are saved in the same base folder as the GRADIEND model (e.g., `results/models/bert-base-cased_params_spl_test.csv`). 
Then, the model metrics can be generated and printed by calling `gradiend.evaluation.analyze_encoder.print_all_models()`.

#### Analysis of Decoder and Generation of (De-)Biased Models

`gradiend.evaluation.analyze_decoder.default_evaluation()` evaluates the decoder of a trained GRADIEND model by generating debiased models for different learning rates and gender factors.
The evaluation results are cached per learning rate and gender factor (`results/cache/decoder`), and plots are shown visualizing the results.

The best debiased, male-biased, and female-biased models according to this evaluation can be generated by executing the `gradiend.evaluation.select_models` script, which saves these models into `results/changed_models`. The models are names `[base model]-[type]`, with type being `N` for the debiased model, `F` for the female model, and `M` for the male model.

Some basic evaluations of these debiased models can be done by calling:
- `gradiend.analyze_decoder.evaluate_all_gender_predictions()` and `gradiend.export.gender_predictions.py` for an overfitting analysis
- `gradiend.export.example_predictions.py` to generate example predictions

### Evaluation of (De-)Biased Models
See [bias-bench](https://github.com/aieng-lab/bias-bench) for a comparison of the (de-)biased models generated with GRADIEND to other debiasing techniques.

### Export
The export package contains functions to export the results of the evaluations, e.g., to print LaTeX tables or to plot images.

Script | Description
-------|------------
`dataset_stats` | prints the statistics of the datasets used in the paper
`encoder_plot` | Plots a violin plot regarding the distribution of encoded values of the encoder analysis
`changed_model_selection` | Generates a table with the statistics of the selected (de-) biased models (from `gradiend.evaluation.analyze_decoder.default_evaluation()`
`gender_predictions` | Plots predicted female and male probabilities for simple masking task to evaluate overfitting
`example_predictions` | Generates example predictions for the selected (de-) biased as a LaTeX table

> **__NOTE:__** To enable LaTeX plotting with your desired font, you need to adjust the `init_matplotlib()` function default arguments in the gradiend.util.py` file.

## Dataset Generation

Although the experiments mentioned above are based on data published on Hugging Face by now, we also provide the code to 
generate the datasets used in the paper.

### Required Datasets
If you want to re-create the datasets generated in the paper, you first need to download the following datasets:

Dataset | Download Link | Notes                                            | Download Directory
--------|---------------|--------------------------------------------------|-------------------
Gender by Name | [Download](https://doi.org/10.24432/C55G7X) | Required for the generation of the name datasets | `data/`

### Dataset Generation
The following scripts will generate the datasets used in the paper:

Dataset | Generation Script
--------|------------------
GENTER  | `gradiend.data.filtering.generate_genter()`
GENEUTRAL | `gradiend.data.generate_geneutral()`
NAMEXACT | `gradiend.data.generate_namexact()`
NAMEXTEND | `gradiend.data.generate_namextend()`

## Citation
```
@misc{drechsel2025gradiendmonosemanticfeaturelearning,
      title={{GRADIEND}: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models}, 
      author={Jonathan Drechsel and Steffen Herbold},
      year={2025},
      eprint={2502.01406},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01406}, 
}
```
