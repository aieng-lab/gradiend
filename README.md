# GRADIEND

[![arXiv:2502.01406](https://img.shields.io/badge/arXiv-2502.01406-blue.svg)](https://arxiv.org/abs/2502.01406)
[![arXiv:2601.09313](https://img.shields.io/badge/arXiv-2601.09313-blue.svg)](https://arxiv.org/abs/2601.09313)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/aieng-lab/gradiend/actions/workflows/tests.yml/badge.svg)](https://github.com/aieng-lab/gradiend/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://aieng-lab.github.io/gradiend/)

GRADIEND (Gradient-based targeted feature learning within neural networks) learns features inside language models by training an encoder-decoder on gradients. Find where a model encodes a feature (e.g. gender, race) and rewrite the model to strengthen or weaken it—for example debias it—while keeping other behaviour. See [GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](https://arxiv.org/abs/2502.01406).

## Installation

```bash
pip install gradiend
```

With plotting, HuggingFace datasets, and safetensors:

```bash
pip install gradiend[recommended]
```

From source:

```bash
git clone https://github.com/aieng-lab/gradiend.git
cd gradiend
pip install -e ".[recommended]"
```

## Quick start

Use enough base sentences so you get at least `train_batch_size` samples per class in the training split (and a non-empty test split for evaluation). Example:

```python
from gradiend import TextPredictionDataCreator, TextFilterConfig, TrainingArguments, TextPredictionTrainer

base = [
    "The chef tasted the soup, then he added pepper.",
    "The players ran; they scored.",
    "She handed the package to the courier and asked them to deliver it.",
    "The committee met on Tuesday and they voted to postpone the decision.",
    "He left the book on the table and she noticed the door was open.",
    "The birds gathered on the wire; when the cat moved they flew away.",
    "The mechanic wiped his hands and said the car would be ready; he had fixed it.",
    "They invited her to the meeting and she accepted.",
    "The dog ran to the door; it wanted to go outside.",
    "The volunteers packed the boxes and said they would load the van at dawn.",
]
creator = TextPredictionDataCreator(
    base_data=base,
    feature_targets=[
        TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
        TextFilterConfig(targets=["they"], id="3PL"),
    ],
)
training = creator.generate_training_data(max_size_per_class=20)
neutral = creator.generate_neutral_data(additional_excluded_words=["he", "she", "it", "they"], max_size=15)

args = TrainingArguments(train_batch_size=4, eval_steps=5, max_steps=25, learning_rate=1e-4)
trainer = TextPredictionTrainer(model="bert-base-uncased", data=training, eval_neutral_data=neutral, args=args)
trainer.train()

enc_result = trainer.evaluate_encoder(plot=True)
dec = trainer.evaluate_decoder()
changed_model = trainer.select_changed_model(decoder_results=dec, metric_key="3SG")
```

Runnable script with more data: `python -m gradiend.examples.start_workflow`

## Documentation

- [Documentation](https://aieng-lab.github.io/gradiend/) (when published)
- [Installation details](docs/installation.md)
- [Start here](docs/start.md) — minimal workflow
- [Tutorials](docs/index.md#tutorials)
- [API reference](docs/api-reference.md)

## Examples

- [start_workflow.py](gradiend/examples/start_workflow.py) — Minimal runnable example
- [gender_de.py](gradiend/examples/gender_de.py) — German gender (masc_nom vs fem_nom)
- [gender_en.py](gradiend/examples/gender_en.py) — English gender with name augmentation
- [gender_de_decoder_only.py](gradiend/examples/gender_de_decoder_only.py) — Decoder-only model with optional MLM head
- [race_religion.py](gradiend/examples/race_religion.py) — Race and religion bias

## Datasets and models

**Datasets (Hugging Face):** [de-gender-case-articles](https://huggingface.co/datasets/aieng-lab/de-gender-case-articles), [gradiend_race_data](https://huggingface.co/datasets/aieng-lab/gradiend_race_data), [gradiend_religion_data](https://huggingface.co/datasets/aieng-lab/gradiend_religion_data), [biasneutral](https://huggingface.co/datasets/aieng-lab/biasneutral), [geneutral](https://huggingface.co/datasets/aieng-lab/geneutral), and more.

**Pre-trained GRADIEND models:** [bert-base-cased-gradiend-gender-debiased](https://huggingface.co/aieng-lab/bert-base-cased-gradiend-gender-debiased), [gpt2-gradiend-gender-debiased](https://huggingface.co/aieng-lab/gpt2-gradiend-gender-debiased), [Llama-3.2-3B-gradiend-gender-debiased](https://huggingface.co/aieng-lab/Llama-3.2-3B-gradiend-gender-debiased), and others.

## Citation

```bibtex
@misc{drechsel2025gradiend,
  title={{GRADIEND}: Feature Learning within Neural Networks Exemplified through Biases},
  author={Jonathan Drechsel and Steffen Herbold},
  year={2025},
  eprint={2502.01406},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.01406},
}
```

For the German definite articles study using GRADIEND:

```bibtex
@misc{drechsel2026understanding,
  title={Understanding or Memorizing? A Case Study of German Definite Articles in Language Models},
  author={Jonathan Drechsel and Erisa Bytyqi and Steffen Herbold},
  year={2026},
  eprint={2601.09313},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.09313},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
