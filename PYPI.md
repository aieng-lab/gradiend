# GRADIEND

**Gradient-based targeted feature learning within neural networks.** Learn where a language model encodes a feature (e.g. gender, race) and rewrite the model to strengthen or weaken it—for example debias it—while keeping other behaviour unchanged.

[![arXiv:2602.23993](https://img.shields.io/badge/arXiv-2602.23993-blue.svg)](https://arxiv.org/abs/2602.23993)
[![arXiv:2502.01406](https://img.shields.io/badge/arXiv-2502.01406-blue.svg)](https://arxiv.org/abs/2502.01406)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/aieng-lab/gradiend/actions/workflows/tests.yml/badge.svg)](https://github.com/aieng-lab/gradiend/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://aieng-lab.github.io/gradiend/)

Paper: 

- [GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](https://arxiv.org/abs/2502.01406) 
- [The GRADIEND Python Package: An End-to-End System for Gradient-Based Feature Learning](https://arxiv.org/abs/2602.23993)

---

## Install

```bash
pip install gradiend
```

Optional extras (install one or combine with e.g. `gradiend[data,plot]`):

| Tag | Command | Includes                                     |
|-----|---------|----------------------------------------------|
| **recommended** | `pip install gradiend[recommended]` | full data generation, training, and plotting |
| **data** | `pip install gradiend[data]` | for advanced data generation and import |
| **plot** | `pip install gradiend[plot]` | for plotting support |                                                                                   |
| **dev** | `pip install gradiend[dev]` | for building docs and running tests |

---

## Links

| | |
|---|---|
| **Documentation** | [aieng-lab.github.io/gradiend](https://aieng-lab.github.io/gradiend/) |
| **Source & issues** | [github.com/aieng-lab/gradiend](https://github.com/aieng-lab/gradiend) |
| **Example scripts & notebooks** | [gradiend/examples](https://github.com/aieng-lab/gradiend/tree/main/gradiend/examples) — start_workflow, train_english_pronouns, gender_de, gender_en, train_race_religion, etc. |
| **Datasets (Hugging Face)** | [de-gender-case-articles](https://huggingface.co/datasets/aieng-lab/de-gender-case-articles), [gradiend_race_data](https://huggingface.co/datasets/aieng-lab/gradiend_race_data), [gradiend_religion_data](https://huggingface.co/datasets/aieng-lab/gradiend_religion_data), [biasneutral](https://huggingface.co/datasets/aieng-lab/biasneutral), [geneutral](https://huggingface.co/datasets/aieng-lab/geneutral) |
| **Pre-trained GRADIEND models** | [bert-base-cased-gradiend-gender-debiased](https://huggingface.co/aieng-lab/bert-base-cased-gradiend-gender-debiased), [gpt2-gradiend-gender-debiased](https://huggingface.co/aieng-lab/gpt2-gradiend-gender-debiased), [Llama-3.2-3B-gradiend-gender-debiased](https://huggingface.co/aieng-lab/Llama-3.2-3B-gradiend-gender-debiased) |

---

## Citation

The Python package paper:
```bibtex
@misc{drechsel2026gradiendpythonpackage,
      title={The {GRADIEND} Python Package: An End-to-End System for Gradient-Based Feature Learning}, 
      author={Jonathan Drechsel and Steffen Herbold},
      year={2026},
      eprint={2602.23993},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.23993}, 
}
```

The original GRADIEND method paper:
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

---

## License

Apache 2.0. See [LICENSE](https://github.com/aieng-lab/gradiend/blob/main/LICENSE) on GitHub.
