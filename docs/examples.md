# Examples

These examples are intended as inspiration for data handling and workflow variations. Links point to the source in the repository.

## Quick start

- [start_workflow.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py) — Minimal start-to-end workflow with `TextPredictionDataCreator`. Uses 75+ artificial sentences (3SG/3PL), creates data on the fly, trains, and evaluates. Matches [docs/start.md](start.md).

## Training workflows

- [gender_de.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de.py) — Gender bias in German (minimal workflow, single pair).
- [gender_de_detailed.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_detailed.py) — German gender workflow for various target class combinations with pruning, encoder plots, top-k overlap heatmap, and training convergence.
- [gender_de_decoder_only.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_decoder_only.py) — Decoder-only model with optional MLM head (`DecoderModelWithMLMHead`).
- [gender_en.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_en.py) — Gender bias in English with name augmentation and GENTypes-based decoder metrics (BPI, FPI, MPI).
- [race_religion.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/race_religion.py) — Race and religion bias (multi-class, multiple bias types in a loop).
- [english_pronouns.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronouns.py) — English pronouns (3SG vs 3PL); loads data from `data_creation_pronouns` output. Optional `class_merge_map` for number/person (e.g. singular vs plural). See also the [english_pronouns.ipynb](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronouns.ipynb) notebook for a step-by-step run (data creation from Wikipedia → training → evaluation).

## Data creation

- [data_creator_demo.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/data_creator_demo.py) — Build training and neutral data with `TextPredictionDataCreator` (German articles der/die/das, syncretism handling).
- [data_creation_pronouns.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/data_creation_pronouns.py) — Create English pronoun data (1SG, 1PL, 2, 3SG, 3PL) from Wikipedia via HuggingFace; used by `english_pronouns.py`.

## Notebooks (interactive)

Interactive Jupyter notebooks for step-by-step workflow discovery:

- [gender_de_detailed.ipynb](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_detailed.ipynb) — Start with a single gender-case pair, explore each step, then optionally loop over configs and compare with top-k overlap heatmaps.
- [english_pronouns.ipynb](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronouns.ipynb) — Data creation from Wikipedia → training 3SG vs 3PL → evaluation, with optional `class_merge_map` for singular vs plural.
