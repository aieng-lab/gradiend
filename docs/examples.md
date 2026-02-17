# Examples

These examples are intended as inspiration for data handling and workflow variations. Links point to the source in the repository.

## Training workflows

- [gender_de.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de.py) — Gender bias in German (minimal workflow).
- [gender_de_detailed.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_detailed.py) — German gender workflow for various combination of target classes with pruning and plots.
- [gender_de_decoder_only.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_decoder_only.py) — Decoder-only model and optional MLM head.
- [gender_en.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_en.py) — Gender bias in English.
- [race_religion.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/race_religion.py) — Race and religion bias (multi-class).

## Data creation

- [data_creator_demo.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/data_creator_demo.py) — Build training and neutral data with `TextPredictionDataCreator`.
- [pronoun_workflow.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/pronoun_workflow.py) — Pronoun-focused data creation and training.
