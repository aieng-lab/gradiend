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
- [english_pronoun_singular_plural.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronoun_singular_plural.py) — English pronouns (3SG vs 3PL); optional class_merge_map for number/person.
