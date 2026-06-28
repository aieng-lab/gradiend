# Examples

These examples are intended as runnable starting points and inspiration for
workflow variations. They are **not installed with the pip package**; read or
download them from [gradiend/examples](https://github.com/aieng-lab/gradiend/tree/main/gradiend/examples).

Examples in this page are grouped by how a user should approach them. Start with
the quick examples, then move to heavier workflows only after the basic training,
evaluation, and plotting loop is clear.

## Quick examples

- [:material-file-code-outline: `start_workflow.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py) — [Train your first GRADIEND model](start.md)
- [:material-file-code-outline: `train_english_pronouns.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_english_pronouns.py)
- [:material-file-code-outline: `train_gender_de.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de.py)

## Real-data workflows

- [:material-file-code-outline: `train_sentiment.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment.py)
- [:material-file-code-outline: `train_race_symmetric_suite.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_race_symmetric_suite.py) — [Trainer suites](guides/trainer-suites.md)
- [:material-file-code-outline: `train_sentiment_positive_suite.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment_positive_suite.py)
- [:material-file-code-outline: `train_sentiment_positive_suite_all_but_one.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment_positive_suite_all_but_one.py)
- [:material-file-code-outline: `train_multi_seed_stability.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_multi_seed_stability.py) — [Multi-seed](guides/multi-seed.md)
- [:material-file-code-outline: `train_gender_de_decoder_only.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_decoder_only.py)
- [:material-file-code-outline: `train_seq2seq_encoder_mlm.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_seq2seq_encoder_mlm.py)
- [:material-file-code-outline: `train_seq2seq_decoder_sequence.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_seq2seq_decoder_sequence.py) (experimental decoder objective)

## Heavier examples

- [:material-file-code-outline: `train_gender_de_detailed.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.py)
- [:material-file-code-outline: `train_gender_de_detailed.ipynb`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.ipynb)
- [:material-file-code-outline: `train_english_pronouns.ipynb`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_english_pronouns.ipynb)

## Experiments (repo root)

- [:material-file-code-outline: `multilingual_gradiend_demo_small.py`](https://github.com/aieng-lab/gradiend/blob/main/experiments/multilingual_gradiend_demo_small.py) — [Cross-encoding matrix](guides/cross-encoding-matrix.md)
- [:material-file-code-outline: `multilingual_gradiend_demo.py`](https://github.com/aieng-lab/gradiend/blob/main/experiments/multilingual_gradiend_demo.py)

## Diagnostics and visualization

- [:material-file-code-outline: `plot_visualization_gallery.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/plot_visualization_gallery.py)

For documentation plot regeneration, see [docs/img/README.md](img/README.md) and `<!-- DOC_PLOT: ... -->` comments in the guides.

## Experimental inspiration

- [:material-file-code-outline: `train_text_classification_experimental.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_text_classification_experimental.py)
- [:material-file-code-outline: `train_gender_en.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_en.py)

## Data creation

- [:material-file-code-outline: `create_german_article_data.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/create_german_article_data.py)
- [:material-file-code-outline: `create_english_pronoun_data.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/create_english_pronoun_data.py)
