# GRADIEND demo notebooks

Run them in this order, or pick one. They build on each other: the first introduces the workflow, the next two vary it, the last shows how to learn a feature of your own.
A GPU makes training take minutes; the saved outputs show a full run. All notebooks open directly in Colab (choose a GPU runtime).

1. [`01_quickstart_english_pronouns.ipynb`](01_quickstart_english_pronouns.ipynb) ([Colab](https://colab.research.google.com/github/aieng-lab/gradiend/blob/main/gradiend/examples/demo/01_quickstart_english_pronouns.ipynb)): create data, train, check and rewrite with GRADIEND (singular vs plural pronouns in BERT).
2. [`02_sentiment.ipynb`](02_sentiment.ipynb) ([Colab](https://colab.research.google.com/github/aieng-lab/gradiend/blob/main/gradiend/examples/demo/02_sentiment.ipynb)): a feature without a closed word list: positive vs negative sentiment, evaluated on held-out words.
3. [`03_compare_features.ipynb`](03_compare_features.ipynb) ([Colab](https://colab.research.google.com/github/aieng-lab/gradiend/blob/main/gradiend/examples/demo/03_compare_features.ipynb)): compare 45 trained GRADIEND models from the Hugging Face Hub (top-k overlap and cross-encoding). No training.
4. [`04_learn_your_own_feature.ipynb`](04_learn_your_own_feature.ipynb) ([Colab](https://colab.research.google.com/github/aieng-lab/gradiend/blob/main/gradiend/examples/demo/04_learn_your_own_feature.ipynb)): turn a feature of your own into data and learn it, with an automatic learning rate. The example is British vs American spelling.

```
pip install gradiend[recommended]
```

Each training notebook defines the base model once at the top (`MODEL`); any masked language model works.
