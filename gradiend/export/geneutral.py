from datasets import DatasetInfo, GeneratorBasedBuilder, Split, SplitGenerator, load_dataset, Features, Value


class GeneutralDataset(GeneratorBasedBuilder):
    """
    This dataset filters entries from BookCorpus based on provided indices in the Geneutral dataset.
    """

    _CITATION = """
@misc{drechsel2025gradiendmonosemanticfeaturelearning,
  title={{GRADIEND}: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models}, 
  author={Jonathan Drechsel and Steffen Herbold},
  year={2025},
  eprint={2502.01406},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.01406}, 
}
    """

    def _info(self):
        return DatasetInfo(
            description="This dataset consists of BookCorpus entries containing only gender-neutral words (excluding e.g., he, actor, ...).",
            features=Features({
                "index": Value("int32"),
                "text": Value("string"),
            }),
            supervised_keys=None,
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager):
        # URL for your indices file hosted on Hugging Face
        index_file_url = "https://huggingface.co/datasets/aieng-lab/geneutral/resolve/main/indices.csv"

        # Download the indices file
        indices_file = dl_manager.download_and_extract(index_file_url)

        # Load BookCorpus dataset
        print("Loading BookCorpus dataset...")
        bookcorpus = load_dataset('bookcorpus', trust_remote_code=True)['train']
        print("BookCorpus dataset loaded.")

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"indices_file": indices_file, "bookcorpus": bookcorpus}),
        ]

    def _generate_examples(self, indices_file: str, bookcorpus):
        """
        Generate examples by filtering the BookCorpus dataset using provided indices.
        """

        # Load indices from the file
        with open(indices_file, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            indices_set = {int(line.strip().split(",")[0]) for line in f}

        # Filter BookCorpus based on indices and yield examples
        for idx, sample in enumerate(bookcorpus):
            if idx in indices_set:
                yield idx, {"index": idx, "text": sample['text']}
