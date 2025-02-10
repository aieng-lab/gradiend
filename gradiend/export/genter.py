import re

from datasets import DatasetInfo, GeneratorBasedBuilder, Split, SplitGenerator, load_dataset, Features, Value


class GenterDataset(GeneratorBasedBuilder):
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
            description="This dataset consists of template sentences associating first names ([NAME]) with third-person singular pronouns ([PRONOUN])",
            features=Features({
                "index": Value("int32"),
                "text": Value("string"),
                "masked": Value("string"),
                "label": Value("string"),
                "name": Value("string"),
                "pronoun": Value("string"),
                "pronoun_count": Value("uint8"),
            }),
            supervised_keys=None,
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager):
        # URL for your indices file hosted on Hugging Face
        indices_files = {}
        for split in ["train", "val", "test"]:
            index_file_url = f"https://huggingface.co/datasets/aieng-lab/genter/resolve/main/genter_indices_{split}.csv"
            # Download the indices file
            indices_file = dl_manager.download_and_extract(index_file_url)
            indices_files[split] = indices_file

        # Load BookCorpus dataset
        print("Loading BookCorpus dataset...")
        bookcorpus = load_dataset('bookcorpus', trust_remote_code=True)['train']
        print("BookCorpus dataset loaded.")

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"indices_file": indices_files['train'], "bookcorpus": bookcorpus}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"indices_file": indices_files['val'], "bookcorpus": bookcorpus}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"indices_file": indices_files['test'], "bookcorpus": bookcorpus}),
        ]

    def _generate_examples(self, indices_file: str, bookcorpus):
        """
        Generate examples by filtering the BookCorpus dataset using provided indices.
        """
        try:
            import pandas as pd
            df = pd.read_csv(indices_file)
        except ImportError:
            raise ImportError("Please install pandas to generate GENTER.")

            # Filter BookCorpus based on indices and yield examples
        for _, sample in df.iterrows():
            idx = sample['index']
            name = sample['name']
            pronoun = sample['pronoun']
            assert pronoun in {'he', 'she'}
            label = 'M' if pronoun == 'she' else 'F'

            text = bookcorpus[idx]['text']

            masked = re.sub(rf'\b{name}\b', '[NAME]', text)

            pronoun_count = len(re.findall(rf'\b{pronoun}\b', text))
            masked = re.sub(rf'\b{pronoun}\b', '[PRONOUN]', masked)

            yield idx, {
                "index": idx,
                "text": text,
                "masked": masked,
                "label": label,
                "name": name,
                "pronoun": pronoun,
                "pronoun_count": pronoun_count,
            }