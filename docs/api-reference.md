# API reference

The following pages are **auto-generated from the source code** (docstrings and type hints) via [mkdocstrings](https://mkdocstrings.github.io/). They reflect the public API of the `gradiend` package.

---

## Package overview

::: gradiend
    options:
      show_root_heading: false
      show_symbol_type_heading: true
      heading_level: 2
      members: false

---

## Core models

<span id="gradiendmodel"></span>
::: gradiend.model.model.GradiendModel
    options:
      show_root_heading: false
      heading_level: 2

<span id="parammappedgradiendmodel"></span>
::: gradiend.model.param_mapped.ParamMappedGradiendModel
    options:
      show_root_heading: false
      heading_level: 2

<span id="modelwithgradiend"></span>
::: gradiend.model.model_with_gradiend.ModelWithGradiend
    options:
      show_root_heading: false
      heading_level: 2

---

## Data creation and loading

::: gradiend.data.text.filter_config.TextFilterConfig
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.data.text.filter_config.SpacyTagSpec
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.data.text.prediction.creator.TextPredictionDataCreator
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.data.text.preprocess.TextPreprocessConfig
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.data.text.preprocess.preprocess_texts
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.data.core.base_loader.resolve_base_data
    options:
      show_root_heading: false
      heading_level: 2

---

## Training

::: gradiend.trainer.core.arguments.TrainingArguments
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.trainer.Trainer
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.text.prediction.trainer.TextPredictionConfig
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.text.prediction.trainer.TextPredictionTrainer
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.core.pruning.PrePruneConfig
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.core.pruning.PostPruneConfig
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.core.stats.load_training_stats
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.factory.create_model_with_gradiend
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.core.dataset.GradientTrainingDataset
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.trainer.text.common.dataset.TextGradientTrainingDataset
    options:
      show_root_heading: false
      heading_level: 2

---

## Evaluation and visualization

::: gradiend.evaluator.evaluator.Evaluator
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.evaluator.encoder.EncoderEvaluator
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.evaluator.decoder.DecoderEvaluator
    options:
      show_root_heading: false
      heading_level: 2

::: gradiend.visualizer.visualizer.Visualizer
    options:
      show_root_heading: false
      heading_level: 2

