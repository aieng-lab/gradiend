"""
Evaluator package: encoder/decoder evaluation and (optionally) visualization.

- Evaluator(trainer): main entry; holds EncoderEvaluator, DecoderEvaluator, optional Visualizer.
- EncoderEvaluator: encode + correlate algorithm.
- DecoderEvaluator: grid search + best training_args algorithm.
- encoder_metrics: get_model_metrics, get_correlation, read_decoder_stats_file.
"""
from gradiend.evaluator.evaluator import Evaluator
from gradiend.evaluator.encoder import EncoderEvaluator
from gradiend.evaluator.decoder import DecoderEvaluator
from gradiend.evaluator.decoder_eval_utils import read_decoder_stats_file
from gradiend.evaluator.encoder_metrics import get_model_metrics, get_correlation

__all__ = [
    "Evaluator",
    "EncoderEvaluator",
    "DecoderEvaluator",
    "read_decoder_stats_file",
    "get_model_metrics",
    "get_correlation",
]
