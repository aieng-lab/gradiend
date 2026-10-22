"""
Core training components for GRADIEND models.
"""

from .training import train
from .callbacks import (
    TrainingCallback,
    EvaluationCallback,
    CheckpointCallback,
    NormalizationCallback,
    LoggingCallback,
    get_default_callbacks,
)
from .transition_selection import TransitionSpec, pair, identity, expand_transition_selection
from .lr_search import LRSearch, LRSearchConfig, LRSearchResult, NoConvergentLearningRate, RunState, tune_learning_rate
from .signal_checks import SignalDiversityCallback, SignalNotDiverseError, assert_signal_diverse

__all__ = [
    'train',
    'TrainingCallback',
    'EvaluationCallback',
    'CheckpointCallback',
    'NormalizationCallback',
    'LoggingCallback',
    'get_default_callbacks',
    'TransitionSpec',
    'pair',
    'identity',
    'expand_transition_selection',
    'LRSearch',
    'LRSearchConfig',
    'LRSearchResult',
    'NoConvergentLearningRate',
    'RunState',
    'tune_learning_rate',
    'assert_signal_diverse',
    'SignalDiversityCallback',
    'SignalNotDiverseError',
]
