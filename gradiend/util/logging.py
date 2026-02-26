"""
Logging configuration for GRADIEND.

This module sets up logging with appropriate levels and formatting.
"""

import logging
import sys
from contextlib import contextmanager
from typing import Generator


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for GRADIEND.
    
    Args:
        level: Logging level (default: INFO)
    """
    if not isinstance(level, int):
        raise TypeError(f"level must be int (e.g. logging.INFO), got {type(level).__name__}")
    # Get root logger
    root_logger = logging.getLogger()
    
    # If logging is already configured, update the level
    if root_logger.handlers:
        root_logger.setLevel(level)
        # Also update all existing handlers
        for handler in root_logger.handlers:
            handler.setLevel(level)
    else:
        # First time setup
        logging.basicConfig(
            level=level,
            # Keep timestamp and level, drop the full logger name to avoid verbose prefixes
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )


def get_logger(name):
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    return logging.getLogger(name)


# Loggers used by Hugging Face tokenizers when emitting the "Token indices sequence
# length is longer than the specified maximum sequence length" warning (truncation is
# applied when max_length/truncation are set, but the warning still appears).
_TOKENIZER_LOGGER_NAMES = (
    "transformers.tokenization_utils",
    "transformers.tokenization_utils_base",
    "transformers.tokenization_utils_fast",
)


@contextmanager
def suppress_tokenizer_length_warning() -> Generator[None, None, None]:
    """
    Temporarily suppress the Hugging Face tokenizer warning about sequence length
    exceeding the model maximum when truncation is explicitly requested (max_length +
    truncation=True). Use only around tokenizer calls that pass truncation and
    max_length.
    """
    loggers = [logging.getLogger(n) for n in _TOKENIZER_LOGGER_NAMES]
    old_levels = [log.level for log in loggers]
    try:
        for log in loggers:
            log.setLevel(logging.WARNING)
        yield
    finally:
        for log, level in zip(loggers, old_levels):
            log.setLevel(level)


# Configure default logging once, the first time this module is imported.
# If the application has already configured logging, this will be a no-op.
if not logging.getLogger().handlers:
    setup_logging(logging.INFO)