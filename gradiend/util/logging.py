"""
Logging configuration for GRADIEND.

This module sets up logging with appropriate levels and formatting.
"""

import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for GRADIEND.
    
    Args:
        level: Logging level (default: INFO)
    """
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
    return logging.getLogger(name)


# Configure default logging once, the first time this module is imported.
# If the application has already configured logging, this will be a no-op.
if not logging.getLogger().handlers:
    setup_logging(logging.INFO)