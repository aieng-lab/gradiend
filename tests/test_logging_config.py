"""setup_logging keeps third-party HTTP clients quiet."""

import logging

from gradiend.util.logging import setup_logging


def test_http_clients_do_not_log_requests_at_info():
    setup_logging(logging.INFO)
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


def test_http_clients_stay_quiet_even_with_debug_logging():
    setup_logging(logging.DEBUG)
    try:
        assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    finally:
        setup_logging(logging.INFO)
