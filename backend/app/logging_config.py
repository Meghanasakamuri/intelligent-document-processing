"""Application-wide logging configuration."""

import logging
import sys

from app.config import LOG_LEVEL


def setup_logging():
    """Configure the root logger and the ``app`` logger hierarchy."""
    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)-28s │ %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    if not root.handlers:
        root.addHandler(handler)

    # Quieten noisy third-party loggers
    for name in ("httpcore", "httpx", "urllib3", "PIL", "pymupdf"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("app").setLevel(LOG_LEVEL)
