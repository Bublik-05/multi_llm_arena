"""Logging configuration."""

import logging

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s  %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call more than once."""
    logging.basicConfig(level=level, format=_FORMAT, force=True)
    # httpx logs every request at INFO; keep our own logs readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
