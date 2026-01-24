from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure logging once for the entire process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or os.environ.get("SCORE2ABC_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    root.setLevel(resolved_level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for score2abc."""
    configure_logging()
    return logging.getLogger(name or "score2abc")
