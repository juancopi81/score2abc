from __future__ import annotations

import logging
from functools import wraps
from contextlib import ContextDecorator
from time import perf_counter
from typing import Optional


class Timer(ContextDecorator):
    """Context manager/decorator that measures elapsed time."""

    def __init__(
        self,
        label: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
    ) -> None:
        self.label = label or "timer"
        self.logger = logger
        self.level = level
        self._start: Optional[float] = None
        self._elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._start is None:
            return False
        self._elapsed = perf_counter() - self._start
        if self.logger is not None:
            self.logger.log(self.level, "%s completed in %.3fs", self.label, self._elapsed)
        return False

    @property
    def elapsed(self) -> Optional[float]:
        return self._elapsed


def timed(
    label: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
):
    """Decorator that measures function runtime using Timer."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(label or func.__name__, logger=logger, level=level):
                return func(*args, **kwargs)

        return wrapper

    return decorator
