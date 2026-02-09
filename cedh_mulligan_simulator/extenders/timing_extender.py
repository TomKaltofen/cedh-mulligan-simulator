"""Timing extender for performance profiling of feature calculations."""

import logging
import time
from typing import Any, Set

from mloda.steward import Extender, ExtenderHook

logger = logging.getLogger(__name__)


class TimingExtender(Extender):
    """Tracks execution time of feature calculations."""

    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Extract feature group name from function's qualified name
        fg_name = getattr(func, "__qualname__", None)
        if fg_name is None:
            fg_name = getattr(func, "__name__", "unknown")

        logger.info(f"{fg_name}: {elapsed:.4f}s")
        return result
