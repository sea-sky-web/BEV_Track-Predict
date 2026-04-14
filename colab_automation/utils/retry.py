"""Simple retry utility."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def retry(
    operation: Callable[[], T],
    attempts: int,
    wait_seconds: float = 1.0,
    backoff: float = 2.0,
    exceptions: Iterable[type[BaseException]] = (Exception,),
    logger: logging.Logger | None = None,
    operation_name: str = "operation",
) -> T:
    """Retry an operation with exponential backoff."""

    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    delay = wait_seconds
    exception_types = tuple(exceptions)
    last_exception: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except exception_types as exc:  # type: ignore[arg-type]
            last_exception = exc
            if attempt >= attempts:
                break
            if logger:
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retry in %.1fs.",
                    operation_name,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
            time.sleep(delay)
            delay *= backoff

    assert last_exception is not None
    raise last_exception

