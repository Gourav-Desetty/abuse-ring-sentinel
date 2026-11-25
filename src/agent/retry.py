"""Retry rate-limited Groq calls and report failures when retries are exhausted."""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, TypeVar

import groq

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_AFTER_MS_RE = re.compile(r"try again in (\d+(?:\.\d+)?)ms")

MAX_RETRIES = 8
BASE_DELAY = 5.0
MAX_DELAY = 30.0


class LLMCallFailed(RuntimeError):
    """Indicate that an LLM call exhausted its retries without producing a judgment."""

    def __init__(self, message: str, *, cause: BaseException | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.cause = cause
        self.attempts = attempts


def invoke_with_retry(fn: Callable[[], T], max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY) -> T:
    """Retry Groq rate limits using the suggested delay or exponential backoff, raising LLMCallFailed on exhaustion."""
    for attempt in range(max_retries):
        try:
            return fn()
        except groq.RateLimitError as e:
            if attempt == max_retries - 1:
                raise LLMCallFailed(
                    f"Groq rate limit not cleared after {max_retries} attempts: {e}", cause=e, attempts=max_retries
                ) from e
            match = _RETRY_AFTER_MS_RE.search(str(e))
            delay = max(float(match.group(1)) / 1000, 1.0) if match else min(base_delay * (2**attempt), MAX_DELAY)
            logger.info(f"rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises
