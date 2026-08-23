"""Retry-with-backoff wrapper for Groq LLM calls.

Groq enforces both a per-minute (TPM) and a *daily* (TPD) token cap; TPM
recovers with backoff, but TPD exhaustion reports minutes-to-hours waits far
beyond MAX_RETRIES x MAX_DELAY. When retries are exhausted the call is a
genuine infrastructure failure, not a real LLM judgment, so it raises
`LLMCallFailed` -- a dedicated type callers catch narrowly and record as "no
answer produced" rather than silently folding into a verdict.
"""

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
    """An LLM call could not be completed -- infrastructure, not judgment.

    Raised when retries are exhausted (typically Groq's daily token quota,
    whose wait times dwarf MAX_RETRIES x MAX_DELAY). Callers catch this to
    mark a run as *failed* rather than letting an absent answer masquerade
    as a negative one.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.cause = cause
        self.attempts = attempts


def invoke_with_retry(fn: Callable[[], T], max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY) -> T:
    """Call `fn()` (a zero-arg callable wrapping one `.invoke(...)`),
    retrying on Groq rate-limit errors. Honors the "try again in Xms" hint
    in the error message when present (it reflects the actual TPM bucket
    refill); otherwise backs off exponentially, capped at MAX_DELAY.

    Raises `LLMCallFailed` if every retry is exhausted.
    """
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
