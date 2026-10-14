"""Schema-validated calling: JSON mode, validation re-prompts, and backoff.

`instructor` does schema injection, validation and re-prompting; `tenacity` does the
backoff. Keeping both here means `client.py` is only wiring and accounting, and the retry
policy is one obvious place to change.
"""
from __future__ import annotations

import asyncio

import instructor
from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or returned nothing usable after all retries."""


RETRYABLE = (RateLimitError, APITimeoutError, APIStatusError, asyncio.TimeoutError)

# instructor re-prompts on ValidationError this many times before giving up on a call.
# Distinct from `retrying()` below, which handles transport failures.
SCHEMA_RETRIES = 2


def json_mode_client(raw: AsyncOpenAI) -> instructor.AsyncInstructor:
    """Wrap the raw SDK client.

    Mode.JSON keeps this portable across every OpenRouter model, including ones without
    native tool-calling.
    """
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)


def retrying(attempts: int):
    """The transport-failure retry policy, as a decorator."""
    return retry(
        retry=retry_if_exception_type(RETRYABLE),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=1, max=20),
        reraise=True,
    )
