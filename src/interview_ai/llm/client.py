"""Async OpenRouter client.

Small on purpose: `structured_output` owns validation and backoff, `models` owns pricing
and usage accounting, `openai` does the HTTP. This file wires them together and adds
concurrency control.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..config import SETTINGS, Settings
from ..utils.security import redact_secrets_for_log
from .models import Usage, fetch_rates
from .structured_output import SCHEMA_RETRIES, LLMUnavailable, json_mode_client, retrying

log = logging.getLogger("interview_ai.llm")

T = TypeVar("T", bound=BaseModel)


class LLM:
    """Async, concurrency-limited, schema-validated wrapper around one OpenRouter model."""

    def __init__(
        self,
        settings: Settings = SETTINGS,
        *,
        limiter: object | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: int | None = None,
        max_concurrency: int | None = None,
        temperature: float | None = None,
        label: str = "primary",
    ):
        """`model`/`base_url`/`api_key` overrides exist so a second, independent judge model
        (e.g. a local Ollama endpoint) can be instantiated from the same class."""
        key = api_key if api_key is not None else settings.api_key
        if not key or "paste-your" in key:
            raise LLMUnavailable(
                "OPENROUTER_API_KEY is not set. Paste your key into the .env file next to this project."
            )
        self.s = settings
        self.label = label
        self.model = model or settings.model
        self.base_url = base_url or settings.base_url
        self.is_openrouter = "openrouter.ai" in self.base_url
        self.temperature = temperature if temperature is not None else settings.temperature
        self.usage = Usage()
        self._limiter = limiter
        self._rates: tuple[float, float] | None = None
        self._rates_fetched = False
        self._sem = asyncio.Semaphore(max_concurrency or settings.max_concurrency)
        self._raw = AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout_s or settings.timeout_s,
            max_retries=0,  # tenacity owns retries so backoff stays observable
            default_headers={
                "HTTP-Referer": settings.site_url,
                "X-Title": settings.app_title,
            },
        )
        self.client = json_mode_client(self._raw)

    # ------------------------------------------------------------------ internals
    def _extra(self, stage: str | None = None) -> dict:
        """gpt-5 family rejects `temperature`; it takes a reasoning-effort budget instead."""
        body: dict = {}
        kwargs: dict = {}

        if "gpt-5" in self.model:
            if self.s.reasoning_effort:
                body["reasoning"] = {"effort": self.s.reasoning_effort}
        else:
            kwargs["temperature"] = self.temperature

        # OpenRouter only returns the charged cost when the request asks for accounting.
        # Without this, `usage.cost` comes back absent and every run reports $0.00.
        # Local endpoints (Ollama) ignore the field.
        if self.is_openrouter:
            body["usage"] = {"include": True}

        if body:
            kwargs["extra_body"] = body
        if stage:
            kwargs["max_tokens"] = self.s.max_tokens_for(stage)
        return kwargs

    def _budget(self) -> None:
        if self._limiter is not None:
            self._limiter.check()

    # ------------------------------------------------------------------ public API
    async def structured(
        self,
        response_model: type[T],
        system: str,
        user: str,
        *,
        max_retries: int | None = None,
        stage: str | None = None,
    ) -> T:
        """One schema-validated call. Concurrency-limited, retried, usage-tracked.

        `stage` selects the per-stage output-token cap from Settings.max_tokens_for().
        """
        self._budget()
        attempts = max_retries if max_retries is not None else self.s.max_retries
        if self._rates is None and not self._rates_fetched:
            self._rates_fetched = True
            self._rates = await fetch_rates(self.model, self.base_url)

        @retrying(attempts)
        async def _call() -> T:
            async with self._sem:
                started = time.perf_counter()
                obj, completion = await self.client.chat.completions.create_with_completion(
                    model=self.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_retries=SCHEMA_RETRIES,  # instructor: re-prompt on schema-validation failure
                    **self._extra(stage),
                )
                self.usage.add(completion, time.perf_counter() - started, self._rates)
                return obj

        try:
            return await _call()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never a raw traceback
            log.warning("LLM call failed: %s", redact_secrets_for_log(str(exc)))
            raise LLMUnavailable(
                f"{response_model.__name__} request failed after {attempts} attempts: "
                f"{redact_secrets_for_log(str(exc))[:300]}"
            ) from exc

    async def gather(self, *coros: object, return_exceptions: bool = True) -> list:
        """Run independent calls concurrently; the semaphore caps real parallelism."""
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

    async def aclose(self) -> None:
        await self._raw.close()
