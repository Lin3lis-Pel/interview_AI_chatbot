"""What a model costs and what a run spent.

Pricing comes from OpenRouter's public model list, which needs no auth. `Usage` is the
per-run accounting the UI and the eval reports print.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("interview_ai.llm.models")


# model id -> (prompt $/token, completion $/token). Fetched once per process from
# OpenRouter's public model list, which needs no auth. Falls back to (0, 0) so a pricing
# outage never breaks a run -- it just reports $0 with cost_estimated left False.
_RATE_CACHE: dict[str, tuple[float, float]] = {}


async def fetch_rates(model: str, base_url: str) -> tuple[float, float] | None:
    """Look up published per-token pricing. Cached per process; None for local endpoints."""
    if model in _RATE_CACHE:
        return _RATE_CACHE[model]
    if "openrouter.ai" not in base_url:
        return None
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/models")
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if entry.get("id") == model:
                    pricing = entry.get("pricing", {})
                    rates = (float(pricing.get("prompt", 0) or 0),
                             float(pricing.get("completion", 0) or 0))
                    _RATE_CACHE[model] = rates
                    return rates
    except Exception as exc:  # noqa: BLE001 - pricing is a nicety, never a hard dependency
        log.debug("Could not fetch pricing for %s: %s", model, exc)
    return None


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False   # True when priced from the rate card, not billed back
    latencies_s: list[float] = field(default_factory=list)

    def add(self, completion: object, elapsed: float,
            rates: tuple[float, float] | None = None) -> None:
        self.calls += 1
        self.latencies_s.append(round(elapsed, 3))
        usage = getattr(completion, "usage", None)
        if not usage:
            return

        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += prompt
        self.completion_tokens += completion_tokens

        # OpenRouter puts the charged cost in `usage.cost`, but only the raw SDK preserves it:
        # instructor re-validates the response into a clean CompletionUsage and drops every
        # extra field (verified -- identical request, raw client keeps `cost`, instructor
        # returns model_extra == {}). So take the reported cost when it survives, and
        # otherwise price the call ourselves from the model's published rates.
        reported = float(getattr(usage, "cost", 0.0) or 0.0)
        if reported:
            self.cost_usd += reported
        elif rates:
            prompt_rate, completion_rate = rates
            self.cost_usd += prompt * prompt_rate + completion_tokens * completion_rate
            self.cost_estimated = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def p50_latency(self) -> float:
        if not self.latencies_s:
            return 0.0
        s = sorted(self.latencies_s)
        return s[len(s) // 2]

    def summary(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_estimated": self.cost_estimated,
            "p50_latency_s": self.p50_latency,
            "max_latency_s": max(self.latencies_s, default=0.0),
        }
