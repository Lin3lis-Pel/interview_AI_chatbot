"""Request controls must match the selected model family."""
from __future__ import annotations

import asyncio

from interview_ai.config import Settings
from interview_ai.llm import LLM


def test_gpt_5_uses_reasoning_effort_instead_of_temperature() -> None:
    llm = LLM(Settings(
        api_key="test-key",
        model="openai/gpt-5-nano",
        reasoning_effort="low",
        temperature=0.8,
    ))
    try:
        request = llm._extra("analyzer")
        assert "temperature" not in request
        assert request["extra_body"]["reasoning"] == {"effort": "low"}
    finally:
        asyncio.run(llm.aclose())
