"""The model layer: one class, one call shape, everything else is wiring.

`LLM.structured(...)` is the only entry point the rest of the system uses -- it takes a
pydantic model and returns an instance of it, or raises `LLMUnavailable`.
"""
from __future__ import annotations

from .client import LLM
from .models import Usage, fetch_rates
from .structured_output import LLMUnavailable

__all__ = ["LLM", "LLMUnavailable", "Usage", "fetch_rates"]
