"""interview_AI -- a job-ad-driven mock technical interviewer.

Layers, outermost first:

    ui/        Streamlit frontend
    pipeline/  the five model-calling stages, in the order the interview runs them
    domain/    the pydantic contracts and the interview state machine
    llm/       one async, schema-validated, concurrency-limited model client
    prompts/   the prompt text itself, as plain files, plus the assembler
    utils/     input safety, job-ad fetching, formatting
    config.py  every tunable, read from config/settings.toml and the environment

Nothing in `domain/` imports `llm/`, and nothing in `prompts/` imports anything at all, so
the contracts and the prompt text can be inspected and tested without a model.
"""
from __future__ import annotations

__version__ = "0.1.0"
