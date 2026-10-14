"""Central configuration.

Every tunable parameter lives in `config/settings.toml` -- open it, change a value, and
restart. Secrets (the API key) live in `.env`.

    precedence:  environment variable  >  config/settings.toml  >  built-in default

This module only *reads* those two files and exposes the result as `SETTINGS`.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv(override=False)

Difficulty = Literal["easy", "medium", "hard"]
Seniority = Literal["intern", "junior", "mid", "senior", "lead"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOML_PATH = PROJECT_ROOT / "config" / "settings.toml"

try:
    _TOML: dict = tomllib.loads(_TOML_PATH.read_text(encoding="utf-8")) if _TOML_PATH.exists() else {}
except (OSError, tomllib.TOMLDecodeError) as exc:
    raise RuntimeError(f"config/settings.toml could not be read: {exc}") from exc


def _toml(section: str, key: str, default):
    return _TOML.get(section, {}).get(key, default)


def _str(env: str, section: str, key: str, default: str) -> str:
    value = os.getenv(env)
    return value if value is not None else str(_toml(section, key, default))


def _int(env: str, section: str, key: str, default: int) -> int:
    try:
        return int(os.getenv(env, _toml(section, key, default)))
    except (TypeError, ValueError):
        return default


def _float(env: str, section: str, key: str, default: float) -> float:
    try:
        return float(os.getenv(env, _toml(section, key, default)))
    except (TypeError, ValueError):
        return default


# Per-stage output-token caps -- see the [max_tokens] section of config/settings.toml for why
# these are generous.
DEFAULT_MAX_OUTPUT_TOKENS: dict[str, int] = {
    stage: _int(f"INTERVIEW_MAX_TOKENS_{stage.upper()}", "max_tokens", stage, default)
    for stage, default in (("analyzer", 4000), ("planner", 5000), ("questioner", 4000),
                           ("evaluator", 3000), ("reporter", 5000))
}


COVERAGE_DEFAULTS: dict[str, float] = {"easy": 0.5, "medium": 0.7, "hard": 0.85}


@dataclass(frozen=True)
class Settings:
    # ---- secrets (.env only, never config/settings.toml) ----
    api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()

    # ---- model ----
    base_url: str = _str("OPENROUTER_BASE_URL", "model", "base_url", "https://openrouter.ai/api/v1")
    model: str = _str("INTERVIEW_MODEL", "model", "model", "openai/gpt-5-nano")
    reasoning_effort: str = _str("INTERVIEW_REASONING_EFFORT", "model", "reasoning_effort", "low")
    temperature: float = _float("INTERVIEW_TEMPERATURE", "model", "temperature", 0.4)

    # ---- limits ----
    max_concurrency: int = _int("INTERVIEW_MAX_CONCURRENCY", "limits", "max_concurrency", 8)
    timeout_s: int = _int("INTERVIEW_TIMEOUT_S", "limits", "timeout_s", 90)
    max_retries: int = _int("INTERVIEW_MAX_RETRIES", "limits", "max_retries", 3)
    app_title: str = _str("OPENROUTER_APP_TITLE", "app", "title", "interview_AI")
    site_url: str = _str("OPENROUTER_SITE_URL", "app", "site_url", "http://localhost:8501")

    # ---- prompting strategy (text lives in prompts/, see prompts/README.md) ----
    # Global default; per-stage overrides in [strategy] let you mix, e.g. one_shot
    # analysis + guided question writing.
    strategy: str = _str("INTERVIEW_STRATEGY", "strategy", "default", "few_shot")

    # ---- guard rails (see interview_ai/utils/security.py) ----
    max_job_ad_chars: int = _int("INTERVIEW_MAX_JOB_AD_CHARS", "limits", "max_job_ad_chars", 24_000)
    max_answer_chars: int = _int("INTERVIEW_MAX_ANSWER_CHARS", "limits", "max_answer_chars", 6_000)
    max_calls_per_session: int = _int("INTERVIEW_MAX_CALLS", "limits", "max_calls_per_session", 300)
    url_fetch_timeout_s: int = _int("INTERVIEW_URL_TIMEOUT_S", "limits", "url_fetch_timeout_s", 15)
    url_max_bytes: int = _int("INTERVIEW_URL_MAX_BYTES", "limits", "url_max_bytes", 2_000_000)

    # ---- interviewer persona (tone of questions and follow-ups only) ----
    persona: str = _str("INTERVIEW_PERSONA", "interview", "persona", "neutral")
    # ---- response length (questions and final report only) ----
    verbosity: str = _str("INTERVIEW_VERBOSITY", "interview", "verbosity", "standard")

    # ---- runtime overrides, set from the UI; None means "use the file" ----
    # A frozen dataclass plus `dataclasses.replace` is how the sidebar tunes a run without
    # mutating global state: each session builds its own Settings and hands it to its LLM.
    max_tokens_overrides: dict[str, int] | None = None
    coverage_override: float | None = None

    # ---- interview shape ----
    default_questions: int = _int("INTERVIEW_QUESTIONS", "interview", "questions", 10)
    max_followups_per_question: int = _int("INTERVIEW_MAX_FOLLOWUPS", "interview", "max_followups", 1)

    # ---- independent judge for the eval harness (optional, e.g. local Ollama) ----
    # Ollama exposes an OpenAI-compatible endpoint, so only model/base_url change.
    # The SSRF guard applies to job-ad URL fetching, not to this endpoint.
    judge_model: str = _str("JUDGE_MODEL", "judge", "model", "")
    judge_base_url: str = _str("JUDGE_BASE_URL", "judge", "base_url", "http://localhost:11434/v1")
    judge_api_key: str = _str("JUDGE_API_KEY", "judge", "api_key", "ollama")
    judge_timeout_s: int = _int("JUDGE_TIMEOUT_S", "judge", "timeout_s", 300)
    judge_max_concurrency: int = _int("JUDGE_MAX_CONCURRENCY", "judge", "max_concurrency", 2)
    judge_temperature: float = _float("JUDGE_TEMPERATURE", "judge", "temperature", 0.0)
    # `guided` by default: a 4B model gains far more from an explicit procedure than the
    # primary model does, and it must not share exemplars with the model it is judging.
    judge_strategy: str = _str("JUDGE_STRATEGY", "judge", "strategy", "guided")

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and "paste-your" not in self.api_key

    @property
    def judge_configured(self) -> bool:
        return bool(self.judge_model)

    def strategy_for(self, stage: str) -> str:
        """Per-stage override (env var, then [strategy] in config/settings.toml), else global."""
        env = os.getenv(f"INTERVIEW_STRATEGY_{stage.upper()}")
        if env:
            return env
        return str(_toml("strategy", stage, self.strategy))

    def max_tokens_for(self, stage: str) -> int:
        if self.max_tokens_overrides and stage in self.max_tokens_overrides:
            return self.max_tokens_overrides[stage]
        return DEFAULT_MAX_OUTPUT_TOKENS.get(stage, 2000)

    def coverage_threshold_for(self, difficulty: str) -> float:
        """How complete an answer must be before the interview moves on.

        This is the mechanical meaning of the difficulty dial: easy accepts a partial
        answer, hard demands a nearly complete one. See [coverage] in config/settings.toml.
        """
        if self.coverage_override is not None:
            return self.coverage_override
        return _float(f"INTERVIEW_COVERAGE_{difficulty.upper()}", "coverage", difficulty,
                      COVERAGE_DEFAULTS.get(difficulty, 0.7))


SETTINGS = Settings()

# Difficulty modifier applied on top of the seniority detected in the ad.
DIFFICULTY_RUBRIC: dict[str, str] = {
    "easy": (
        "Warm-up register. Ask for definitions, day-to-day usage and one concrete example. "
        "Accept textbook-level answers. Do not chain more than one follow-up."
    ),
    "medium": (
        "Standard hiring-bar register. Expect the candidate to justify trade-offs, describe how "
        "they actually used the technology, and handle one realistic complication."
    ),
    "hard": (
        "Senior-bar register. Push into failure modes, scale, cost, security and edge cases. "
        "Ask them to defend a design under a changed constraint. Reward precision, punish hand-waving."
    ),
}

SENIORITY_RUBRIC: dict[str, str] = {
    "intern": "No production experience assumed. Probe fundamentals and learning ability.",
    "junior": "1-2 years. Probe fundamentals, guided project work, and debugging basics.",
    "mid": "3-5 years. Probe autonomous delivery, design choices and ownership of a component.",
    "senior": "5+ years. Probe architecture, trade-offs, mentoring and cross-system impact.",
    "lead": "Probe technical strategy, org-level trade-offs, risk management and influence.",
}

# The Sprint 1 allow-list. The sidebar offers only these, so a run can't drift onto an
# unapproved (or unpriced) model.
ALLOWED_MODELS: dict[str, str] = {
    "openai/gpt-5-nano": "Cheapest. The default and the model the benchmark ran on.",
    "openai/gpt-5-mini": "Recommended course default. Stronger, a few times the cost.",
    "openai/gpt-5": "Highest capability. Slowest and most expensive.",
}

STRATEGY_BLURB: dict[str, str] = {
    "zero_shot": "Rules only. Cheapest and most independent, but format compliance rests entirely on the schema.",
    "one_shot": "Rules plus one worked example. Usually buys consistency for a small token cost.",
    "few_shot": "Rules plus three worked examples. Provides the most example context and the biggest risk of copying examples instead of reading the ad.",
    "guided": "Rules plus an explicit reasoning procedure. No examples to copy, so grounding usually holds, at the cost of longer outputs.",
    "self_critique": "Rules plus a stage-specific draft-audit-revise checklist. Adds a private review pass before the model returns only the corrected JSON.",
}

ROLES = [
    "AI Engineer", "Backend Engineer", "Frontend Engineer", "Full-stack Engineer",
    "Data Engineer", "Data Scientist", "ML Engineer", "DevOps / Cloud",
    "Cybersecurity", "QA / Automation", "Other",
]
