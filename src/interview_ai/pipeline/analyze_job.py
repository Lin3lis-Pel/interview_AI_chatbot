"""Stage 1: read the job ad.

Every stage takes a `strategy` (zero_shot / one_shot / few_shot / guided / self_critique).
It defaults to whatever `Settings.strategy_for(stage)` says, so the app respects `.env`
while the eval harness can force one strategy across a whole sweep.
"""
from __future__ import annotations

from ..config import SETTINGS
from ..domain.job import JobAnalysis
from ..llm import LLM
from ..prompts import Strategy, build_system, render, user_template


class NotAJobAd(ValueError):
    """The supplied text does not describe an open position."""


async def analyze_job(
    llm: LLM, job_ad: str, strategy: Strategy | None = None
) -> JobAnalysis:
    analysis = await llm.structured(
        JobAnalysis,
        build_system("analyzer", strategy or SETTINGS.strategy_for("analyzer")),
        render(user_template("analyzer"), job_ad=job_ad),
        stage="analyzer",
    )
    if not analysis.is_job_ad or not analysis.requirements:
        raise NotAJobAd(
            "That text does not look like a job advertisement -- no requirements could be "
            "extracted from it. Paste the full posting, including the requirements section."
        )
    return analysis
