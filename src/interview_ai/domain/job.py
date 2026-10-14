"""The job advertisement, as the analyzer stage extracts it.

These are the *only* shapes the LLM is allowed to return for stage 1. `instructor`
validates every response against them and re-prompts on failure, so a malformed model
reply never reaches the UI.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Criticality = Literal["critical", "important", "bonus"]
Seniority = Literal["intern", "junior", "mid", "senior", "lead"]


class Requirement(BaseModel):
    name: str = Field(description="Short canonical name of the skill/requirement, e.g. 'Docker'.")
    criticality: Criticality
    evidence: str = Field(description="Verbatim quote from the ad, <= 15 words, that justifies this.")


class JobAnalysis(BaseModel):
    is_job_ad: bool = Field(description="False if the supplied text is not a job advertisement.")
    title: str = ""
    company: str = ""
    role: str = Field(default="Other", description="One of the fixed role classes.")
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] = "mid"
    domain: str = Field(default="", description="Business domain, e.g. fintech, healthcare, adtech.")
    responsibilities: list[str] = Field(default_factory=list, max_length=12)
    technologies: list[str] = Field(default_factory=list, max_length=25)
    requirements: list[Requirement] = Field(default_factory=list, max_length=25)
    emphasis: list[str] = Field(
        default_factory=list, max_length=6,
        description="Themes the ad repeats or foregrounds (title / first paragraph / repeated >=2x).",
    )
    notes: str = Field(default="", description="Anything ambiguous or missing from the ad.")

    @property
    def required_skills(self) -> list[str]:
        return [r.name for r in self.requirements if r.criticality in ("critical", "important")]

    @property
    def preferred_skills(self) -> list[str]:
        return [r.name for r in self.requirements if r.criticality == "bonus"]
