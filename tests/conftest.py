"""Shared fixtures. Everything here runs offline -- no API key, no network."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")

from interview_ai.domain import (  # noqa: E402
    Evaluation,
    InterviewPlan,
    JobAnalysis,
    Question,
    QuestionBatch,
    Requirement,
    Scores,
    Topic,
)

JOB_AD = """\
Junior AI Engineer - Vilnius (hybrid)

We are building retrieval-augmented assistants for the insurance industry.

What you will do
- Build and ship RAG pipelines over customer document sets
- Write production Python services that serve LLM traffic
- Own evaluation of retrieval quality; RAG quality is the metric we live by
- Containerise services with Docker and deploy them to GCP

Requirements
- 1-2 years of Python in production. Python is non-negotiable.
- Hands-on experience with RAG and vector databases
- Comfortable with Docker
- Familiarity with GCP

Nice to have
- SQL
- Experience with prompt evaluation frameworks
"""


@pytest.fixture
def analysis() -> JobAnalysis:
    return JobAnalysis(
        is_job_ad=True,
        title="Junior AI Engineer",
        company="Acme Insurance",
        role="AI Engineer",
        seniority="junior",
        domain="insurance",
        responsibilities=["Build RAG pipelines", "Ship Python services"],
        technologies=["Python", "Docker", "GCP", "vector databases"],
        emphasis=["RAG quality", "Python"],
        requirements=[
            Requirement(name="Python", criticality="critical", evidence="Python is non-negotiable"),
            Requirement(name="RAG", criticality="critical", evidence="Hands-on experience with RAG"),
            Requirement(name="Docker", criticality="important", evidence="Comfortable with Docker"),
            Requirement(name="GCP", criticality="important", evidence="Familiarity with GCP"),
            Requirement(name="SQL", criticality="bonus", evidence="Nice to have: SQL"),
        ],
    )


@pytest.fixture
def plan() -> InterviewPlan:
    return InterviewPlan(
        topics=[
            Topic(name="LLM & RAG", weight=35, rationale="RAG is the core of the ad",
                  grounded_in=["RAG"]),
            Topic(name="Python engineering", weight=30, rationale="Python is non-negotiable",
                  grounded_in=["Python"]),
            Topic(name="Deployment & Cloud", weight=20, rationale="Docker and GCP are listed",
                  grounded_in=["Docker", "GCP"]),
            Topic(name="Collaboration", weight=15, rationale="Hybrid team work",
                  grounded_in=["Python"]),
        ]
    )


@pytest.fixture
def questions() -> list[Question]:
    return [
        Question(topic="LLM & RAG", type="conceptual", text="How would you measure whether your retrieval step is the weak link in a RAG pipeline?",
                 targets=["RAG"], what_good_looks_like="Names recall@k and separates retrieval from generation errors."),
        Question(topic="Python engineering", type="practical", text="Walk me through how you would structure a Python service that serves LLM traffic under load.",
                 targets=["Python"], what_good_looks_like="Async IO, timeouts, connection pooling."),
        Question(topic="Deployment & Cloud", type="architecture", text="Your container works locally but the image is 3 GB. How do you bring that down?",
                 targets=["Docker"], what_good_looks_like="Multi-stage builds, slim base images, layer caching."),
        Question(topic="Collaboration", type="behavioral", text="Tell me about a time you disagreed with a reviewer on an approach and how it resolved.",
                 targets=["Python"], what_good_looks_like="Concrete example, owns the outcome."),
    ]


def make_evaluation(next_action: str = "advance", follow_up: str = "", verdict: str = "partial",
                    score: int = 3, coverage: float = 1.0) -> Evaluation:
    """`coverage` drives the interview loop; default 1.0 closes the question immediately."""
    return Evaluation(
        verdict=verdict,
        information_coverage=coverage,
        scores=Scores(correctness=score, depth=score, practical_experience=score,
                      communication=score, relevance=score),
        covered=["mentioned chunking"],
        missing=["no evaluation metric named"],
        mentioned_technologies=["Docker"],
        next_action=next_action,
        follow_up=follow_up,
    )


class FakeLLM:
    """Drop-in replacement for interview_ai.llm.LLM in tests.

    `queue` maps a pydantic model class to a list of objects (or callables) to return, in
    order. Anything not queued falls back to `defaults`. Every call is recorded so tests can
    assert on the prompts that were sent.
    """

    def __init__(self, queue: dict[type, list[Any]] | None = None, defaults: dict[type, Any] | None = None):
        self.queue = {k: list(v) for k, v in (queue or {}).items()}
        self.defaults = defaults or {}
        self.calls: list[dict[str, Any]] = []
        self.max_parallel = 0
        self._inflight = 0

    async def structured(self, response_model, system, user, **kwargs):
        self._inflight += 1
        self.max_parallel = max(self.max_parallel, self._inflight)
        try:
            await asyncio.sleep(0.01)  # let siblings overlap so concurrency is observable
            self.calls.append({"model": response_model, "system": system, "user": user,
                               "kwargs": kwargs})
            pending = self.queue.get(response_model)
            if pending:
                item = pending.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item(user) if callable(item) else item
            if response_model in self.defaults:
                d = self.defaults[response_model]
                return d(user) if callable(d) else d
            raise AssertionError(f"FakeLLM has nothing queued for {response_model.__name__}")
        finally:
            self._inflight -= 1

    def prompts_for(self, response_model) -> list[str]:
        return [c["system"] + "\n" + c["user"] for c in self.calls if c["model"] is response_model]


@pytest.fixture
def fake_llm(analysis, plan, questions) -> FakeLLM:
    def batch_for(user: str) -> QuestionBatch:
        topic = next((t.name for t in plan.topics if f"SUBJECT: {t.name}" in user), plan.topics[0].name)
        return QuestionBatch(questions=[q for q in questions if q.topic == topic] or [questions[0]])

    return FakeLLM(
        defaults={
            JobAnalysis: analysis,
            InterviewPlan: plan,
            QuestionBatch: batch_for,
            Evaluation: make_evaluation(),
        }
    )
