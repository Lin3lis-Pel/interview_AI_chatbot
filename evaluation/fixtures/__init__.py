"""Gold-labelled fixtures for the eval suite, and the shapes they come in.

    job_ads/   realistic postings with the labels a human recruiter would agree on
    answers/   scripted candidate answers with a gold quality tier
    attacks/   the adversarial half of both: an injected ad, and answers that try to
               talk the scorer into a grade

Attacks live apart from the clean fixtures on purpose. A suite that mixes them reports one
average over two different questions -- "does it read a job ad correctly?" and "can it be
talked out of reading it?" -- and those have to be answered separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdFixture:
    """A job ad plus the labels it is scored against.

      role / seniority          -> classification accuracy
      must_have                 -> requirement-extraction recall
      gold_critical             -> criticality precision (these are hard gates in the ad)
      expected_topic_keywords   -> whether the derived syllabus covers the real job
    """
    id: str
    text: str
    role: str
    seniority: str
    must_have: list[str]
    gold_critical: list[str]
    expected_topic_keywords: list[str] = field(default_factory=list)
    adversarial: bool = False


@dataclass(frozen=True)
class AnswerFixture:
    """One scripted answer, with the tier a human grader would put it in."""
    id: str
    question: str
    rubric: str
    topic: str
    answer: str
    tier: str          # strong | partial | vague | wrong | no_answer
    adversarial: bool = False


# The question every answer fixture answers. One question across all tiers is what makes
# the tiers comparable: only the answer varies.
RAG_Q = "How would you tell whether retrieval or generation is the weak link in a RAG pipeline that is giving wrong answers?"
RAG_RUBRIC = (
    "Separates the two stages, measures retrieval independently (recall@k / hit rate against a "
    "labelled set), then checks whether the answer is supported by the retrieved context."
)

__all__ = ["AdFixture", "AnswerFixture", "RAG_Q", "RAG_RUBRIC"]
