"""Text normalisation shared by the metric modules.

Loose matching on purpose: an extraction that says "CI/CD pipelines" has found the gold
label "ci/cd", and a metric that misses that is measuring string formatting, not skill
extraction.
"""
from __future__ import annotations

import re

SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "lead"]
VERDICT_ORDER = ["no_answer", "wrong", "vague", "partial", "strong"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#/. ]", " ", s.lower()).strip()


def tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if len(t) > 2}


def mentions(needle: str, haystack: str) -> bool:
    """Loose containment: 'ci/cd' matches 'CI/CD pipelines', 'rag' matches 'RAG systems'."""
    return norm(needle) in norm(haystack)
