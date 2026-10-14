"""Small formatting helpers shared by the prompt-building stages."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.job import JobAnalysis


def bullets(items: list[str], empty: str = "(none stated)") -> str:
    return "\n".join(f"- {i}" for i in items) or empty


def requirement_lines(analysis: "JobAnalysis") -> str:
    """Requirements as `name :: criticality :: "evidence"`, one per line."""
    return bullets(
        [f'{r.name} :: {r.criticality} :: "{r.evidence}"' for r in analysis.requirements]
    )
