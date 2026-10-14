"""The evaluation harness: fixtures, metrics, judges, runners.

    fixtures/   gold-labelled job ads, scripted answers, and the adversarial versions of both
    metrics/    pure scoring functions -- no network, no model, unit-tested offline
    judges/     a second, independent model that re-scores what the primary one produced
    runners/    the scripts that drive a whole sweep and write a report

Reports land in `artifacts/`. The pass bar lives in `thresholds.yaml`.
Nothing here is imported by the app: `uv sync` keeps the runtime light, and the eval-only
dependencies live in the `eval` dependency group.
"""
from __future__ import annotations
