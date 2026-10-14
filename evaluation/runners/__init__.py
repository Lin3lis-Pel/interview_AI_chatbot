"""Entry points that drive a whole evaluation and write a report to `artifacts/`.

    benchmark.py      the full sweep: every strategy arm x every fixture, scored and tabled
    judge_answers.py  the offline re-score: replays saved artifacts past an independent judge

Both are thin CLIs over `evaluation.metrics` and `evaluation.judges`, run with `python -m`.
"""
from __future__ import annotations
