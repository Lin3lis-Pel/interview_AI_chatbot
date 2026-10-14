"""Persist what the model actually said, not just what we scored it as.

The first live run cost real money and kept only aggregate metrics -- the analyses, plans,
questions and evaluations were thrown away. That makes a metric regression impossible to
diagnose ("recall dropped to 0.7" tells you nothing about *which* requirement was missed) and
makes comparing two prompt versions a matter of trusting the numbers.

So every model output is appended to a JSONL file as it arrives. Two consequences:

  * a crash halfway through keeps everything produced up to that point (the aggregate report
    is written only at the very end, so it used to be all-or-nothing);
  * two runs can be diffed record by record, which is how you see that few_shot changed the
    *wording* of questions without moving any metric.

One file per run: artifacts/artifacts-<stamp>.jsonl
Each line: {run, n, ts, strategy, stage, fixture, payload, meta}

`n` numbers the records 1, 2, 3, ... within the run, so a record can be referred to
("answer 17") and re-used later: `python -m evaluation.runners.judge_answers` feeds the saved
`evaluation` records to the local judge model without repeating a single paid API call.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from interview_ai.utils.security import redact_pii


class ArtifactLog:
    """Append-only JSONL sink. Never raises: losing a log line must not fail an eval run."""

    def __init__(self, path: Path, run_id: str, enabled: bool = True):
        self.path = path
        self.run_id = run_id
        self.enabled = enabled
        self.count = 0
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        strategy: str,
        stage: str,
        fixture: str,
        payload: Any,
        **meta: Any,
    ) -> None:
        if not self.enabled:
            return
        if isinstance(payload, BaseModel):
            body = payload.model_dump(mode="json")
        elif isinstance(payload, list) and payload and isinstance(payload[0], BaseModel):
            body = [p.model_dump(mode="json") for p in payload]
        else:
            body = payload

        record = {
            "run": self.run_id,
            "n": self.count + 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": strategy,
            "stage": stage,
            "fixture": fixture,
            "payload": body,
            "meta": meta or {},
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            # Fixture answers are synthetic, but a real transcript could be logged here later.
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(redact_pii(line) + "\n")
            self.count += 1
        except (OSError, TypeError, ValueError):
            pass  # a log failure must never take the run down


def load_artifacts(path: str | Path) -> list[dict]:
    """Read one artifacts file back. Skips malformed lines from an interrupted run."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def latest_artifacts(reports_dir: str | Path) -> Path | None:
    files = sorted(Path(reports_dir).glob("artifacts-*.jsonl"))
    return files[-1] if files else None


def questions_from(records: list[dict], strategy: str | None = None) -> list[dict]:
    """Flatten every generated question, for eyeballing or diffing two runs."""
    out = []
    for r in records:
        if r.get("stage") != "questions":
            continue
        if strategy and r.get("strategy") != strategy:
            continue
        for q in r.get("payload") or []:
            out.append({
                "strategy": r["strategy"],
                "fixture": r["fixture"],
                "difficulty": r.get("meta", {}).get("difficulty", ""),
                "topic": q.get("topic", ""),
                "type": q.get("type", ""),
                "text": q.get("text", ""),
            })
    return out
