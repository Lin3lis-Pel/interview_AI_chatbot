# Evaluation

The `evaluation/` folder is optional. The Streamlit application does not import it.

It exists to answer a simple question: **does the AI produce grounded, useful, and safe
interviews?**

## What is measured

Labelled examples provide known job roles, requirements, and answer-quality bands. Plain
Python metrics then check:

- whether the role and seniority are detected correctly;
- whether extracted requirements can be quoted from the job ad;
- whether the interview plan covers those requirements;
- whether questions are grounded and varied;
- whether strong answers score above vague or incorrect answers;
- whether prompt-injection attempts fail;
- whether every AI response matches its Pydantic schema.

Thresholds are stored in `evaluation/thresholds.yaml`.

## Five comparable prompting techniques

The five pipeline stages are fixed tasks, not prompting techniques. Each of those same tasks
(analysis, planning, question generation, answer evaluation, and reporting) can be run under
five alternative prompt treatments:

1. `zero_shot` — shared rules only;
2. `one_shot` — shared rules plus one worked example;
3. `few_shot` — shared rules plus three worked examples;
4. `guided` — shared rules plus a stage-specific step-by-step procedure;
5. `self_critique` — shared rules plus a stage-specific draft-audit-revise checklist.

The rules, inputs, schemas, model settings, and fixtures remain fixed; only the appended
technique block differs. This creates five comparable strategy arms across the same five
tasks, rather than counting five unrelated stage prompts as five techniques. The parity is
enforced by `tests/integration/test_strategies.py`.

## Run it

The benchmark calls OpenRouter and costs a small amount:

```powershell
uv run python -m evaluation.runners.benchmark --strategies all
```

It creates an `artifacts/` folder containing JSON and Markdown reports. Ordinary generated
runs are ignored by Git and can be deleted safely; the latest five-arm evidence bundle is
explicitly retained so the README's report links work in a submitted repository.

The offline unit tests validate the metric code without calling an AI model:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Latest retained conclusion

The full comparison retained on 1 September 2026 evaluated all five techniques with
`openai/gpt-5-nano` and an independent local `gemma4:e4b` judge. Few-shot won with a 0.9697
composite and was the only arm to pass all 29 quality thresholds.

| Strategy | Composite | Thresholds | Cost | p50 latency |
|---|---:|---:|---:|---:|
| Zero-shot | 0.9230 | 27/29 | $0.0356 | 6.87 s |
| One-shot | 0.9474 | 28/29 | $0.0391 | **5.86 s** |
| Few-shot | **0.9697** | **29/29** | $0.0384 | 6.33 s |
| Guided | 0.9545 | 28/29 | $0.0459 | 7.83 s |
| Self-critique | 0.9362 | 28/29 | $0.0413 | 6.79 s |

Self-critique achieved perfect schema validity, security checks, score discrimination, and a
0.986 scorer rank correlation. Its failed threshold was independent verdict agreement
(0.583 against the 0.60 gate), and it logged one strong-answer case that proposed drilling
deeper instead of advancing. The result does not show a general advantage over the simpler
prompt treatments in this single-repeat sample.

The complete run produced 210 raw records, cost $0.2002 across the five primary-model arms,
and took 3,828 seconds. The local judge had no API charge.

The retained report, machine-readable results, and raw model-output provenance are available
in `artifacts/latest.md`, `artifacts/eval-20260901-163722.json`, and
`artifacts/artifacts-20260901-163722.jsonl`. See the root `README.md` for the broader comparison,
findings, and important sample-size caveats.
