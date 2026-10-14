# Experiment: max-tokens

- **Run:** 20261004-103315
- **Model:** `openai/gpt-5-nano` (reasoning effort: low)
- **Prompt version:** `2026-08-31.v7-self-critique`
- **Ads:** ai_engineer_junior, devops_senior, data_engineer_mid, frontend_mid
- **Wall time:** 73s

| Planner max_tokens   | Valid plans   | Failed on length   | Mean output tokens (successes)   | Median time per call   | Recorded cost (successes only)   |
|----------------------|---------------|--------------------|----------------------------------|------------------------|---------------|
| 1000                 | 0/8           | 8                  | -                                | 40.8s                  | $0.0000       |
| 2000                 | 6/8           | 2                  | 1961                             | 17.6s                  | $0.0051       |
| 5000                 | 8/8           | 0                  | 2040                             | 15.2s                  | $0.0070       |

Failed calls are still billed by OpenRouter, but usage is only reported back on a successful response, so they cost more than this column shows. The median time includes instructor's re-prompts: a truncated call is retried before it fails.
