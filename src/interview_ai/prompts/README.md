# Prompts

Each AI stage has its own folder:

- `analyzer` reads the job ad;
- `planner` creates interview topics;
- `questioner` writes questions;
- `evaluator` scores answers;
- `reporter` writes the final report.

Within each stage, `system.txt` contains the main instructions, `user.txt` contains the
input template, `guide.txt` supports the guided strategy, and `critique.txt` contains the
self-critique checklist. Worked examples are grouped under `examples/`. Shared security
rules live in `_shared/`.

Prompt strategies are selected in `config/settings.toml`:

- `zero_shot` — rules only;
- `one_shot` — rules plus one example;
- `few_shot` — rules plus three examples;
- `guided` — rules plus a step-by-step guide;
- `self_critique` — rules plus a stage-specific draft-audit-revise checklist.

These are five alternative prompt treatments applied to each of the same five product
stages (`analyzer`, `planner`, `questioner`, `evaluator`, and `reporter`). The shared rules,
input, output schema, model settings, and fixtures stay fixed, so `--strategies all` is a
controlled five-arm comparison rather than five unrelated prompts.

After changing prompt text, update `VERSION.txt` and restart the app. To inspect the exact
assembled prompts, run:

```powershell
uv run python -m interview_ai.prompts
```

The resulting files under `artifacts/prompts/` are generated output and are safe to delete.
