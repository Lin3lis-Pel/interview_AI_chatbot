# Experiment: verbosity

- **Run:** 20261004-104433
- **Model:** `openai/gpt-5-nano` (reasoning effort: low)
- **Prompt version:** `2026-10-04.v8-verbosity`
- **Ads:** ai_engineer_junior, devops_senior, data_engineer_mid, frontend_mid
- **Wall time:** 164s

| Verbosity   | Words / question   | Question grounding   | Rubric present   | Duplicate rate   | Valid reports   | Report words   | Report list items   | Overall score   | Unmet requirements   |
|-------------|--------------------|----------------------|------------------|------------------|-----------------|----------------|---------------------|-----------------|----------------------|
| standard    | 32.7               | 0.58                 | 1.00             | 0.00             | 4/4             | 505            | 13.5                | 56.6            | 5.5                  |
| concise     | 22.0               | 0.58                 | 1.00             | 0.00             | 4/4             | 233            | 14.5                | 59.0            | 6.2                  |
| detailed    | 55.2               | 0.71                 | 1.00             | 0.00             | 4/4             | 617            | 13.0                | 56.6            | 5.2                  |

First question for `ai_engineer_junior`, per setting:

- **standard:** Describe a minimal Python-based RAG pipeline using a vector store; outline a lightweight evaluation metric you would rely on to judge retrieval quality and how you would verify it in production without interrupting users.
- **concise:** Describe how you ensured production-grade retrieval quality in a RAG system you built end-to-end.
- **detailed:** You’re building a small RAG pipeline in Python with a vector store and Dockerized services. You’ll need to run a quick evaluation to show you can quantify retrieval quality and iterate on improvements before a release. Walk me through how you would measure retrieval effectiveness in production, what metrics you’d track, and a concrete first hardening step you’d take within a week.

Whole experiment: 112 calls, $0.0476.
