# Experiment: verbosity

- **Run:** 20261004-104039
- **Model:** `openai/gpt-5-nano` (reasoning effort: low)
- **Prompt version:** `2026-08-31.v7-self-critique`
- **Ads:** ai_engineer_junior, devops_senior, data_engineer_mid, frontend_mid
- **Wall time:** 167s

| Verbosity   | Words / question   | Question grounding   | Rubric present   | Duplicate rate   | Valid reports   | Report words   | Report list items   | Overall score   | Unmet requirements   |
|-------------|--------------------|----------------------|------------------|------------------|-----------------|----------------|---------------------|-----------------|----------------------|
| standard    | 35.3               | 0.79                 | 1.00             | 0.00             | 4/4             | 528            | 12.8                | 56.8            | 7.0                  |
| concise     | 33.5               | 0.88                 | 1.00             | 0.00             | 4/4             | 398            | 12.0                | 61.9            | 6.2                  |
| detailed    | 40.9               | 0.83                 | 1.00             | 0.00             | 4/4             | 546            | 13.2                | 55.3            | 7.8                  |

First question for `ai_engineer_junior`, per setting:

- **standard:** Walk me through building a small RAG system for a chat assistant using Python and a vector store. Explain your chunking strategy, how you'd index embeddings, and how you'd connect the retriever and generator, including deployment considerations on Cloud Run.
- **concise:** Describe a concrete RAG pipeline you implemented, focusing on how you chunked data, integrated vector search, and the trade-offs you made between latency and retrieval quality.
- **detailed:** You're building a two‑week RAG prototype for internal docs: chunk 100k documents, store embeddings in a vector store, and deploy a Python service on Cloud Run. Describe how you would implement the chunking and vector search workflow, and what trade-offs you’d justify under tight latency and cost constraints.

Whole experiment: 118 calls, $0.0503.
