"""Scripted candidate answers with a gold quality tier.

Used to measure the two things the interview lives or dies on:
  * discrimination -- does the scorer separate a strong answer from a vague one?
  * adaptivity     -- does the chosen next_action match the interview policy?

The adversarial answers live in `evaluation.fixtures.attacks` and are re-exported here for the
runners that sweep both.
"""
from __future__ import annotations

from .. import RAG_Q, RAG_RUBRIC, AnswerFixture
from ..attacks import ADVERSARIAL_ANSWERS

EXPECTED_ACTIONS: dict[str, set[str]] = {
    "strong": {"raise_difficulty", "advance", "probe_technology"},
    "partial": {"drill_deeper", "probe_technology"},
    "vague": {"drill_deeper", "probe_technology"},
    "wrong": {"clarify", "drill_deeper"},
    "no_answer": {"clarify", "advance", "drill_deeper"},
}

# Expected mean score band (1-5) per tier. Wide enough to tolerate model variance,
# tight enough that a scorer which rates everything a 4 fails.
EXPECTED_BANDS: dict[str, tuple[float, float]] = {
    "strong": (3.6, 5.0),
    "partial": (2.2, 4.0),
    "vague": (1.0, 3.0),
    "wrong": (1.0, 2.6),
    "no_answer": (1.0, 2.0),
}


ANSWERS: list[AnswerFixture] = [
    AnswerFixture(
        id="rag_strong", question=RAG_Q, rubric=RAG_RUBRIC, topic="LLM & RAG", tier="strong",
        answer=(
            "I split it. First I build a small labelled set, maybe 100 questions with the gold "
            "passage marked, and measure recall@5 on retrieval alone. On our policy-document "
            "assistant that came out at 0.62, so retrieval was clearly the problem. Then, for the "
            "cases where the gold passage was retrieved, I check whether the generated answer is "
            "actually grounded in it - I ran a support check with a second model call and about "
            "9% of grounded-context answers still hallucinated. Fixing chunking, we went from 512 "
            "token fixed chunks to section-aware chunks with overlap, and recall@5 went to 0.88. "
            "Only after that did I touch the prompt."
        ),
    ),
    AnswerFixture(
        id="rag_partial", question=RAG_Q, rubric=RAG_RUBRIC, topic="LLM & RAG", tier="partial",
        answer=(
            "I would look at what the retriever returns for the failing questions and read the "
            "chunks myself. If the right chunk is not there it is a retrieval problem, and I would "
            "try different chunk sizes or a better embedding model. If the chunk is there but the "
            "answer is still wrong then it is the prompt or the model. I have not set up formal "
            "metrics for this before, mostly manual checking."
        ),
    ),
    AnswerFixture(
        id="rag_vague", question=RAG_Q, rubric=RAG_RUBRIC, topic="LLM & RAG", tier="vague",
        answer=(
            "RAG combines retrieval with generation, so you need both parts to work well. I would "
            "debug it carefully, check the logs, and iterate on the prompt until the quality is "
            "good. Evaluation is very important in these systems and you should always monitor it."
        ),
    ),
    AnswerFixture(
        id="rag_wrong", question=RAG_Q, rubric=RAG_RUBRIC, topic="LLM & RAG", tier="wrong",
        answer=(
            "Retrieval cannot really be wrong because the vector database always returns the "
            "nearest neighbours, so it is mathematically correct. Any wrong answer must come from "
            "the model, and the fix is to lower the temperature to zero, which removes "
            "hallucinations entirely. Embeddings are basically a compression of the text, so if "
            "the answer is wrong you just need a bigger context window."
        ),
    ),
    AnswerFixture(
        id="rag_no_answer", question=RAG_Q, rubric=RAG_RUBRIC, topic="LLM & RAG", tier="no_answer",
        answer="I do not know, I have never worked with RAG.",
    ),
    AnswerFixture(
        id="docker_probe", topic="Deployment & Cloud", tier="partial",
        question="Your service image is 3 GB and deploys are slow. What do you do?",
        rubric="Multi-stage build, slim base image, layer ordering, removing build tooling from the final layer.",
        answer=(
            "I used Docker for all of my projects and it worked well. I would probably look at the "
            "Dockerfile and see what can be removed. We also used Kubernetes to deploy it."
        ),
    ),
]


__all__ = [
    "ADVERSARIAL_ANSWERS", "ANSWERS", "EXPECTED_ACTIONS", "EXPECTED_BANDS",
    "AnswerFixture", "RAG_Q", "RAG_RUBRIC",
]
