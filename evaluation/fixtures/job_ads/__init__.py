"""Clean job ads, written to be realistic.

That includes the noise real postings carry: benefits blurbs, buzzwords, and a bit of
misdirection between the title and the actual work. The adversarial ad lives in
`evaluation.fixtures.attacks`; `ALL_ADS` includes it, `CLEAN_ADS` does not.
"""
from __future__ import annotations

from .. import AdFixture
from ..attacks import INJECTED

AI_ENGINEER = AdFixture(
    id="ai_engineer_junior",
    role="AI Engineer",
    seniority="junior",
    must_have=["python", "rag", "llm", "docker", "gcp"],
    gold_critical=["python", "rag"],
    expected_topic_keywords=["rag", "retrieval", "llm", "python", "deploy", "cloud", "behaviour", "communication"],
    text="""\
Junior AI Engineer (RAG focus) - Vilnius, hybrid

Who we are
We build retrieval-augmented assistants that read insurance policy documents so that claims
handlers do not have to. Retrieval quality is the product. We obsess over retrieval quality.

What you will do
- Build and ship RAG pipelines over large, messy PDF document sets
- Write production Python services that serve LLM traffic to internal users
- Own the evaluation harness for retrieval quality and prove that changes actually improve it
- Containerise services with Docker and deploy to GCP (Cloud Run)
- Work with prompt engineering day to day, and version prompts like code

What we need from you
- 1-2 years writing Python in production. Python is non-negotiable for this role.
- Hands-on experience building RAG systems, including chunking and vector search
- You are comfortable with Docker and can debug a container that will not start
- Familiarity with GCP or another major cloud

Nice to have
- SQL for analytics on our usage data
- Experience with an LLM evaluation framework
- Lithuanian language

We offer a learning budget, hybrid work and a team that reviews every pull request.
""",
)

DEVOPS = AdFixture(
    id="devops_senior",
    role="DevOps / Cloud",
    seniority="senior",
    must_have=["kubernetes", "terraform", "aws", "ci/cd", "observability"],
    gold_critical=["kubernetes", "terraform", "aws"],
    expected_topic_keywords=["cloud", "kubernetes", "ci", "cd", "pipeline", "infrastructure",
                             "terraform", "observability", "incident", "troubleshoot", "behaviour", "communication"],
    text="""\
Senior Platform / DevOps Engineer - Remote (EU)

The role
Our platform team keeps 200+ microservices running for a payments company. You will own the
paved road that every product team ships on. Reliability is the job. Reliability is how we
are measured, and reliability is what you will be asked about in this interview process.

Responsibilities
- Own and evolve our Kubernetes platform (EKS), including autoscaling and cost control
- Write and review Terraform for all AWS infrastructure; nothing is created by hand
- Build and maintain CI/CD pipelines in GitHub Actions used by 40 engineers
- Improve observability: metrics, tracing and actionable alerting
- Lead incident response and run blameless postmortems
- Mentor engineers on the platform and set technical direction for the team

You must have
- 5+ years in infrastructure or platform engineering
- Deep Kubernetes knowledge, in production, at scale - this is a hard requirement
- Terraform expertise. All of our AWS infrastructure is managed as code.
- Strong AWS background (VPC, IAM, EKS, RDS)
- Experience owning production incidents end to end

Bonus points
- Go or Python for internal tooling
- Experience with service meshes
- FinOps and cloud cost optimisation
""",
)

DATA_ENGINEER = AdFixture(
    id="data_engineer_mid",
    role="Data Engineer",
    seniority="mid",
    must_have=["sql", "python", "airflow", "spark", "dbt"],
    gold_critical=["sql", "python"],
    expected_topic_keywords=["sql", "python", "pipeline", "orchestration", "airflow", "spark",
                             "modelling", "warehouse", "data quality", "behaviour", "communication"],
    text="""\
Data Engineer - Berlin

We are a marketplace with 12 million monthly users and a data platform that has to keep up.

Your work
- Design and maintain batch and streaming pipelines feeding our Snowflake warehouse
- Model data with dbt so analysts can self-serve
- Orchestrate everything in Airflow
- Tune Spark jobs that currently take too long and cost too much
- Guard data quality with tests and contracts; broken dashboards are treated as incidents

Requirements
- 3-5 years as a data engineer
- Expert SQL. You will be asked to write and reason about non-trivial SQL.
- Strong Python
- Production experience with Airflow
- Experience with Spark

Preferred
- dbt
- Kafka or another streaming platform
- Terraform
""",
)

FRONTEND = AdFixture(
    id="frontend_mid",
    role="Frontend Engineer",
    seniority="mid",
    must_have=["react", "typescript", "css", "accessibility", "testing"],
    gold_critical=["react", "typescript"],
    expected_topic_keywords=["react", "typescript", "component", "state", "performance",
                             "accessibility", "css", "testing", "behaviour", "communication"],
    text="""\
Frontend Engineer (React) - London, 2 days in office

We run a booking product used by 3 million travellers a month. Accessibility is not optional
here: we are legally required to meet WCAG 2.1 AA, and accessibility bugs are treated as P1.

What you will be doing
- Build and maintain features in our React + TypeScript application
- Own component design and our design-system contributions
- Keep Core Web Vitals green; performance regressions block release
- Write component and end-to-end tests (Jest, Playwright)
- Make every feature accessible, and review peers for accessibility

Requirements
- 3+ years of React in production. Deep React knowledge is required.
- TypeScript, and you can explain why a type is wrong, not just silence it
- Strong CSS and layout skills
- Practical accessibility experience (ARIA, keyboard navigation, screen readers)
- A testing habit

Nice to have
- Next.js
- Experience with a design system at scale
- GraphQL
""",
)


ALL_ADS: list[AdFixture] = [AI_ENGINEER, DEVOPS, DATA_ENGINEER, FRONTEND, INJECTED]
CLEAN_ADS: list[AdFixture] = [a for a in ALL_ADS if not a.adversarial]

NOT_A_JOB_AD = """\
Weekly team notes - 14 August
- Standup moved to 9:45
- Remember to submit expenses before Friday
- Coffee machine is fixed
- Anna is on holiday next week, ping Marcus for deploy approvals
"""
