# interview_AI - AI performance report

- **Run:** 2026-09-01T14:41:10+00:00
- **Model:** `openai/gpt-5-nano` (reasoning effort: low)
- **Prompt version:** `2026-08-31.v7-self-critique`
- **Strategies:** zero_shot, one_shot, few_shot, guided, self_critique
- **Suites:** ads, answers, security - **repeats:** 1
- **Independent judge:** `gemma4:e4b` (strategy: guided)
- **Total wall time:** 3828.2s
- **Raw outputs:** 210 records in `artifacts-20260901-163722.jsonl`

**zero_shot** - FAIL (27/29 thresholds met) - $0.0356, 183,855 tokens, p50 6.87s
**one_shot** - FAIL (28/29 thresholds met) - $0.0391, 246,200 tokens, p50 5.86s
**few_shot** - PASS (29/29 thresholds met) - $0.0384, 284,742 tokens, p50 6.33s
**guided** - FAIL (28/29 thresholds met) - $0.0459, 267,212 tokens, p50 7.83s
**self_critique** - FAIL (28/29 thresholds met) - $0.0413, 226,999 tokens, p50 6.79s

## Metric x strategy matrix

| Metric                            |   zero_shot |   one_shot |   few_shot |    guided |   self_critique |      Best |   Threshold |
|-----------------------------------|-------------|------------|------------|-----------|-----------------|-----------|-------------|
| adaptive_action_policy            |       0.833 |  **1.000** |      1.000 |     1.000 |           0.833 |  one_shot |     >= 0.70 |
| analysis_exemplar_clean           |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.90 |
| coverage_discrimination           |       0.750 |      0.750 |  **1.000** |     1.000 |           0.750 |  few_shot |     >= 0.65 |
| coverage_matches_verdict          |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.80 |
| criticality_precision             |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.60 |
| difficulty_separation             |       0.475 |      0.400 |      0.350 |     0.375 |           0.450 |         - |  diagnostic |
| emphasis_detected                 |       1.000 |      1.000 |      1.000 |     1.000 |           1.000 |         - |  diagnostic |
| evidence_grounding                |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.80 |
| follow_up_style                   |       0.833 |  **1.000** |      1.000 |   0.667 ⚠ |           0.833 |  one_shot |     >= 0.80 |
| judge_score_agreement             |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.60 |
| judge_score_bias                  |      -0.200 |     -0.200 |     -0.167 |    -0.100 |          -0.100 |         - |  diagnostic |
| judge_verdict_agreement           |     0.583 ⚠ |      0.667 |  **0.833** |     0.833 |         0.583 ⚠ |  few_shot |     >= 0.60 |
| judged_difficulty_fit             |       0.792 |      0.833 |      0.833 |     0.750 |           0.792 |         - |  diagnostic |
| judged_question_quality           |     0.708 ⚠ |    0.708 ⚠ |  **0.792** |     0.764 |           0.750 |  few_shot |     >= 0.75 |
| plan_behavioral_balanced          |       0.900 |      0.700 |      0.800 |     1.000 |           0.800 |         - |  diagnostic |
| plan_exemplar_clean               |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.85 |
| plan_topic_count_ok               |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.90 |
| plan_topic_grounding              |       0.863 |  **0.936** |      0.923 |     0.857 |           0.883 |  one_shot |     >= 0.80 |
| plan_weights_valid                |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 1.00 |
| question_duplicate_rate           |   **0.000** |      0.000 |      0.000 |     0.000 |           0.000 | zero_shot |     <= 0.15 |
| question_exemplar_clean           |   **1.000** |      1.000 |      1.000 |     0.967 |           1.000 | zero_shot |     >= 0.90 |
| question_grounding                |       0.875 |  **0.950** |      0.938 |     0.887 |           0.938 |  one_shot |     >= 0.80 |
| question_rubric_present           |       0.988 |  **1.000** |      1.000 |     1.000 |           1.000 |  one_shot |     >= 0.90 |
| question_type_diversity           |   **1.000** |      1.000 |      1.000 |     0.986 |           1.000 | zero_shot |     >= 0.55 |
| rejects_non_job_ad                |       1.000 |      1.000 |      1.000 |     1.000 |           1.000 |         - |  diagnostic |
| report_flags_unmet_requirements   |       1.000 |      1.000 |      1.000 |     0.000 |           0.000 |         - |  diagnostic |
| report_gives_study_plan           |       1.000 |      1.000 |      1.000 |     1.000 |           1.000 |         - |  diagnostic |
| report_score_matches_answers      |       1.000 |      1.000 |      1.000 |     1.000 |           1.000 |         - |  diagnostic |
| requirement_recall                |       0.900 |  **1.000** |      0.950 |     1.000 |           0.900 |  one_shot |     >= 0.70 |
| role_accuracy                     |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.75 |
| schema_validity                   |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 1.00 |
| score_band_accuracy               |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.60 |
| score_discrimination              |       0.750 |      0.750 |  **1.000** |     1.000 |           1.000 |  few_shot |     >= 0.65 |
| security_no_prompt_leak           |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 1.00 |
| security_resisted_ad_injection    |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.75 |
| security_resisted_score_injection |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.90 |
| seniority_exact                   |       1.000 |      1.000 |      1.000 |     1.000 |           1.000 |         - |  diagnostic |
| seniority_within_one              |   **1.000** |      1.000 |      1.000 |     1.000 |           1.000 | zero_shot |     >= 0.90 |
| syllabus_coverage                 |       0.684 |      0.714 |      0.686 | **0.720** |           0.679 |    guided |     >= 0.55 |

⚠ marks a value the wrong side of its threshold.

## Which prompting strategy wins

Composite = mean of the 29 thresholded metrics every arm produced, orientation-corrected. Cost is reported alongside but not folded in.

| Rank   |      Strategy |   Composite |   Thresholds met |    Cost |   Tokens |   p50 latency |
|--------|---------------|-------------|------------------|---------|----------|---------------|
| 1      |      few_shot |      0.9697 |            29/29 | $0.0384 |  284,742 |         6.33s |
| 2      |        guided |      0.9545 |            28/29 | $0.0459 |  267,212 |         7.83s |
| 3      |      one_shot |      0.9474 |            28/29 | $0.0391 |  246,200 |         5.86s |
| 4      | self_critique |      0.9362 |            28/29 | $0.0413 |  226,999 |         6.79s |
| 5      |     zero_shot |      0.9230 |            27/29 | $0.0356 |  183,855 |         6.87s |

**Best quality:** `few_shot` - Rules plus three worked examples. Provides the most example context and the biggest risk of copying examples instead of reading the ad.

**Cheapest:** `zero_shot` at $0.0356 (0.9230 composite).

| Strategy      |   Composite |   $ per run |   $ per composite point |
|---------------|-------------|-------------|-------------------------|
| few_shot      |      0.9697 |     $0.0384 |                 $0.0396 |
| guided        |      0.9545 |     $0.0459 |                 $0.0481 |
| one_shot      |      0.9474 |     $0.0391 |                 $0.0412 |
| self_critique |      0.9362 |     $0.0413 |                 $0.0441 |
| zero_shot     |      0.9230 |     $0.0356 |                 $0.0386 |

## Scorer discrimination

Mean 1-5 score given to each scripted answer tier. Values must decrease left to right; if they do not, the scorer is not separating good answers from bad ones.

| Strategy      |   strong |   partial |   vague |   wrong |   no_answer |   ordered |
|---------------|----------|-----------|---------|---------|-------------|-----------|
| zero_shot     |     4.00 |      2.80 |    1.80 |    2.00 |        1.00 |      0.75 |
| one_shot      |     4.80 |      2.40 |    1.80 |    1.80 |        1.20 |      0.75 |
| few_shot      |     5.00 |      2.50 |    1.80 |    1.60 |        1.20 |      1.00 |
| guided        |     4.80 |      2.60 |    2.00 |    1.80 |        1.20 |      1.00 |
| self_critique |     4.40 |      2.80 |    2.20 |    1.80 |        1.00 |      1.00 |

## Coverage discrimination

Mean `information_coverage` (0-1) given to each answer tier. This is the number the interview loop runs on: reaching the difficulty threshold ends the question, falling short buys a follow-up, and a question that never gets there is reported as an unmet requirement. It must fall left to right.

| Strategy      |   strong |   partial |   vague |   wrong |   no_answer |   ordered |
|---------------|----------|-----------|---------|---------|-------------|-----------|
| zero_shot     |     0.80 |      0.50 |    0.50 |    0.35 |        0.00 |      0.75 |
| one_shot      |     0.90 |      0.43 |    0.25 |    0.25 |        0.20 |      0.75 |
| few_shot      |     1.00 |      0.38 |    0.25 |    0.20 |        0.00 |      1.00 |
| guided        |     0.90 |      0.48 |    0.30 |    0.25 |        0.00 |      1.00 |
| self_critique |     0.80 |      0.50 |    0.25 |    0.30 |        0.00 |      0.75 |

## Independent judge

Second opinion from `gemma4:e4b`, which never sees the primary score. `judge_score_bias` is signed: positive means the primary model rated its own generations higher than the judge did, which is the self-preference signature.

| Strategy      |   score agreement |   verdict agreement |   mean bias |   question quality |
|---------------|-------------------|---------------------|-------------|--------------------|
| zero_shot     |             1.000 |               0.583 |      -0.200 |              0.708 |
| one_shot      |             1.000 |               0.667 |      -0.200 |              0.708 |
| few_shot      |             1.000 |               0.833 |      -0.167 |              0.792 |
| guided        |             1.000 |               0.833 |      -0.100 |              0.764 |
| self_critique |             1.000 |               0.583 |      -0.100 |              0.750 |

## Statistics

Computed with scikit-learn and scipy. These answer *how well* rather than *did it hold*, which is what the thresholded metrics above cover.

### Role classification

| Strategy      |   accuracy |   macro F1 |
|---------------|------------|------------|
| zero_shot     |      1.000 |      1.000 |
| one_shot      |      1.000 |      1.000 |
| few_shot      |      1.000 |      1.000 |
| guided        |      1.000 |      1.000 |
| self_critique |      1.000 |      1.000 |

Confusion matrix (`zero_shot`), rows = gold:

| gold \ predicted | AI Engineer | Frontend Engin | Data Engineer | DevOps |
|---|---:|---:|---:|---:|
| **AI Engineer** | 1 | 0 | 0 | 0 |
| **Frontend Engin** | 0 | 1 | 0 | 0 |
| **Data Engineer** | 0 | 0 | 1 | 0 |
| **DevOps** | 0 | 0 | 0 | 1 |

### Requirement extraction

Recall alone rewards a model that lists every noun in the ad; precision is what stops it.

| Strategy      |   precision |   recall |    F1 |   ads |
|---------------|-------------|----------|-------|-------|
| zero_shot     |       0.586 |    0.900 | 0.709 |     4 |
| one_shot      |       0.626 |    1.000 | 0.757 |     4 |
| few_shot      |       0.585 |    0.950 | 0.722 |     4 |
| guided        |       0.519 |    1.000 | 0.675 |     4 |
| self_critique |       0.577 |    0.900 | 0.703 |     4 |

### Scorer rank correlation

Spearman rho between the gold answer tier and the score given. A stricter reading than the adjacent-pair fraction: near 1.0 means the scorer tracks answer quality, near 0 means it is not measuring the answer at all. `nan` means one side was constant (e.g. everything scored the same).

| Strategy      |   rho |      p |   n |
|---------------|-------|--------|-----|
| zero_shot     | 0.941 | 0.0051 |   6 |
| one_shot      | 0.971 | 0.0013 |   6 |
| few_shot      | 0.986 | 0.0003 |   6 |
| guided        | 0.986 | 0.0003 |   6 |
| self_critique | 0.986 | 0.0003 |   6 |

### Inter-rater reliability vs the independent judge

Cohen's kappa (quadratic-weighted, because the verdict scale is ordinal) corrects for the agreement two scorers reach by chance. `mean_bias` is signed: positive means the primary model rated its own generations above the second opinion.

| Strategy      |   kappa |   rho |   MAE |   mean bias |   exact match |   n |
|---------------|---------|-------|-------|-------------|---------------|-----|
| zero_shot     |   0.632 | 0.893 | 0.467 |      -0.200 |         0.333 |   6 |
| one_shot      |   0.727 | 0.955 | 0.267 |      -0.200 |         0.500 |   6 |
| few_shot      |   0.909 | 0.941 | 0.233 |      -0.167 |         0.667 |   6 |
| guided        |   0.909 | 0.941 | 0.233 |      -0.100 |         0.667 |   6 |
| self_critique |   0.632 | 0.941 | 0.300 |      -0.100 |         0.333 |   6 |

## Failures

### zero_shot (1)
- [rag_strong] tier strong: next_action 'drill_deeper' not in ['advance', 'probe_technology', 'raise_difficulty']

### one_shot (0)
- none

### few_shot (0)
- none

### guided (0)
- none

### self_critique (1)
- [rag_strong] tier strong: next_action 'drill_deeper' not in ['advance', 'probe_technology', 'raise_difficulty']

## How to read this

- **evidence_grounding** is the anti-hallucination metric: every extracted requirement must be quotable from the ad.
- **question_grounding / plan_topic_grounding** check the interview is built from the ad, not from the model's idea of the role.
- **score_discrimination** is the single most important number: if it is not 1.0, the final report is noise.
- **\*_exemplar_clean** is the cost side of few-shot prompting: 1.0 means the output carries no content copied from the worked examples. Expect few_shot to score worst here and best on schema_validity -- that trade-off is the reason both are measured.
- **security_\*** metrics must be 1.0. Anything less is a live prompt-injection hole.
- **judge_score_bias** consistently positive across tiers means the primary model is marking its own homework generously; the gold bands in evaluation/fixtures/answers/, not the judge, remain the ground truth.
- Bump the version in `interview_ai/prompts/VERSION.txt` when tuning, then diff two reports.