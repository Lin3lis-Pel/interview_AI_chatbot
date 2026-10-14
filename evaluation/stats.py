"""Statistics for the eval report, using standard libraries rather than hand-rolled maths.

Why not keep the hand-rolled versions in metrics.py
---------------------------------------------------
`metrics.py` answers "did this specific thing hold?" with fractions in [0, 1], which is what
the PASS/FAIL thresholds need. It is deliberately dependency-free and unit-tested.

This module answers the *measurement* questions that need real statistics:

  * **Inter-rater reliability** between the primary model and the independent judge.
    Raw agreement is misleading -- two scorers that both say "partial" most of the time agree
    often by chance. Cohen's kappa corrects for that, and quadratic weighting is the right
    choice here because the verdict scale is ordinal (confusing `strong` with `partial` is a
    smaller error than confusing it with `wrong`).
  * **Rank correlation** between gold answer tiers and the scores actually given. Spearman rho
    is a stricter, continuous version of the ordering fraction in
    `metrics.discrimination()` -- it uses how far apart the tiers land, not just whether they
    are in order.
  * **Classification quality** for role prediction: per-class precision/recall/F1 and a
    confusion matrix, so a systematic mistake (e.g. every ML Engineer ad read as Data
    Scientist) is visible instead of averaged away.
  * **Set retrieval quality** for requirement extraction: precision as well as recall, since
    `requirement_recall` alone cannot see a model that dumps every noun in the ad.

scikit-learn / scipy / pandas live in the `eval` dependency group, not in the app's runtime
dependencies -- Streamlit does not need them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from interview_ai.config import ROLES

VERDICT_SCALE = ["no_answer", "wrong", "vague", "partial", "strong"]
TIER_RANK = {"no_answer": 0, "wrong": 1, "vague": 2, "partial": 3, "strong": 4}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#/. ]", " ", (s or "").lower()).strip()


# --------------------------------------------------------------------------- inter-rater
@dataclass
class AgreementReport:
    n: int = 0
    cohen_kappa: float = float("nan")
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")
    mae: float = float("nan")
    mean_bias: float = float("nan")
    exact_verdict_match: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


def rater_agreement(
    primary_scores: list[float],
    judge_scores: list[float],
    primary_verdicts: list[str],
    judge_verdicts: list[str],
) -> AgreementReport:
    """Compare the primary model against the independent judge.

    `mean_bias` is signed and is the number to watch: positive means the primary model rated
    its own generations above the second opinion, which is the self-preference signature.
    """
    n = min(len(primary_scores), len(judge_scores))
    if n == 0:
        return AgreementReport()

    ps, js = primary_scores[:n], judge_scores[:n]
    report = AgreementReport(n=n)
    report.mae = round(float(mean_absolute_error(ps, js)), 3)
    report.mean_bias = round(sum(p - j for p, j in zip(ps, js)) / n, 3)

    # Spearman is undefined when either side is constant -- a scorer giving everything a 4
    # legitimately has no rank information, so report nan rather than a fake 0.
    if len(set(ps)) > 1 and len(set(js)) > 1:
        result = spearmanr(ps, js)
        report.spearman_rho = round(float(result.statistic), 3)
        report.spearman_p = round(float(result.pvalue), 4)

    pv, jv = primary_verdicts[:n], judge_verdicts[:n]
    if pv and jv:
        report.exact_verdict_match = round(sum(a == b for a, b in zip(pv, jv)) / len(pv), 3)
        if len(set(pv) | set(jv)) > 1:
            report.cohen_kappa = round(
                float(cohen_kappa_score(pv, jv, labels=VERDICT_SCALE, weights="quadratic")), 3
            )
    return report


def tier_rank_correlation(tiers: list[str], scores: list[float]) -> dict[str, float]:
    """Spearman between the gold answer tier and the score the scorer gave.

    A stricter reading of discrimination than the adjacent-pair fraction: rho near 1.0 means
    the scorer tracks answer quality, near 0 means it is not measuring the answer at all.
    """
    if len(tiers) < 3:
        return {"n": len(tiers), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    ranks = [TIER_RANK.get(t, 0) for t in tiers]
    if len(set(ranks)) < 2 or len(set(scores)) < 2:
        return {"n": len(tiers), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    result = spearmanr(ranks, scores)
    return {
        "n": len(tiers),
        "spearman_rho": round(float(result.statistic), 3),
        "spearman_p": round(float(result.pvalue), 4),
    }


# --------------------------------------------------------------------------- classification
def role_classification(gold: list[str], predicted: list[str]) -> dict:
    """Per-class precision/recall/F1 plus a confusion matrix for role prediction."""
    if not gold:
        return {}
    labels = [r for r in ROLES if r in set(gold) | set(predicted)]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=labels, zero_division=0
    )
    cm = confusion_matrix(gold, predicted, labels=labels)
    return {
        "labels": labels,
        "per_class": {
            label: {
                "precision": round(float(p), 3),
                "recall": round(float(r), 3),
                "f1": round(float(f), 3),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "macro_f1": round(float(f1.mean()), 3) if len(f1) else 0.0,
        "accuracy": round(sum(g == p for g, p in zip(gold, predicted)) / len(gold), 3),
        "confusion_matrix": cm.tolist(),
        "confusion_index": labels,
    }


def requirement_retrieval(gold_terms: list[str], extracted: list[str]) -> dict[str, float]:
    """Set precision/recall/F1 for one ad.

    Recall alone rewards a model that lists every noun in the ad; precision is what stops it.
    Matching is loose in both directions ("GCP" vs "Google Cloud Platform").
    """
    if not gold_terms:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    ex_norm = [_norm(e) for e in extracted]
    gold_norm = [_norm(g) for g in gold_terms]

    hits = sum(any(g in e or e in g for e in ex_norm) for g in gold_norm)
    recall = hits / len(gold_norm)
    matched = sum(any(g in e or e in g for g in gold_norm) for e in ex_norm)
    precision = matched / len(ex_norm) if ex_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


# --------------------------------------------------------------------------- matrix export
@dataclass
class Collected:
    """Raw observations gathered during a run, for the statistics above."""
    gold_roles: list[str] = field(default_factory=list)
    pred_roles: list[str] = field(default_factory=list)
    primary_scores: list[float] = field(default_factory=list)
    judge_scores: list[float] = field(default_factory=list)
    primary_verdicts: list[str] = field(default_factory=list)
    judge_verdicts: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    tier_scores: list[float] = field(default_factory=list)
    requirement_prf: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> dict:
        out: dict = {}
        if self.gold_roles:
            out["role_classification"] = role_classification(self.gold_roles, self.pred_roles)
        if self.tiers:
            out["tier_rank_correlation"] = tier_rank_correlation(self.tiers, self.tier_scores)
        if self.judge_scores:
            out["judge_agreement"] = rater_agreement(
                self.primary_scores, self.judge_scores,
                self.primary_verdicts, self.judge_verdicts,
            ).as_dict()
        if self.requirement_prf:
            df = pd.DataFrame(self.requirement_prf)
            out["requirement_retrieval"] = {
                k: round(float(v), 3) for k, v in df.mean().items()
            } | {"n": len(df)}
        return out


def matrix_frame(arms: list[dict]) -> pd.DataFrame:
    """metric x strategy DataFrame, serialised into the run JSON as `matrix`."""
    data = {
        arm["strategy"]: {name: stats["mean"] for name, stats in arm["metrics"].items()}
        for arm in arms
    }
    df = pd.DataFrame(data).sort_index()
    df.index.name = "metric"
    return df


def usage_frame(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        u = arm["usage"]
        rows.append({
            "strategy": arm["strategy"],
            "calls": u["calls"],
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "total_tokens": u["total_tokens"],
            "cost_usd": u["cost_usd"],
            "p50_latency_s": u["p50_latency_s"],
            "wall_time_s": arm["wall_time_s"],
            "thresholds_met": f"{arm['n_passed']}/{arm['n_thresholded']}",
        })
    return pd.DataFrame(rows).set_index("strategy")


def render_confusion(stats: dict) -> list[str]:
    """Markdown confusion matrix, rows = gold, columns = predicted."""
    labels = stats.get("confusion_index") or []
    cm = stats.get("confusion_matrix") or []
    if not labels or not cm:
        return []
    short = [l.split(" /")[0][:14] for l in labels]
    lines = ["| gold \\ predicted | " + " | ".join(short) + " |",
             "|---|" + "---:|" * len(short)]
    for label, row in zip(short, cm):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    return lines
