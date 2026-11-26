from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, Field

from anthromesh.gate.calibration import compute_ece, jensen_shannon_divergence


class PersonalObservation(BaseModel):
    backend: str
    system: str
    participant_id: str
    case_id: str
    expected_action: str
    predicted_action: str
    confidence: float = Field(ge=0, le=1)
    escalated: bool
    out_of_distribution: bool


class SurveyObservation(BaseModel):
    backend: str
    system: str
    question_id: str
    observed: dict[str, float]
    predicted: dict[str, float]
    ordered_options: list[str] | None = None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0 or any(value < 0 for value in values.values()):
        raise ValueError("Survey distributions require finite nonnegative mass and a positive total.")
    return {key: value / total for key, value in values.items()}


def _personal(rows: list[PersonalObservation]) -> dict[str, Any]:
    correct = [row.predicted_action == row.expected_action for row in rows]
    accepted = sum(not row.escalated for row in rows)
    escalated = len(rows) - accepted
    tp = sum(row.escalated and row.out_of_distribution for row in rows)
    false_positive = sum(row.escalated and not row.out_of_distribution for row in rows)
    false_negative = sum(not row.escalated and row.out_of_distribution for row in rows)
    return {
        "observations": len(rows),
        "participants": len({row.participant_id for row in rows}),
        "selective_accuracy": _ratio(
            sum(not row.escalated and ok for row, ok in zip(rows, correct)), accepted,
        ),
        "ood_escalation_f1": _ratio(2 * tp, 2 * tp + false_positive + false_negative),
        "ece": compute_ece([row.confidence for row in rows], correct),
        "escalation_rate": escalated / len(rows),
        "avoided_errors": sum(row.escalated and not ok for row, ok in zip(rows, correct)),
    }


def _survey(rows: list[SurveyObservation]) -> dict[str, Any]:
    jsd = []
    distances = []
    entropy = []
    recalled = total_tail = 0
    for row in rows:
        if set(row.observed) != set(row.predicted) or len(row.observed) < 2:
            raise ValueError("Survey predictions and observations must declare the same options.")
        observed, predicted = _distribution(row.observed), _distribution(row.predicted)
        jsd.append(jensen_shannon_divergence(observed, predicted))
        entropy.append(-sum(p * math.log2(p) for p in predicted.values() if p > 0)
                       / math.log2(len(predicted)))
        tail = [key for key, p in observed.items() if 0 < p < 0.10]
        total_tail += len(tail)
        recalled += sum(predicted[key] >= 0.5 * observed[key] for key in tail)
        if row.ordered_options is not None:
            if set(row.ordered_options) != set(observed) or len(row.ordered_options) != len(observed):
                raise ValueError("Ordered options must list every response exactly once.")
            gap = distance = 0.0
            for option in row.ordered_options[:-1]:
                gap += observed[option] - predicted[option]
                distance += abs(gap)
            distances.append(distance)
    return {
        "questions": len(rows),
        "mean_jsd": fmean(jsd),
        "mean_ordinal_wasserstein": fmean(distances) if distances else None,
        "ordinal_questions": len(distances),
        "mean_normalized_entropy": fmean(entropy),
        "long_tail_recall_at_half_observed_mass": _ratio(recalled, total_tail),
    }


def _paired_improvement(
    baseline: list[PersonalObservation | SurveyObservation],
    candidate: list[PersonalObservation | SurveyObservation],
    mode: str,
) -> dict[str, Any]:
    def key(row):
        return ((row.participant_id, row.case_id) if mode == "personal" else (row.question_id,))

    by_key = {key(row): row for row in baseline}
    if set(by_key) != {key(row) for row in candidate}:
        raise ValueError("Baseline and candidate must cover the same evaluation observations.")
    clusters = defaultdict(list)
    for row in candidate:
        other = by_key[key(row)]
        if mode == "personal":
            if (row.expected_action, row.out_of_distribution) != (
                other.expected_action, other.out_of_distribution,
            ):
                raise ValueError("Paired personal outcomes must use the same ground truth.")
        elif row.observed != other.observed or row.ordered_options != other.ordered_options:
            raise ValueError("Paired survey outcomes must use the same ground truth and ordering.")
        clusters[key(row)[0]].append((other, row))

    def reduction(pairs) -> float:
        reference, evaluated = zip(*pairs)
        if mode == "personal":
            return _personal(reference)["ece"] - _personal(evaluated)["ece"]
        return _survey(reference)["mean_jsd"] - _survey(evaluated)["mean_jsd"]

    groups = list(clusters.values())
    result = {
        "metric": "ece" if mode == "personal" else "mean_jsd",
        "baseline_minus_candidate": reduction([pair for group in groups for pair in group]),
        "cluster_count": len(groups),
        "confidence_interval_95": None,
    }
    if len(groups) >= 2:
        rng = random.Random(42)
        draws = sorted(
            reduction([pair for group in rng.choices(groups, k=len(groups)) for pair in group])
            for _ in range(1000)
        )
        result["confidence_interval_95"] = [draws[24], draws[974]]
    return result


def score_study(path: Path, mode: str, baseline: str | None = None) -> list[dict[str, Any]]:
    """Score independently collected observations without loading a model or uploading records."""
    subjects = mode == "personal"
    model = PersonalObservation if subjects else SurveyObservation
    groups = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = model.model_validate(json.loads(line))
            group = (row.backend, row.system)
            observation = (row.participant_id, row.case_id) if subjects else (row.question_id,)
            if (group, observation) in seen:
                raise ValueError("Duplicate study prediction within a system and backend.")
            seen.add((group, observation))
            groups[group].append(row)
    if not groups:
        raise ValueError("Provide at least one study observation.")
    summarize = _personal if subjects else _survey
    result = []
    for (backend, system), rows in sorted(groups.items()):
        summary = {"backend": backend, "system": system, **summarize(rows)}
        if baseline and system != baseline:
            reference = groups.get((backend, baseline))
            if not reference:
                raise ValueError(f"Missing baseline {baseline!r} for backend {backend!r}.")
            summary["paired_improvement"] = _paired_improvement(reference, rows, mode)
        result.append(summary)
    return result
