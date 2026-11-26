from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from anthromesh.models import DecisionLog, QueryContext
from anthromesh.twin.engine import PersonalDecisionTwin

from .conformal import CalibrationProfile, normalize_scores


def _timestamp(item: DecisionLog) -> datetime:
    stamp = datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Held-out decisions require timezone-aware timestamps.")
    return stamp.astimezone(timezone.utc)


def fit_profiles(
    twin: PersonalDecisionTwin, held_out: list[DecisionLog], alpha: float,
) -> list[CalibrationProfile]:
    """Score held-out decisions using a frozen history preceding each calibration group."""
    if not held_out or not 0 < alpha < 1:
        raise ValueError("Provide held-out decisions and an alpha strictly between zero and one.")
    held_out_ids = {item.decision_id for item in held_out}
    if len(held_out_ids) != len(held_out):
        raise ValueError("Calibration decisions must have unique decision IDs.")
    groups = defaultdict(list)
    for item in held_out:
        groups[item.subject_id, item.domain].append(item)

    result = []
    for (subject, domain), decisions in sorted(groups.items()):
        cutoff = min(map(_timestamp, decisions)).isoformat()
        history = [
            item for item in twin.vault.list_decisions(subject, domain, before=cutoff)
            if item.decision_id not in held_out_ids
        ]
        scores = []
        for item in decisions:
            query = QueryContext(
                subject_id=subject, domain=domain, context=item.context,
                candidate_options=item.options_considered, risk_tier=item.risk_tier,
            )
            options = {option.option_id for option in query.candidate_options}
            if item.chosen_action not in options:
                raise ValueError("Held-out choice must appear in its candidate options.")
            evidence, _ = twin.gate.select_evidence(query, history)
            prediction = twin.adapter.predict_decision(query, evidence)
            probabilities = normalize_scores(prediction.option_scores, options)
            scores.append(1 - probabilities.get(item.chosen_action, 0.0))
        result.append(CalibrationProfile(
            subject_id=subject, domain=domain, adapter=twin.adapter.adapter_name,
            model=twin.adapter.model, alpha=alpha, scores=scores,
        ))
    return result
