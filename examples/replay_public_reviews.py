
"""Replay normalized public review records through AnthroMesh's production memory path."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from itertools import groupby
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, AwareDatetime, Field

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.models import CandidateOption, DecisionLog, GateAction, QueryContext
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    reviewed_at: AwareDatetime
    context: str = Field(min_length=1)
    action: str
    query_metadata: dict = Field(default_factory=dict)
    changed_condition: bool = False


OPTIONS = [
    CandidateOption(option_id="APPROVED", label="Approve"),
    CandidateOption(option_id="CHANGES_REQUESTED", label="Request changes"),
]


def replay(rows: list[Review]) -> dict:
    if len({row.event_id for row in rows}) != len(rows):
        raise ValueError("Review events must be unique.")
    allowed = {option.option_id for option in OPTIONS}
    if any(row.action not in allowed | {"COMMENTED"} for row in rows):
        raise ValueError("Unexpected review action.")
    counts = Counter()
    subjects = set()
    with TemporaryDirectory(prefix="anthromesh-review-") as folder:
        vault = BehavioralVault(str(Path(folder) / "replay.db"))
        twin = PersonalDecisionTwin(vault, PolicyEngine(), HeuristicAdapter(), memory_only=True)
        chronological = sorted(rows, key=lambda row: row.reviewed_at)
        for stamp, records in groupby(chronological, key=lambda row: row.reviewed_at):
            batch = []
            class ReplayClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return stamp.astimezone(tz)
            with patch("anthromesh.gate.fast.datetime", ReplayClock):
                for row in records:
                    if row.action == "COMMENTED":
                        counts["comments"] += 1
                        continue
                    subjects.add(row.subject_id)
                    batch.append(row)
                    query = QueryContext(
                        subject_id=row.subject_id, domain="pull_request_review",
                        context=row.context, candidate_options=OPTIONS, metadata=row.query_metadata,
                    )
                    decision = twin.query_decision(query)
                    approved = decision.action == GateAction.APPROVED
                    counts["eligible"] += 1
                    counts["approved"] += approved
                    counts["escalated"] += decision.action == GateAction.ESCALATE
                    counts["denied"] += decision.action == GateAction.DENIED
                    counts["correct_approvals"] += approved and decision.chosen_action == row.action
                    if row.changed_condition:
                        counts["changed_conditions"] += 1
                        counts["changed_condition_approvals"] += approved
                        counts["changed_condition_wrong_approvals"] += approved and decision.chosen_action != row.action
            # Outcomes with identical submission times cannot inform each other.
            for row in batch:
                if not row.query_metadata:
                    vault.add_decision(DecisionLog(
                        subject_id=row.subject_id, domain="pull_request_review",
                        context=row.context, options_considered=OPTIONS, chosen_action=row.action,
                        rationale="Observed review", created_at=row.reviewed_at.isoformat(),
                    ))
    def ratio(num, denominator):
        return counts[num] / counts[denominator] if counts[denominator] else None
    return {
        "scope": "production memory-only gate with historical query clock",
        "participants": len(subjects),
        "counts": {name: counts[name] for name in (
            "eligible", "approved", "escalated", "denied", "correct_approvals", "comments",
            "changed_conditions", "changed_condition_approvals", "changed_condition_wrong_approvals",
        )},
        "approval_coverage": ratio("approved", "eligible"),
        "accuracy_on_approvals": ratio("correct_approvals", "approved"),
        "escalation_rate": ratio("escalated", "eligible"),
        "changed_condition_approval_rate": ratio("changed_condition_approvals", "changed_conditions"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", type=Path, help="Normalized review JSON lines")
    args = parser.parse_args()
    with args.reviews.open(encoding="utf-8") as source:
        print(json.dumps(replay([Review.model_validate_json(line) for line in source if line.strip()]), indent=2))

