from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from anthromesh.models import CandidateOption, DecisionLog


def _decision(
    subject: str, domain: str, source: str, source_id: str, timestamp: str, context: str,
    options: dict[str, str], chosen: str, rationale: str,
) -> DecisionLog:
    when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("Capture timestamps must include a timezone.")
    digest = hashlib.sha256(
        "\x00".join((subject, source, source_id)).encode()
    ).hexdigest()
    return DecisionLog(
        decision_id=f"capture_{digest}", subject_id=subject, domain=domain, context=context,
        options_considered=[CandidateOption(option_id=key, label=label)
                            for key, label in options.items()],
        chosen_action=chosen, rationale=rationale, created_at=when.astimezone(timezone.utc).isoformat(),
        tags=["workflow_capture"], metadata={"source": source, "source_id": source_id},
    )


def github_reviews(events: list[dict[str, Any]], subject: str, account: str) -> list[DecisionLog]:
    decisions = []
    actions = {"approve": "Approve", "request_changes": "Request changes"}
    states = {"approved": "approve", "changes_requested": "request_changes"}
    for payload in events:
        review = payload["review"]
        if payload.get("action") != "submitted" or review["user"]["login"].casefold() != account.casefold():
            continue
        chosen = states.get(review["state"].lower())
        if chosen is None:
            continue
        repository, pull = payload["repository"]["full_name"], payload["pull_request"]
        summary = payload.get("diff_summary")
        pieces = [f"{repository}#{pull['number']} {pull['title']}"]
        if summary:
            pieces.append(summary)
        decisions.append(_decision(
            subject, "code_review", "github_review", f"{repository}:{review['id']}",
            review["submitted_at"], "\n".join(pieces), actions, chosen, review.get("body") or "",
        ))
    return decisions


def calendar_responses(events: list[dict[str, Any]], subject: str) -> list[DecisionLog]:
    decisions = []
    actions = {"accept": "Accept", "decline": "Decline", "tentative": "Tentative"}
    states = {"accepted": "accept", "declined": "decline", "tentative": "tentative"}
    for event in events:
        if event.get("status") == "cancelled":
            continue
        self_record = next((row for row in event.get("attendees", []) if row.get("self") is True), {})
        chosen = states.get(self_record.get("responseStatus"))
        if chosen is None:
            continue
        timestamp = event["updated"]
        start = event.get("start", {})
        context = f"{event.get('summary') or 'Calendar event'} | start: {start.get('dateTime') or start.get('date') or 'unknown'}"
        decisions.append(_decision(
            subject, "calendar", "google_calendar_response", f"{event['id']}:{chosen}",
            timestamp, context, actions, chosen, self_record.get("comment") or "",
        ))
    return decisions
