
from examples.replay_public_reviews import Review, replay


def test_replay_respects_equal_times_expiry_and_changed_context():
    def event(event_id, date, **changes):
        return Review(**{
            "event_id": event_id, "subject_id": "owner", "reviewed_at": date + "T12:00:00Z",
            "context": "Checks passed; tests updated", "action": "APPROVED", **changes,
        })

    rows = [
        event("fresh", "2026-01-03"),
        event("first", "2026-01-01"),
        event("same-time", "2026-01-01"),
        event("changed", "2026-01-04", context="Checks failed; tests updated",
              action="CHANGES_REQUESTED", changed_condition=True),
        event("metadata", "2026-01-05", query_metadata={"ci": "failed"},
              action="CHANGES_REQUESTED", changed_condition=True),
        event("comment", "2026-01-06", action="COMMENTED"),
        event("expired", "2026-04-15"),
    ]
    result = replay(rows)
    assert result["participants"] == 1
    assert result["counts"] == {
        "eligible": 6, "approved": 1, "escalated": 5, "denied": 0,
        "correct_approvals": 1, "comments": 1, "changed_conditions": 2,
        "changed_condition_approvals": 0, "changed_condition_wrong_approvals": 0,
    }


def test_public_collector_discards_future_checks_and_deduplicates(monkeypatch):
    from examples import public_review_pilot

    review = {
        "id": 12, "user": {"id": 42, "type": "User"},
        "author_association": "MEMBER", "state": "APPROVED",
        "commit_id": "abc", "submitted_at": "2026-01-04T12:00:00Z",
    }

    def fetch(path):
        if "/reviews?" in path:
            return [review, dict(review, id=13)]
        if "/check-runs?" in path:
            return {"check_runs": [
                {"conclusion": "success", "completed_at": "2026-01-04T11:00:00Z"},
                {"conclusion": "failure", "completed_at": "2026-01-04T13:00:00Z"},
            ]}
        return {"parents": [{"sha": "previous"}], "files": [{"filename": "tests/test_example.py"}]}

    captured = []
    monkeypatch.setattr(public_review_pilot, "fetch", fetch)
    monkeypatch.setattr(public_review_pilot, "replay", captured.extend)
    result = public_review_pilot.collect("example/project", [1])
    assert result["collection"]["later_reviews_on_same_pr"] == 1
    assert len(captured) == 1
    assert '"check_conclusions": ["success"]' in captured[0].context
    assert '"tests_touched": true' in captured[0].context
