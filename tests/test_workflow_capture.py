import hmac
import json

import pytest

from anthromesh.capture.github_webhook import create_github_capture_app
from anthromesh.capture.workflows import calendar_responses, github_reviews
from anthromesh.vault.storage import BehavioralVault


def test_capture_filters_subject_and_is_idempotent(tmp_path) -> None:
    github_event = {
        "action": "submitted", "repository": {"full_name": "repo/project"},
        "pull_request": {"number": 7, "title": "Improve the test suite"},
        "review": {"id": 9, "user": {"login": "reviewer"}, "state": "approved",
                   "submitted_at": "2026-05-07T14:00:00Z", "body": "Tests cover behavior."},
    }
    calendar_event = {
        "id": "one", "updated": "2026-05-07T15:00:00Z", "summary": "Planning",
        "attendees": [{"self": True, "responseStatus": "declined"}],
    }
    assert github_reviews([github_event], "subject", "other") == []
    decisions = github_reviews([github_event], "subject", "Reviewer")
    decisions += calendar_responses([calendar_event], "subject")
    vault = BehavioralVault(str(tmp_path / "capture.db"))
    assert vault.import_captured_decisions(decisions) == 2
    assert vault.import_captured_decisions(decisions) == 0
    assert {item.chosen_action for item in vault.list_decisions()} == {"approve", "decline"}
    assert len(vault.list_audit_logs()) == 2
    assert vault.verify_audit_chain()


def test_github_capture_checks_signature_scope_and_duplicates(tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    vault = BehavioralVault(str(tmp_path / "signed.db"))
    app = create_github_capture_app(
        vault, secret="test-key", subject="owner", account="reviewer", repositories={"repo/project"},
    )
    payload = {
        "action": "submitted", "repository": {"full_name": "repo/project"},
        "pull_request": {"number": 8, "title": "Improve ingestion"},
        "review": {"id": 10, "user": {"login": "reviewer"}, "state": "approved",
                   "submitted_at": "2026-05-07T14:00:00Z"},
    }
    body = json.dumps(payload).encode()
    with TestClient(app) as client:
        headers = {"X-GitHub-Event": "pull_request_review", "X-Hub-Signature-256": "sha256=bad"}
        assert client.post("/webhooks/github", content=body, headers=headers).status_code == 401
        headers["X-Hub-Signature-256"] = "sha256=" + hmac.digest(b"test-key", body, "sha256").hex()
        assert client.post("/webhooks/github", content=body, headers=headers).json() == {"imported": 1}
        assert client.post("/webhooks/github", content=body, headers=headers).json() == {"imported": 0}
