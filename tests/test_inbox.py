import re

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.models import CandidateOption, QueryContext
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.twin.inbox import create_inbox_app
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def test_inbox_displays_escaped_context_and_resolves_the_selected_ticket(tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    vault = BehavioralVault(str(tmp_path / "inbox.db"))
    twin = PersonalDecisionTwin(vault, PolicyEngine(), HeuristicAdapter())
    pending = twin.query_decision(QueryContext(
        subject_id="owner", domain="calendar", context="<script>untrusted text</script>",
        candidate_options=[CandidateOption(option_id="yes", label="Yes"),
                           CandidateOption(option_id="no", label="No")],
    )).escalation_ticket

    def resolve(ticket_id, action, rationale):
        twin.resolve_escalation(ticket_id, action, rationale)
        return {}

    app = create_inbox_app(vault, resolve, "test-password")
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/").status_code == 401
        auth = ("owner", "test-password")
        inbox = client.get("/", auth=auth)
        assert inbox.status_code == 200
        assert "&lt;script&gt;" in inbox.text
        endpoint = f"/resolve/{pending.ticket_id}"
        form = {"csrf": "wrong", "action": "yes", "rationale": "Keep timing."}
        assert client.post(endpoint, auth=auth, data=form).status_code == 403
        form["csrf"] = re.search(r'name="csrf" value="([0-9a-f]+)"', inbox.text)[1]
        assert client.post(endpoint, auth=auth, data=form, follow_redirects=False).status_code == 303
        assert "No pending escalations" in client.get("/", auth=auth).text
