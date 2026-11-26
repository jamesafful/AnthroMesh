import pytest

from anthromesh.capture.graph import GraphCapture
from anthromesh.vault.storage import BehavioralVault


def test_graph_calendar_resumes_and_records_rsvp_and_reschedule(tmp_path):
    httpx = pytest.importorskip("httpx")
    vault = BehavioralVault(str(tmp_path / "capture.db"))
    calls = []
    route = "https://graph.microsoft.com/v1.0/me/calendarView/delta"

    def provider(request):
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.url.host == "graph.microsoft.com"
        if request.url.path == "/v1.0/me":
            return httpx.Response(200, json={"id": "account", "mail": "owner@example.org"})
        calls.append(request.url)
        marker = request.url.params.get("$deltatoken")
        original = {"id": "scheduled", "subject": "Weekly meeting", "isOrganizer": True,
                    "start": {"dateTime": "2026-11-25T13:00:00", "timeZone": "UTC"}}
        rsvp = {"id": "invite", "subject": "Review", "responseStatus": {
            "response": "accepted", "time": "2026-11-22T13:00:00Z",
        }}
        if marker == "2":
            return httpx.Response(200, json={"value": [], "@odata.deltaLink": f"{route}?$deltatoken=2"})
        if marker == "1":
            original["lastModifiedDateTime"] = "2026-11-23T18:00:00Z"
            original["start"]["dateTime"] = "2026-11-25T14:00:00"
            return httpx.Response(200, json={
                "value": [original, rsvp], "@odata.deltaLink": f"{route}?$deltatoken=2",
            })
        return httpx.Response(200, json={
            "value": [original, rsvp], "@odata.deltaLink": f"{route}?$deltatoken=1",
        })

    config = {"start": "2026-11-20T00:00:00Z", "end": "2026-11-27T00:00:00Z",
              "track_reschedules": True}
    with GraphCapture(vault, "subject", "owner@example.org", "test-token",
                      transport=httpx.MockTransport(provider)) as capture:
        assert capture.sync("calendar", **config) == 1
        assert capture.sync("calendar", **config) == 1
        assert capture.sync("calendar", **config) == 0
    assert [item.chosen_action for item in vault.list_decisions(subject_id="subject")] == [
        "reschedule", "accept",
    ]
    assert calls[1].params["$deltatoken"] == "1"


def test_graph_mail_captures_only_linked_drafts_and_rejects_foreign_cursor(tmp_path):
    httpx = pytest.importorskip("httpx")
    vault = BehavioralVault(str(tmp_path / "mail.db"))
    route = "https://graph.microsoft.com/v1.0/me/mailFolders/sent-id/messages/delta"
    foreign = False

    def provider(request):
        assert request.url.host == "graph.microsoft.com"
        if request.url.path == "/v1.0/me":
            return httpx.Response(200, json={"id": "account"})
        if request.url.path.endswith("/sentitems"):
            return httpx.Response(200, json={"id": "sent-id"})
        if request.url.path.endswith("/delta"):
            return httpx.Response(200, json={
                "value": [{"id": "message", "internetMessageHeaders": [
                    {"name": "X-AnthroMesh-Draft-ID", "value": "draft-1"},
                ]}, {"id": "untagged"}],
                "@odata.deltaLink": "https://outside.example/mail?token=bad" if foreign
                                    else f"{route}?$deltatoken=1",
            })
        assert request.url.path == "/v1.0/me/messages/message"
        return httpx.Response(200, json={
            "id": "message", "sentDateTime": "2026-11-23T18:00:00Z", "isDraft": False,
            "body": {"contentType": "text", "content": "Prefer Friday"},
        })

    with GraphCapture(vault, "owner", "account", "test-token",
                      transport=httpx.MockTransport(provider)) as capture:
        assert capture.sync("email", drafts={"draft-1": "Prefer Monday"}) == 1
        foreign = True
        with pytest.raises(ValueError, match="untrusted"):
            capture.sync("email", drafts={"draft-1": "Prefer Monday"})
    logged = vault.list_decisions(subject_id="owner")
    assert len(logged) == 1 and logged[0].chosen_action == "edit_and_send"
    assert "+Prefer Friday" in logged[0].context
