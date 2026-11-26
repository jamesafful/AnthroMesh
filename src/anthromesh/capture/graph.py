from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from difflib import unified_diff
from typing import Any
from urllib.parse import quote, unquote, urlencode, urlsplit

from anthromesh.models import DecisionLog
from anthromesh.vault.storage import BehavioralVault

from .workflows import _decision


_GRAPH = "https://graph.microsoft.com"
_CALENDAR = "/v1.0/me/calendarView/delta"
_SENT_FOLDER = "/v1.0/me/mailFolders/sentitems"
_RSVP = {"accepted": "accept", "declined": "decline", "tentativelyAccepted": "tentative"}
_RSVP_OPTIONS = {"accept": "Accept", "decline": "Decline", "tentative": "Tentative"}


def _utc(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Capture timestamps require an explicit timezone.")
    return stamp.astimezone(timezone.utc)


def _validate_cursor(url: str, route: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc.lower() != "graph.microsoft.com"
            or unquote(parsed.path).casefold() != unquote(route).casefold() or not parsed.query or parsed.fragment):
        raise ValueError("Graph returned an untrusted continuation URL.")


def _calendar_observation(
    event: dict[str, Any], subject: str, starts: dict[str, str], track_reschedules: bool,
) -> tuple[list[DecisionLog], str | None]:
    event_id = event["id"]
    if "@removed" in event or event.get("isCancelled"):
        return [], None
    start_info = event.get("start") or {}
    start = start_info.get("dateTime")
    subject_text = (event.get("subject") or "Calendar event")[:500]
    context = f"{subject_text} | start: {start or 'unknown'} UTC"
    status = event.get("responseStatus") or {}
    response = _RSVP.get(status.get("response"))
    observed = []
    if response and status.get("time") and _utc(status["time"]).year > 1:
        time_stamp = _utc(status["time"]).isoformat()
        observed.append(_decision(
            subject, "calendar", "graph_calendar_rsvp", f"{event_id}:{response}:{time_stamp}",
            time_stamp, context, _RSVP_OPTIONS, response, "",
        ))
    previous = starts.get(event_id)
    changed = event.get("lastModifiedDateTime")
    if track_reschedules and event.get("isOrganizer") is True and start and previous and previous != start and changed:
        stamp = _utc(changed).isoformat()
        moved = _decision(
            subject, "calendar", "graph_calendar_reschedule", f"{event_id}:{stamp}:{start}",
            stamp, f"{subject_text} | previous: {previous} UTC | new: {start} UTC",
            {"reschedule": "Reschedule", "keep_time": "Keep time"}, "reschedule", "",
        )
        moved.metadata["provenance"] = "Organizer event changed between syncs"
        observed.append(moved)
    return observed, start or previous


def _draft_id(message: dict[str, Any]) -> str | None:
    values = [header["value"] for header in message.get("internetMessageHeaders", [])
              if header.get("name", "").casefold() == "x-anthromesh-draft-id"]
    return values[0] if len(values) == 1 and isinstance(values[0], str) else None


def _email_observation(
    message: dict[str, Any], subject: str, draft_id: str, drafts: dict[str, str],
) -> DecisionLog | None:
    if message.get("isDraft") or draft_id not in drafts:
        return None
    body = message.get("body") or {}
    if str(body.get("contentType", "")).lower() != "text":
        raise ValueError("Graph did not return plain-text content for a matching sent draft.")
    draft = drafts[draft_id].replace("\r\n", "\n").strip()
    sent = body.get("content", "").replace("\r\n", "\n").strip()
    if max(len(draft), len(sent)) > 100_000:
        raise ValueError("A matching email draft exceeds the capture size limit.")
    changed = draft != sent
    edit = "\n".join(unified_diff(draft.splitlines(), sent.splitlines(), n=1))[:4000] if changed else ""
    context = "Sent a linked draft"
    if edit:
        context += "\nEdits:\n" + edit
    return _decision(
        subject, "email", "graph_sent_draft", message["id"], message["sentDateTime"],
        context, {"send_as_drafted": "Send as drafted", "edit_and_send": "Edit and send"},
        "edit_and_send" if changed else "send_as_drafted", "",
    )


class GraphCapture:
    """Opt-in, incremental Microsoft Graph ingestion for one delegated mailbox."""

    def __init__(
        self, vault: BehavioralVault, subject: str, account: str, access_token: str,
        *, transport: Any = None,
    ) -> None:
        import httpx

        if not subject or not account or not access_token:
            raise ValueError("Configure a subject, account and delegated Microsoft Graph access token.")
        self.vault = vault
        self.subject = subject
        self.account = account
        self.client = httpx.Client(
            timeout=15, follow_redirects=False, trust_env=False, transport=transport,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json",
                     "Prefer": 'IdType="ImmutableId", outlook.timezone="UTC", outlook.body-content-type="text"'},
        )

    def __enter__(self) -> GraphCapture:
        return self

    def __exit__(self, *_):
        self.client.close()

    def _get(self, url: str) -> dict:
        for attempt in range(3):
            response = self.client.get(url)
            if response.status_code in {429, 503, 504} and attempt < 2:
                advisory = response.headers.get("retry-after", "")
                seconds = min(int(advisory), 30) if advisory.isdecimal() else 2 ** attempt
                time.sleep(seconds)
                continue
            if response.status_code == 410:
                raise ValueError("Graph expired the delta cursor. Reset this capture stream before retrying.")
            if response.status_code != 200:
                raise ValueError(f"Microsoft Graph capture request failed with HTTP {response.status_code}.")
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("Microsoft Graph returned an invalid response.")
            return value
        raise RuntimeError("Graph response retry limit exhausted.")

    def sync(
        self, stream: str, *, start: str | None = None, end: str | None = None,
        drafts: dict[str, str] | None = None, track_reschedules: bool = False,
        max_pages: int = 100, reset: bool = False,
    ) -> int:
        if stream not in {"calendar", "email"} or max_pages < 1:
            raise ValueError("Invalid capture stream or page limit.")
        if stream == "calendar":
            if not start or not end or _utc(start) >= _utc(end):
                raise ValueError("Calendar capture requires an increasing, timezone-aware time window.")
            start, end = _utc(start).isoformat(), _utc(end).isoformat()
            route = _CALENDAR
            params = {"startDateTime": start, "endDateTime": end}
        else:
            if not isinstance(drafts, dict) or not drafts or any(
                not isinstance(value, str) for value in drafts.values()
            ):
                raise ValueError("Email capture requires a draft-ID to plain-text manifest.")
            params = {"$select": "id,isDraft,internetMessageHeaders,sentDateTime"}

        user = self._get(_GRAPH + "/v1.0/me?$select=id,mail,userPrincipalName")
        identities = {str(user.get(field) or "").casefold() for field in ("id", "mail", "userPrincipalName")}
        if self.account.casefold() not in identities or not user.get("id"):
            raise ValueError("Delegated Graph identity does not match the configured account.")
        folder_id = None
        if stream == "email":
            folder = self._get(_GRAPH + _SENT_FOLDER + "?$select=id")
            folder_id = quote(folder["id"], safe="")
            route = f"/v1.0/me/mailFolders/{folder_id}/messages/delta"
        scope = hashlib.sha256(
            repr((user["id"], stream, start, end, track_reschedules, folder_id)).encode()
        ).hexdigest()
        stream_id = f"graph:{stream}:{scope}"
        if reset:
            self.vault.reset_capture(self.subject, stream_id)
        checkpoint = self.vault.capture_checkpoint(self.subject, stream_id)
        starts = self.vault.capture_event_starts(self.subject, stream_id) if stream == "calendar" else {}
        url = checkpoint or f"{_GRAPH}{route}?{urlencode(params)}"
        total = 0
        for _ in range(max_pages):
            _validate_cursor(url, route)
            page = self._get(url)
            next_link = page.get("@odata.nextLink")
            cursor = next_link or page.get("@odata.deltaLink")
            if not isinstance(cursor, str):
                raise ValueError("Graph omitted both continuation and delta links.")
            _validate_cursor(cursor, route)
            decisions = []
            events = {}
            for item in page["value"]:
                if stream == "calendar":
                    captured, event_start = _calendar_observation(
                        item, self.subject, starts, track_reschedules,
                    )
                    events[item["id"]] = event_start
                    if event_start is None:
                        starts.pop(item["id"], None)
                    else:
                        starts[item["id"]] = event_start
                    decisions.extend(captured)
                elif "@removed" not in item and not item.get("isDraft"):
                    identifier = _draft_id(item)
                    if identifier in drafts:
                        message_id = quote(item["id"], safe="")
                        detailed = self._get(
                            f"{_GRAPH}/v1.0/me/messages/{message_id}?$select=id,isDraft,body,sentDateTime"
                        )
                        captured = _email_observation(detailed, self.subject, identifier, drafts)
                        if captured:
                            decisions.append(captured)
            total += self.vault.commit_capture_page(
                self.subject, stream_id, expected_cursor=checkpoint, next_cursor=cursor,
                decisions=decisions, events=events,
            )
            if not next_link:
                return total
            url = checkpoint = cursor
        return total
