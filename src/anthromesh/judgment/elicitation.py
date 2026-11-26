from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anthromesh.models import CandidateOption, DecisionLog
from anthromesh.vault.storage import BehavioralVault

from .settings import get_judgment_settings


class ElicitationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    subject_id: str
    domain: str
    kind: Literal["scenario", "tradeoff", "counterfactual", "retest", "stated_revealed", "blind_prediction_check"]
    context: str = Field(min_length=1, max_length=12000)
    candidate_options: list[CandidateOption] = Field(min_length=2, max_length=16)
    purpose: Literal["maintenance", "study"] = "maintenance"
    parent_case_id: str | None = None
    changed_condition: str | None = Field(default=None, max_length=500)
    observed_decision_id: str | None = None
    available_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    share_with_twin: bool = False

    @field_validator("available_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Availability must include a timezone.")
        return value.astimezone(timezone.utc)


class ElicitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chosen_action: str
    rationale: str = Field(default="", max_length=4000)
    switch_condition: str = Field(default="", max_length=2000)
    answered_at: datetime
    decision_id: str | None = None
    matches_observed_action: bool | None = None


class Elicitation:
    def __init__(self, vault: BehavioralVault) -> None:
        self.vault = vault

    @staticmethod
    def _source_row(conn, identifier: str, subject: str):
        return conn.execute(
            "SELECT payload_json, response_json FROM elicitation_cases WHERE case_id = ? AND subject_id = ?",
            (identifier, subject),
        ).fetchone()

    def assign(self, case: ElicitationCase) -> ElicitationCase:
        if case.kind == "blind_prediction_check":
            raise ValueError("Prediction audit cases are assigned by the monitor.")
        if len({item.option_id for item in case.candidate_options}) != len(case.candidate_options):
            raise ValueError("Scenario options must have distinct IDs.")
        if case.share_with_twin and (case.purpose == "study" or case.kind == "retest"):
            raise ValueError("Study scenarios and retests must stay out of training.")
        followup = case.kind in {"counterfactual", "retest"}
        if bool(case.parent_case_id) != followup:
            raise ValueError("Counterfactuals and retests require one prior scenario.")
        if bool(case.changed_condition) != (case.kind == "counterfactual"):
            raise ValueError("Specify exactly one changed condition on a counterfactual.")
        if bool(case.observed_decision_id) != (case.kind == "stated_revealed"):
            raise ValueError("A stated-versus-revealed scenario requires its observed decision ID.")
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, case.subject_id)
            settings = get_judgment_settings(conn, case.subject_id, case.domain)
            if not settings or not settings.enable_elicitation:
                raise ValueError("The subject has not enabled judgment elicitation for this domain.")
            if case.share_with_twin and not settings.allow_training:
                raise ValueError("Training on elicited answers requires a separate opt-in.")
            if followup:
                row = self._source_row(conn, case.parent_case_id, case.subject_id)
                if row is None or row["response_json"] is None:
                    raise ValueError("The prior scenario has to be answered first.")
                parent = ElicitationCase.model_validate_json(row["payload_json"])
                response = ElicitationResponse.model_validate_json(row["response_json"])
                if (parent.domain != case.domain or parent.purpose != case.purpose
                        or parent.kind == "retest"
                        or sorted(parent.candidate_options, key=lambda item: item.option_id)
                        != sorted(case.candidate_options, key=lambda item: item.option_id)):
                    raise ValueError("Follow-ups need the same domain, purpose, and candidate options.")
                if case.kind == "retest" and (
                    case.context != parent.context
                    or case.available_at < response.answered_at + timedelta(hours=24)
                ):
                    raise ValueError("A retest repeats the original wording at least 24 hours later.")
                if case.kind == "counterfactual" and case.context == parent.context:
                    raise ValueError("A counterfactual must change the scenario context.")
            if case.observed_decision_id and self._observed(conn, case) is None:
                raise ValueError("The observed decision does not match this subject, domain, and option set.")
            conn.execute(
                """INSERT INTO elicitation_cases(case_id, subject_id, domain, available_at, payload_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (case.case_id, case.subject_id, case.domain, case.available_at.isoformat(),
                 case.model_dump_json()),
            )
        return case

    @staticmethod
    def _observed(conn, case: ElicitationCase) -> DecisionLog | None:
        row = conn.execute(
            "SELECT payload_json FROM decisions WHERE decision_id = ? AND subject_id = ? AND domain = ?",
            (case.observed_decision_id, case.subject_id, case.domain),
        ).fetchone()
        if row is None:
            return None
        observed = DecisionLog.model_validate_json(row["payload_json"])
        if {item.option_id for item in observed.options_considered} != {
            item.option_id for item in case.candidate_options
        }:
            return None
        return observed

    def pending(self, subject: str) -> list[dict]:
        now = datetime.now(timezone.utc)
        with self.vault._connect() as conn:
            rows = conn.execute(
                """SELECT c.payload_json, s.enable_elicitation, s.audit_probability
                   FROM elicitation_cases c JOIN judgment_settings s
                   ON s.subject_id = c.subject_id AND s.domain = c.domain
                   WHERE c.subject_id = ? AND c.response_json IS NULL
                   ORDER BY c.available_at, c.case_id""", (subject,),
            ).fetchall()
        available = []
        for row in rows:
            case = ElicitationCase.model_validate_json(row["payload_json"])
            enabled = row["audit_probability"] > 0 if case.kind == "blind_prediction_check" else row["enable_elicitation"]
            if enabled:
                available.append(case)
        return [{
            "case_id": case.case_id, "kind": case.kind, "domain": case.domain,
            "context": case.context, "candidate_options": [item.model_dump() for item in case.candidate_options],
            "changed_condition": case.changed_condition,
        } for case in available if case.available_at <= now]

    def respond(
        self, subject: str, case_id: str, action: str, rationale: str = "", switch_condition: str = "",
    ) -> ElicitationResponse:
        now = datetime.now(timezone.utc)
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, subject)
            row = self._source_row(conn, case_id, subject)
            if row is None:
                raise KeyError(case_id)
            if row["response_json"] is not None:
                previous = ElicitationResponse.model_validate_json(row["response_json"])
                if (previous.chosen_action, previous.rationale, previous.switch_condition) == (
                    action, rationale, switch_condition,
                ):
                    return previous
                raise ValueError("The response is already recorded.")
            case = ElicitationCase.model_validate_json(row["payload_json"])
            settings = get_judgment_settings(conn, subject, case.domain)
            enabled = settings is not None and (
                settings.audit_probability > 0 if case.kind == "blind_prediction_check" else settings.enable_elicitation
            )
            if not enabled or case.available_at > now:
                raise ValueError("This case is unavailable for response.")
            if action not in {option.option_id for option in case.candidate_options}:
                raise ValueError("Choose one of the case's registered options.")
            if case.kind == "tradeoff" and (not rationale.strip() or not switch_condition.strip()):
                raise ValueError("Explain the tradeoff and what would change the decision.")
            decision = None
            if case.share_with_twin and settings.allow_training:
                decision = DecisionLog(
                    subject_id=subject, domain=case.domain, context=case.context,
                    options_considered=case.candidate_options, chosen_action=action, rationale=rationale,
                    metadata={"elicitation_case_id": case_id, "switch_condition": switch_condition},
                    tags=["elicited_judgment"],
                )
            observed = self._observed(conn, case) if case.observed_decision_id else None
            response = ElicitationResponse(
                chosen_action=action, rationale=rationale, switch_condition=switch_condition,
                answered_at=now, decision_id=decision.decision_id if decision else None,
                matches_observed_action=(observed.chosen_action == action) if observed else None,
            )
            if decision:
                self.vault._insert_decision(conn, decision)
            conn.execute("UPDATE elicitation_cases SET response_json = ? WHERE case_id = ?",
                         (response.model_dump_json(), case_id))
            if case.kind == "blind_prediction_check":
                updated = conn.execute(
                    """UPDATE prediction_reviews SET outcome_action = ?, outcome_at = ?, answer_source = ?
                       WHERE subject_id = ? AND case_id = ? AND outcome_action IS NULL""",
                    (action, now.isoformat(), "human_owner", subject, case_id),
                )
                if updated.rowcount != 1:
                    raise ValueError("This audit observation is no longer available.")
            self.vault._append_audit(
                conn, "elicitation_response", "human_owner", subject, case.domain, action, None,
                {"elicitation_case_id": case_id, "decision_id": response.decision_id},
            )
        return response
