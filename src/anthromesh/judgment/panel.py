from __future__ import annotations

import json
import secrets
import time
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from anthromesh.models import CandidateOption
from anthromesh.protocols.delegation import _decode, _https, _public_key
from anthromesh.vault.storage import BehavioralVault


class PersonAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    iss: str
    aud: str
    subject_id: str
    person_reference: str = Field(min_length=8, max_length=256)
    assurance: Literal["verified_human"]
    iat: int
    exp: int


class PanelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    subject_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    domains: list[str] = Field(min_length=1, max_length=30)
    expertise: list[str] = Field(default_factory=list, max_length=30)
    agreed_to_answer: bool
    minimum_pay_minor: int = Field(ge=1, le=100000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class PanelQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(default_factory=lambda: secrets.token_hex(16),
                            pattern=r"^[A-Za-z0-9_-]{1,80}$")
    topic: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    stimulus: str = Field(min_length=1, max_length=12000)
    options: list[CandidateOption] = Field(min_length=2, max_length=20)
    expertise: list[str] = Field(default_factory=list, max_length=20)
    minimum_reviews: int = Field(default=0, ge=0, le=1000)
    minimum_approval_rate: float = Field(default=0, ge=0, le=1)
    answer_count: int = Field(default=3, ge=1, le=100)
    pay_per_approved_answer_minor: int = Field(ge=1, le=100000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    budget_minor: int = Field(ge=1, le=10000000)
    ttl_seconds: int = Field(default=86400, ge=60, le=2592000)
    repeat_of: str | None = None


class PanelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str
    rationale: str = Field(min_length=15, max_length=5000)
    attests_independent_answer: Literal[True]


class HumanPanel:
    """Routes research questions to enrolled people and records approved compensation."""

    def __init__(self, vault: BehavioralVault) -> None:
        self.vault = vault

    def enroll(
        self, profile: PanelProfile, attestation: str, *, issuer: str, audience: str,
        public_key: str, now: int | None = None,
    ) -> None:
        now = int(time.time()) if now is None else now
        credential = _decode(
            attestation, _public_key(public_key), "anthromesh-person-attestation+jwt", PersonAttestation,
        )
        if (credential.iss != _https(issuer) or credential.aud != _https(audience)
                or credential.subject_id != profile.subject_id or not credential.iat <= now < credential.exp
                or credential.exp - credential.iat > 90 * 86400 or not profile.agreed_to_answer):
            raise ValueError("Panel enrollment requires current consent and a trusted identity attestation.")
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, profile.subject_id)
            fingerprint = self.vault._subject_digest(
                conn, "panel_identity:" + json.dumps([issuer, credential.person_reference]), create=True,
            )
            other = conn.execute(
                "SELECT subject_id FROM panel_members WHERE person_digest = ?", (fingerprint,),
            ).fetchone()
            enrolled = conn.execute(
                "SELECT person_digest FROM panel_members WHERE subject_id = ?", (profile.subject_id,),
            ).fetchone()
            if other and other["subject_id"] != profile.subject_id:
                raise ValueError("This identity is already associated with another participant.")
            if enrolled and enrolled["person_digest"] != fingerprint:
                raise ValueError("A panel account cannot replace its verified identity.")
            conn.execute(
                """INSERT INTO panel_members(subject_id, person_digest, verified_by,
                   verification_expires_at, profile_json, active) VALUES (?, ?, ?, ?, ?, 1)
                   ON CONFLICT(subject_id) DO UPDATE SET verified_by = excluded.verified_by,
                   verification_expires_at = excluded.verification_expires_at,
                   profile_json = excluded.profile_json, active = 1""",
                (profile.subject_id, fingerprint, issuer, credential.exp, profile.model_dump_json()),
            )
            self.vault._append_audit(
                conn, "panel_enrolled", "panel_operator", profile.subject_id, "panel", "enrolled", None,
                {"verified_by": issuer},
            )

    def withdraw(self, subject: str) -> None:
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute("UPDATE panel_members SET active = 0 WHERE subject_id = ? AND active = 1",
                                   (subject,))
            if changed.rowcount != 1:
                return
            conn.execute("UPDATE panel_assignments SET status = 'withdrawn' WHERE subject_id = ? AND status = 'assigned'",
                         (subject,))
            self.vault._append_audit(conn, "panel_withdrawn", "panel_operator", subject, "panel",
                                    "withdrawn", None, {})

    def submit_question(self, question: PanelQuestion, *, now: int | None = None) -> str:
        now = int(time.time()) if now is None else now
        if (len({option.option_id for option in question.options}) != len(question.options)
                or question.answer_count * question.pay_per_approved_answer_minor > question.budget_minor):
            raise ValueError("Provide distinct options and a budget covering the requested answers.")
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if question.repeat_of:
                parent = conn.execute(
                    "SELECT payload_json FROM panel_questions WHERE question_id = ?", (question.repeat_of,),
                ).fetchone()
                if parent is None:
                    raise ValueError("The reference panel question does not exist.")
                original = PanelQuestion.model_validate_json(parent["payload_json"])
                first_options = sorted(original.options, key=lambda item: item.option_id)
                new_options = sorted(question.options, key=lambda item: item.option_id)
                if (original.repeat_of or original.topic != question.topic
                        or original.stimulus != question.stimulus or first_options != new_options):
                    raise ValueError("A quality repeat must use the original topic, wording and options.")
            conn.execute("INSERT INTO panel_questions VALUES (?, ?, ?, ?, ?)",
                         (question.question_id, question.topic, question.model_dump_json(),
                          now + question.ttl_seconds, now))
        return question.question_id

    def claim(self, subject: str, topic: str, *, now: int | None = None) -> dict | None:
        now = int(time.time()) if now is None else now
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, subject)
            member = conn.execute(
                "SELECT profile_json, verification_expires_at FROM panel_members WHERE subject_id = ? AND active = 1",
                (subject,),
            ).fetchone()
            if member is None or member["verification_expires_at"] <= now:
                raise ValueError("Only currently verified and opted-in panel members can claim questions.")
            profile = PanelProfile.model_validate_json(member["profile_json"])
            if topic not in profile.domains:
                raise ValueError("The verified panel member is not routed to this topic.")
            conn.execute("UPDATE panel_assignments SET status = 'expired' WHERE status = 'assigned' AND expires_at <= ?",
                         (now,))
            mine = conn.execute(
                """SELECT a.assignment_id, a.expires_at, q.payload_json FROM panel_assignments a
                   JOIN panel_questions q ON q.question_id = a.question_id
                   WHERE a.subject_id = ? AND a.status = 'assigned' ORDER BY a.assigned_at LIMIT 1""", (subject,),
            ).fetchone()
            if mine:
                question = PanelQuestion.model_validate_json(mine["payload_json"])
                return self._assignment(question, mine["assignment_id"], mine["expires_at"]) if topic == question.topic else None
            history = conn.execute(
                """SELECT status, count(*) AS count FROM panel_assignments
                   WHERE subject_id = ? AND status IN ('approved', 'rejected') GROUP BY status""",
                (subject,),
            ).fetchall()
            reviewed = sum(row["count"] for row in history)
            passed = sum(row["count"] for row in history if row["status"] == "approved")
            rating = (passed + 1) / (reviewed + 2)
            questions = conn.execute(
                """SELECT q.payload_json, q.expires_at FROM panel_questions q WHERE q.topic = ?
                   AND q.expires_at > ? AND NOT EXISTS (
                       SELECT 1 FROM panel_assignments a WHERE a.question_id = q.question_id AND a.subject_id = ?
                   ) ORDER BY q.created_at, q.question_id""", (topic, now, subject),
            ).fetchall()
            for row in questions:
                question = PanelQuestion.model_validate_json(row["payload_json"])
                if (question.currency != profile.currency
                        or question.pay_per_approved_answer_minor < profile.minimum_pay_minor
                        or not set(question.expertise).issubset(profile.expertise)
                        or reviewed < question.minimum_reviews or rating < question.minimum_approval_rate):
                    continue
                if question.repeat_of:
                    first = conn.execute(
                        """SELECT 1 FROM panel_assignments WHERE question_id = ? AND subject_id = ?
                           AND status = 'approved' AND answered_at <= ?""",
                        (question.repeat_of, subject, now - 86400),
                    ).fetchone()
                    if first is None:
                        continue
                booked = conn.execute(
                    """SELECT count(*) AS count FROM panel_assignments WHERE question_id = ?
                       AND status IN ('assigned', 'submitted')""", (question.question_id,),
                ).fetchone()["count"]
                earned = conn.execute(
                    "SELECT count(*) AS count FROM panel_compensation WHERE question_id = ?",
                    (question.question_id,),
                ).fetchone()["count"]
                if booked + earned >= question.answer_count:
                    continue
                assignment = secrets.token_hex(16)
                until = min(now + 3600, row["expires_at"], member["verification_expires_at"])
                conn.execute(
                    """INSERT INTO panel_assignments
                       (assignment_id, question_id, subject_id, status, assigned_at, expires_at)
                       VALUES (?, ?, ?, 'assigned', ?, ?)""",
                    (assignment, question.question_id, subject, now, until),
                )
                return self._assignment(question, assignment, until)
            return None

    def current(self, subject: str, topic: str, *, now: int | None = None) -> dict | None:
        now = int(time.time()) if now is None else now
        with self.vault._connect() as conn:
            row = conn.execute(
                """SELECT a.assignment_id, a.expires_at, q.payload_json
                   FROM panel_assignments a JOIN panel_questions q ON q.question_id = a.question_id
                   JOIN panel_members m ON m.subject_id = a.subject_id
                   WHERE a.subject_id = ? AND q.topic = ? AND a.status = 'assigned'
                     AND a.expires_at > ? AND m.active = 1 AND m.verification_expires_at > ?""",
                (subject, topic, now, now),
            ).fetchone()
        if row is None:
            return None
        return self._assignment(PanelQuestion.model_validate_json(row["payload_json"]),
                                row["assignment_id"], row["expires_at"])

    @staticmethod
    def _assignment(question: PanelQuestion, assignment: str, until: int) -> dict:
        return {
            "assignment_id": assignment, "question_id": question.question_id, "topic": question.topic,
            "stimulus": question.stimulus, "options": [option.model_dump() for option in question.options],
            "expires_at": until, "payment_minor_if_approved": question.pay_per_approved_answer_minor,
            "currency": question.currency,
        }

    def answer(self, subject: str, assignment_id: str, answer: PanelAnswer, *, now: int | None = None) -> None:
        now = int(time.time()) if now is None else now
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, subject)
            row = conn.execute(
                """SELECT a.status, a.expires_at, a.answer_json, q.payload_json
                   FROM panel_assignments a JOIN panel_questions q ON q.question_id = a.question_id
                   JOIN panel_members m ON m.subject_id = a.subject_id
                   WHERE a.assignment_id = ? AND a.subject_id = ? AND m.active = 1
                     AND m.verification_expires_at > ?""",
                (assignment_id, subject, now),
            ).fetchone()
            if row is None:
                raise ValueError("No current assignment exists for this participant.")
            if row["answer_json"] == answer.model_dump_json():
                return
            if row["status"] != "assigned" or row["expires_at"] <= now:
                raise ValueError("The assignment expired or already has a response.")
            question = PanelQuestion.model_validate_json(row["payload_json"])
            if answer.option_id not in {option.option_id for option in question.options}:
                raise ValueError("The answer is outside the question's options.")
            conn.execute(
                "UPDATE panel_assignments SET status = 'submitted', answer_json = ?, answered_at = ? WHERE assignment_id = ?",
                (answer.model_dump_json(), now, assignment_id),
            )

    def review(self, assignment_id: str, reviewer: str, approved: bool, reason: str,
               *, now: int | None = None) -> None:
        now = int(time.time()) if now is None else now
        if not reviewer.strip() or not 8 <= len(reason.strip()) <= 2000:
            raise ValueError("An independent reviewer and a concise reason are required.")
        new_status = "approved" if approved else "rejected"
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT a.status, a.subject_id, a.reviewer, a.review_reason, q.payload_json
                   FROM panel_assignments a JOIN panel_questions q ON q.question_id = a.question_id
                   WHERE a.assignment_id = ?""", (assignment_id,),
            ).fetchone()
            if row is None or row["subject_id"] == reviewer:
                raise ValueError("The review must be conducted by someone other than the participant.")
            if (row["status"], row["reviewer"], row["review_reason"]) == (new_status, reviewer, reason):
                return
            if row["status"] != "submitted":
                raise ValueError("Only unreviewed submitted responses can be reviewed.")
            question = PanelQuestion.model_validate_json(row["payload_json"])
            conn.execute("UPDATE panel_assignments SET status = ?, reviewer = ?, review_reason = ? WHERE assignment_id = ?",
                         (new_status, reviewer, reason, assignment_id))
            if approved:
                conn.execute(
                    """INSERT INTO panel_compensation(assignment_id, question_id, subject_id, amount_minor, currency, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""", (assignment_id, question.question_id, row["subject_id"],
                                                   question.pay_per_approved_answer_minor, question.currency, now),
                )
            self.vault._append_audit(
                conn, "panel_quality_review", reviewer, row["subject_id"], question.topic, new_status, None,
                {"assignment_id": assignment_id, "question_id": question.question_id},
            )

    def reconcile(self, assignment_id: str, reference: str, *, now: int | None = None) -> None:
        now = int(time.time()) if now is None else now
        if not reference.strip() or len(reference) > 128:
            raise ValueError("Supply an external payment reference.")
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT subject_id, external_reference FROM panel_compensation WHERE assignment_id = ?",
                               (assignment_id,)).fetchone()
            if row is None or (row["external_reference"] and row["external_reference"] != reference):
                raise ValueError("No matching unsettled compensation entry.")
            if row["external_reference"] == reference:
                return
            conn.execute("UPDATE panel_compensation SET settled_at = ?, external_reference = ? WHERE assignment_id = ?",
                         (now, reference, assignment_id))
            self.vault._append_audit(conn, "panel_compensation_reconciled", "panel_operator",
                                    row["subject_id"] or "panel", "panel", "reconciled", None,
                                    {"assignment_id": assignment_id})

    def pending_reviews(self, topic: str) -> list[dict]:
        with self.vault._connect() as conn:
            rows = conn.execute(
                """SELECT a.assignment_id, a.subject_id, a.answer_json, q.payload_json
                   FROM panel_assignments a JOIN panel_questions q ON q.question_id = a.question_id
                   WHERE a.status = 'submitted' AND q.topic = ? ORDER BY a.answered_at""", (topic,),
            ).fetchall()
        return [{
            "assignment_id": row["assignment_id"], "participant": row["subject_id"],
            "answer": json.loads(row["answer_json"]),
            "question": PanelQuestion.model_validate_json(row["payload_json"]).stimulus,
        } for row in rows]

    def question(self, identifier: str) -> PanelQuestion:
        with self.vault._connect() as conn:
            row = conn.execute("SELECT payload_json FROM panel_questions WHERE question_id = ?",
                               (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return PanelQuestion.model_validate_json(row["payload_json"])

    def report(self, question_id: str, *, expected_topic: str | None = None) -> dict:
        with self.vault._connect() as conn:
            row = conn.execute("SELECT payload_json FROM panel_questions WHERE question_id = ?",
                               (question_id,)).fetchone()
            if row is None:
                raise KeyError(question_id)
            question = PanelQuestion.model_validate_json(row["payload_json"])
            if expected_topic is not None and question.topic != expected_topic:
                raise KeyError(question_id)
            replies = conn.execute(
                """SELECT a.answer_json, m.verified_by FROM panel_assignments a JOIN panel_members m
                   ON m.subject_id = a.subject_id WHERE a.question_id = ? AND a.status = 'approved'
                   AND m.active = 1""", (question_id,),
            ).fetchall()
        choices = Counter(json.loads(row["answer_json"])["option_id"] for row in replies)
        counts = {option.option_id: choices[option.option_id] for option in question.options}
        return {"question_id": question_id, "topic": question.topic, "answer_source": "human_panel",
                "credential_verified_participants": len(replies), "quality_reviewed_answers": len(replies),
                "answers_requested": question.answer_count,
                "option_counts": counts if len(replies) >= 3 else None,
                "verified_identity_providers": sorted({row["verified_by"] for row in replies})}

    def quality(self, subject: str) -> dict:
        with self.vault._connect() as conn:
            rows = conn.execute(
                """SELECT a.status, a.answer_json, q.payload_json FROM panel_assignments a
                   JOIN panel_questions q ON q.question_id = a.question_id
                   WHERE a.subject_id = ? AND a.status IN ('approved', 'rejected')""", (subject,),
            ).fetchall()
        approved = sum(row["status"] == "approved" for row in rows)
        good = {}
        repeated = []
        for row in rows:
            if row["status"] == "approved":
                question = PanelQuestion.model_validate_json(row["payload_json"])
                choice = json.loads(row["answer_json"])["option_id"]
                good[question.question_id] = choice
                if question.repeat_of:
                    repeated.append((question.repeat_of, choice))
        matched = [good[key] == answer for key, answer in repeated if key in good]
        return {"reviews": len(rows), "approved": approved,
                "smoothed_approval_rate": (approved + 1) / (len(rows) + 2),
                "repeat_pairs": len(matched), "repeat_agreement": sum(matched) / len(matched) if matched else None}
