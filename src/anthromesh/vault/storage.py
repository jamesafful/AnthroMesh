from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from anthromesh.models import (
    CohortProfile,
    DecisionLog,
    EscalationStatus,
    EscalationTicket,
)
from anthromesh.models.decision import preference_key, utc_now_iso


_AUDIT_FIELDS = (
    "id", "event_type", "caller_agent", "subject_or_cohort_id", "domain",
    "action", "confidence", "details_json", "created_at",
)
_AUDIT_GENESIS = "0" * 64


VAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    chosen_action TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_subject_domain
    ON decisions(subject_id, domain);

CREATE TABLE IF NOT EXISTS escalations (
    ticket_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_escalations_status
    ON escalations(status);

CREATE TABLE IF NOT EXISTS cohorts (
    cohort_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judgment_settings (
    subject_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    enable_elicitation INTEGER NOT NULL DEFAULT 0,
    allow_training INTEGER NOT NULL DEFAULT 0,
    audit_probability REAL NOT NULL DEFAULT 0,
    person_type TEXT NOT NULL,
    PRIMARY KEY(subject_id, domain)
);
CREATE TABLE IF NOT EXISTS elicitation_cases (
    case_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    available_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    response_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_elicitation_subject ON elicitation_cases(subject_id, domain);

CREATE TABLE IF NOT EXISTS prediction_reviews (
    subject_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    person_type TEXT NOT NULL,
    backend TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    disposition TEXT NOT NULL,
    audit_probability DOUBLE PRECISION NOT NULL,
    route TEXT NOT NULL,
    case_id TEXT UNIQUE,
    ticket_id TEXT UNIQUE,
    outcome_action TEXT,
    outcome_at TEXT,
    answer_source TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(subject_id, query_id)
);
CREATE INDEX IF NOT EXISTS idx_reviews_domain_group
    ON prediction_reviews(domain, person_type, backend, route);

CREATE TABLE IF NOT EXISTS delegation_grants (
    grant_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    grant_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    revoked_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_delegation_subject ON delegation_grants(subject_id);
CREATE TABLE IF NOT EXISTS delegation_challenges (
    challenge TEXT PRIMARY KEY,
    audience TEXT NOT NULL,
    intent_hash TEXT NOT NULL,
    expires_at BIGINT NOT NULL,
    consumed_at BIGINT
);

CREATE TABLE IF NOT EXISTS panel_members (
    subject_id TEXT PRIMARY KEY,
    person_digest TEXT NOT NULL UNIQUE,
    verified_by TEXT NOT NULL,
    verification_expires_at BIGINT NOT NULL,
    profile_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS panel_questions (
    question_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    expires_at BIGINT NOT NULL,
    created_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS panel_assignments (
    assignment_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL,
    assigned_at BIGINT NOT NULL,
    expires_at BIGINT NOT NULL,
    answer_json TEXT,
    answered_at BIGINT,
    reviewer TEXT,
    review_reason TEXT,
    UNIQUE(question_id, subject_id)
);
CREATE INDEX IF NOT EXISTS idx_panel_assignment_question
    ON panel_assignments(question_id, status);
CREATE TABLE IF NOT EXISTS panel_compensation (
    assignment_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    subject_id TEXT,
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    settled_at BIGINT,
    external_reference TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS cohort_twin_consents (
    cohort_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    archetype_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    options_json TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    expires_at BIGINT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(cohort_id, subject_id)
);
CREATE INDEX IF NOT EXISTS idx_cohort_twin_subject ON cohort_twin_consents(subject_id);

CREATE TABLE IF NOT EXISTS vault_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subject_tombstones (
    subject_digest TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS decision_tombstones (
    decision_digest TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS calibration_revocations (
    subject_digest TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS calibration_profiles (
    subject_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    adapter TEXT NOT NULL,
    model TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(subject_id, domain, adapter, model)
);

CREATE TABLE IF NOT EXISTS capture_checkpoints (
    subject_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    cursor_url TEXT NOT NULL,
    PRIMARY KEY(subject_id, stream_id)
);
CREATE TABLE IF NOT EXISTS capture_events (
    subject_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_start TEXT NOT NULL,
    PRIMARY KEY(subject_id, stream_id, event_id)
);

CREATE TABLE IF NOT EXISTS fast_evidence (
    decision_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    preference_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fast_evidence_lookup
    ON fast_evidence(subject_id, domain, preference_key, created_at);

CREATE TABLE IF NOT EXISTS dense_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_name TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    UNIQUE(index_name, decision_id)
);
CREATE INDEX IF NOT EXISTS idx_dense_decision ON dense_records(decision_id);

CREATE TABLE IF NOT EXISTS capability_tokens (
    token_id TEXT PRIMARY KEY,
    token_hash TEXT UNIQUE NOT NULL,
    subject_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    scopes_json TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    requests_per_minute INTEGER NOT NULL,
    window_number INTEGER NOT NULL DEFAULT 0,
    request_count INTEGER NOT NULL DEFAULT 0,
    revoked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mcp_continuations (
    token_hash TEXT PRIMARY KEY,
    argument_hash TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS a2a_tasks (
    task_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    caller_agent TEXT NOT NULL,
    subject_or_cohort_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _audit_hash(previous: str, row: sqlite3.Row) -> str:
    payload = json.dumps([row[field] for field in _AUDIT_FIELDS], separators=(",", ":"))
    return hashlib.sha256(bytes.fromhex(previous) + payload.encode("utf-8")).hexdigest()


class BehavioralVault:
    """Local-first SQLite behavioral vault storing decision logs, escalations, cohorts, and audit trails."""

    def __init__(
        self, db_path: str = "./anthromesh_vault.db", *, encryption_key: str | None = None
    ) -> None:
        self.db_path = db_path
        self._key = encryption_key
        self._driver = sqlite3
        if encryption_key is not None:
            if not encryption_key or "\x00" in encryption_key:
                raise ValueError("Vault encryption requires a non-empty passphrase.")
            try:
                from sqlcipher3 import dbapi2
            except ImportError as exc:
                raise RuntimeError("Install AnthroMesh with the encryption extra.") from exc
            self._driver = dbapi2
            if Path(db_path).is_file():
                with Path(db_path).open("rb") as source:
                    if source.read(16) == b"SQLite format 3\x00":
                        raise ValueError("Export the existing plaintext vault to an encrypted vault first.")
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._driver.connect(self.db_path)
        try:
            if self._key is not None:
                if conn.execute("PRAGMA cipher_version").fetchone() is None:
                    raise RuntimeError("The configured SQLite driver lacks SQLCipher support.")
                escaped_key = self._key.replace("'", "''")
                conn.execute(f"PRAGMA key = '{escaped_key}'")
            conn.row_factory = self._driver.Row
            conn.execute("PRAGMA secure_delete = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(VAULT_SCHEMA)

            conn.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(audit_logs)")}
            if "entry_hash" not in columns:
                conn.execute("ALTER TABLE audit_logs ADD COLUMN entry_hash TEXT")
                previous = _AUDIT_GENESIS
                for row in conn.execute("SELECT * FROM audit_logs ORDER BY id"):
                    previous = _audit_hash(previous, row)
                    conn.execute(
                        "UPDATE audit_logs SET entry_hash = ? WHERE id = ?", (previous, row["id"])
                    )

    @staticmethod
    def _subject_digest(conn, subject_id: str, *, create: bool = False) -> str | None:
        row = conn.execute("SELECT value FROM vault_settings WHERE key = 'privacy_salt'").fetchone()
        if row is None:
            if not create:
                return None
            salt = secrets.token_hex(32)
            conn.execute("INSERT INTO vault_settings VALUES (?, ?)", ("privacy_salt", salt))
        else:
            salt = row["value"]
        return hashlib.sha256(bytes.fromhex(salt) + subject_id.encode("utf-8")).hexdigest()

    @classmethod
    def _assert_active_subject(cls, conn, subject_id: str) -> None:
        digest = cls._subject_digest(conn, subject_id)
        if digest and conn.execute(
            "SELECT 1 FROM subject_tombstones WHERE subject_digest = ?", (digest,),
        ).fetchone():
            raise ValueError("This subject has been erased from the vault.")

    @classmethod
    def _is_erased_decision(cls, conn, decision_id: str) -> bool:
        digest = cls._subject_digest(conn, "decision:" + decision_id)
        return bool(digest and conn.execute(
            "SELECT 1 FROM decision_tombstones WHERE decision_digest = ?", (digest,),
        ).fetchone())

    def assert_active_subject(self, subject_id: str) -> None:
        with self._connect() as conn:
            self._assert_active_subject(conn, subject_id)

    def file_calibration_revoked(self, subject_id: str) -> bool:
        with self._connect() as conn:
            digest = self._subject_digest(conn, subject_id)
            if digest is None:
                return False
            return bool(conn.execute(
                "SELECT 1 FROM calibration_revocations WHERE subject_digest = ?", (digest,),
            ).fetchone())

    def save_calibration_profiles(self, profiles: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for profile in profiles:
                self._assert_active_subject(conn, profile["subject_id"])
                conn.execute(
                    """INSERT INTO calibration_profiles VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(subject_id, domain, adapter, model) DO UPDATE SET
                       payload_json = excluded.payload_json""",
                    (profile["subject_id"], profile["domain"], profile["adapter"], profile["model"],
                     json.dumps(profile)),
                )

    def load_calibration_profile(self, subject_id: str, domain: str, adapter: str, model: str):
        with self._connect() as conn:
            row = conn.execute(
                """SELECT payload_json FROM calibration_profiles WHERE subject_id = ? AND domain = ?
                   AND adapter = ? AND model = ?""", (subject_id, domain, adapter, model),
            ).fetchone()
            return json.loads(row["payload_json"]) if row else None

    def add_capability(
        self, token_id: str, token_hash: str, subject_id: str, client_id: str, scopes: list[str],
        expires_at: int, requests_per_minute: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_active_subject(conn, subject_id)
            conn.execute(
                """INSERT INTO capability_tokens
                   (token_id, token_hash, subject_id, client_id, scopes_json, expires_at,
                    requests_per_minute) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (token_id, token_hash, subject_id, client_id, json.dumps(scopes), expires_at,
                 requests_per_minute),
            )

    def revoke_capability(self, token_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE capability_tokens SET revoked = 1 WHERE token_id = ?",
                (token_id,),
            )
            return cursor.rowcount == 1

    def consume_capability(self, token_hash: str, now: int) -> dict[str, Any] | None:
        window = now // 60
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE capability_tokens
                   SET request_count = CASE WHEN window_number = ? THEN request_count + 1 ELSE 1 END,
                       window_number = ?
                   WHERE token_hash = ? AND revoked = 0 AND expires_at > ?
                     AND (window_number != ? OR request_count < requests_per_minute)""",
                (window, window, token_hash, now, window),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                """SELECT subject_id, client_id, scopes_json
                   FROM capability_tokens WHERE token_hash = ?""",
                (token_hash,),
            ).fetchone()
            return dict(row)

    def save_mcp_continuation(
        self, token_hash: str, argument_hash: str, ticket_id: str, now: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM escalations WHERE ticket_id = ?", (ticket_id,)).fetchone() is None:
                raise ValueError("Escalation is no longer available.")
            conn.execute("DELETE FROM mcp_continuations WHERE expires_at <= ?", (now,))
            conn.execute(
                "INSERT INTO mcp_continuations VALUES (?, ?, ?, ?)",
                (token_hash, argument_hash, ticket_id, now + 86400),
            )

    def get_mcp_continuation(self, token_hash: str, argument_hash: str, now: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT ticket_id FROM mcp_continuations
                   WHERE token_hash = ? AND argument_hash = ? AND expires_at > ?""",
                (token_hash, argument_hash, now),
            ).fetchone()
        return row["ticket_id"] if row else None

    def create_a2a_task(self, task: dict[str, Any]) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if task.get("subject_id"):
                self._assert_active_subject(conn, task["subject_id"])
            cursor = conn.execute(
                "INSERT INTO a2a_tasks VALUES (?, ?) ON CONFLICT(task_id) DO NOTHING",
                (task["id"], json.dumps(task)),
            )
            return cursor.rowcount == 1

    def save_a2a_task(
        self, task: dict[str, Any], *, expected_state: str
    ) -> bool:
        with self._connect() as conn:
            return self._update_a2a_task(conn, task, expected_state)

    @staticmethod
    def _update_a2a_task(conn: sqlite3.Connection, task: dict[str, Any], expected_state: str) -> bool:
        cursor = conn.execute(
            """UPDATE a2a_tasks SET payload_json = ? WHERE task_id = ?
               AND json_extract(payload_json, '$.status.state') = ?""",
            (json.dumps(task), task["id"], expected_state),
        )
        return cursor.rowcount == 1

    def cancel_a2a_task(self, task: dict[str, Any], expected_state: str) -> bool:
        with self._connect() as conn:
            if not self._update_a2a_task(conn, task, expected_state):
                return False
            ticket_id = task.get("escalation_ticket_id")
            if ticket_id:
                self._expire_escalation(conn, ticket_id)
            return True

    def expire_escalation(self, ticket_id: str) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire_escalation(conn, ticket_id)

    @staticmethod
    def _expire_escalation(conn: sqlite3.Connection, ticket_id: str) -> None:
        row = conn.execute(
            "SELECT payload_json FROM escalations WHERE ticket_id = ? AND status = ?",
            (ticket_id, EscalationStatus.PENDING.value),
        ).fetchone()
        if row is None:
            return
        ticket = EscalationTicket.model_validate_json(row["payload_json"])
        ticket.status = EscalationStatus.EXPIRED
        conn.execute(
            "UPDATE escalations SET status = ?, payload_json = ? WHERE ticket_id = ?",
            (ticket.status.value, ticket.model_dump_json(), ticket_id),
        )

    def get_a2a_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM a2a_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def get_a2a_task_for_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT payload_json FROM a2a_tasks
                   WHERE json_extract(payload_json, '$.escalation_ticket_id') = ?""",
                (ticket_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list_a2a_tasks(
        self, *, after: str, limit: int, context: str | None = None,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        conditions = ["1 = 1"]
        values: list[Any] = []
        if context is not None:
            conditions.append("json_extract(payload_json, '$.context_id') = ?")
            values.append(context)
        if status is not None:
            conditions.append("json_extract(payload_json, '$.status.state') = ?")
            values.append(status)
        where = " AND ".join(conditions)
        with self._connect() as conn:
            conn.execute("BEGIN")
            total = conn.execute(
                f"SELECT count(*) AS total FROM a2a_tasks WHERE {where}", values
            ).fetchone()["total"]
            rows = conn.execute(
                f"""SELECT payload_json FROM a2a_tasks WHERE {where} AND task_id > ?
                    ORDER BY task_id LIMIT ?""",
                [*values, after, limit],
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows], total

    def add_decision(self, decision: DecisionLog) -> DecisionLog:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_active_subject(conn, decision.subject_id)
            if self._is_erased_decision(conn, decision.decision_id):
                raise ValueError("This decision has been erased from the vault.")
            self._insert_decision(conn, decision)
        return decision

    def import_captured_decisions(self, decisions: list[DecisionLog]) -> int:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._capture_rows(conn, decisions)

    def capture_checkpoint(self, subject_id: str, stream_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cursor_url FROM capture_checkpoints WHERE subject_id = ? AND stream_id = ?",
                (subject_id, stream_id),
            ).fetchone()
            return row["cursor_url"] if row else None

    def reset_capture(self, subject_id: str, stream_id: str) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in ("capture_checkpoints", "capture_events"):
                conn.execute(f"DELETE FROM {table} WHERE subject_id = ? AND stream_id = ?",
                             (subject_id, stream_id))

    def capture_event_starts(self, subject_id: str, stream_id: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, event_start FROM capture_events WHERE subject_id = ? AND stream_id = ?",
                (subject_id, stream_id),
            ).fetchall()
            return {row["event_id"]: row["event_start"] for row in rows}

    def commit_capture_page(
        self, subject_id: str, stream_id: str, *, expected_cursor: str | None,
        next_cursor: str, decisions: list[DecisionLog], events: dict[str, str | None],
    ) -> int:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT cursor_url FROM capture_checkpoints WHERE subject_id = ? AND stream_id = ?",
                (subject_id, stream_id),
            ).fetchone()
            if (existing["cursor_url"] if existing else None) != expected_cursor:
                raise ValueError("Capture stream advanced elsewhere; start a new sync.")
            self._assert_active_subject(conn, subject_id)
            inserted = self._capture_rows(conn, decisions)
            for event, start in events.items():
                if start is None:
                    conn.execute(
                        "DELETE FROM capture_events WHERE subject_id = ? AND stream_id = ? AND event_id = ?",
                        (subject_id, stream_id, event),
                    )
                else:
                    conn.execute(
                        """INSERT INTO capture_events VALUES (?, ?, ?, ?)
                           ON CONFLICT(subject_id, stream_id, event_id) DO UPDATE SET
                           event_start = excluded.event_start""",
                        (subject_id, stream_id, event, start),
                    )
            conn.execute(
                """INSERT INTO capture_checkpoints VALUES (?, ?, ?)
                   ON CONFLICT(subject_id, stream_id) DO UPDATE SET
                   cursor_url = excluded.cursor_url""",
                (subject_id, stream_id, next_cursor),
            )
            return inserted

    def _capture_rows(self, conn, decisions: list[DecisionLog]) -> int:
        count = 0
        for decision in decisions:
            self._assert_active_subject(conn, decision.subject_id)
            if self._is_erased_decision(conn, decision.decision_id):
                continue
            existing = conn.execute(
                "SELECT 1 FROM decisions WHERE decision_id = ?", (decision.decision_id,),
            ).fetchone()
            if existing:
                continue
            self._insert_decision(conn, decision)
            self._append_audit(
                conn, "decision_captured", "human_owner", decision.subject_id,
                decision.domain, decision.chosen_action, None,
                {"decision_id": decision.decision_id, "source": decision.metadata["source"]},
            )
            count += 1
        return count

    @staticmethod
    def _insert_decision(conn: sqlite3.Connection, decision: DecisionLog) -> None:
        conn.execute(
            """
            INSERT INTO decisions
            (decision_id, subject_id, domain, chosen_action, risk_tier, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(decision_id) DO UPDATE SET
                subject_id = excluded.subject_id, domain = excluded.domain,
                chosen_action = excluded.chosen_action, risk_tier = excluded.risk_tier,
                payload_json = excluded.payload_json, created_at = excluded.created_at
            """,
            (
                decision.decision_id,
                decision.subject_id,
                decision.domain,
                decision.chosen_action,
                decision.risk_tier.value,
                decision.model_dump_json(),
                decision.created_at,
            ),
        )
        BehavioralVault._upsert_fast_evidence(conn, decision)

    @staticmethod
    def _upsert_fast_evidence(conn, decision: DecisionLog) -> None:
        if decision.risk_tier.value != "low":
            conn.execute("DELETE FROM fast_evidence WHERE decision_id = ?", (decision.decision_id,))
            return
        conn.execute(
            """INSERT INTO fast_evidence VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(decision_id) DO UPDATE SET subject_id = excluded.subject_id,
               domain = excluded.domain, preference_key = excluded.preference_key,
               created_at = excluded.created_at""",
            (decision.decision_id, decision.subject_id, decision.domain,
             preference_key(decision.context, decision.options_considered), decision.created_at),
        )

    def initialize_fast_evidence(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM vault_settings WHERE key = 'fast_index_v1'").fetchone():
                return
            for row in conn.execute("SELECT payload_json FROM decisions"):
                self._upsert_fast_evidence(conn, DecisionLog.model_validate_json(row["payload_json"]))
            conn.execute("INSERT INTO vault_settings VALUES (?, ?)", ("fast_index_v1", "1"))

    def matching_fast_evidence(self, subject_id: str, domain: str, key: str) -> list[DecisionLog]:
        with self._connect() as conn:
            self._assert_active_subject(conn, subject_id)
            rows = conn.execute(
                """SELECT d.payload_json FROM fast_evidence f JOIN decisions d
                   ON f.decision_id = d.decision_id WHERE f.subject_id = ? AND f.domain = ?
                   AND f.preference_key = ? ORDER BY f.created_at DESC LIMIT 32""",
                (subject_id, domain, key),
            ).fetchall()
        return [DecisionLog.model_validate_json(row["payload_json"]) for row in rows]

    def audit_fast_decision(self, query, chosen: str, evidence: list[DecisionLog], confidence: float) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_active_subject(conn, query.subject_id)
            matches = list(conn.execute(
                """SELECT d.payload_json FROM fast_evidence f JOIN decisions d
                   ON f.decision_id = d.decision_id WHERE f.subject_id = ? AND f.domain = ?
                   AND f.preference_key = ? ORDER BY f.created_at DESC LIMIT 32""",
                (query.subject_id, query.domain, preference_key(query.context, query.candidate_options)),
            ))
            if [DecisionLog.model_validate_json(row["payload_json"]) for row in matches] != evidence:
                return False
            self._append_audit(
                conn, "gate_decision", query.caller_agent, query.subject_id, query.domain,
                "approved", confidence,
                {"query_id": query.query_id, "chosen_action": chosen, "inference_tier": "fast",
                 "evidence_ids": [item.decision_id for item in evidence]},
            )
            return True

    def list_decisions(
        self,
        subject_id: str | None = None,
        domain: str | None = None,
        limit: int = 500,
        *,
        before: str | None = None,
    ) -> list[DecisionLog]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject_id:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if before is not None:
            clauses.append("julianday(created_at) < julianday(?)")
            params.append(before)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT payload_json FROM decisions {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [DecisionLog.model_validate_json(r["payload_json"]) for r in rows]

    def get_decision(self, decision_id: str) -> DecisionLog | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        return DecisionLog.model_validate_json(row["payload_json"]) if row else None

    def save_escalation(self, ticket: EscalationTicket) -> EscalationTicket:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_active_subject(conn, ticket.subject_id)
            conn.execute(
                """
                INSERT INTO escalations
                (ticket_id, query_id, subject_id, domain, status, payload_json, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    query_id = excluded.query_id, subject_id = excluded.subject_id,
                    domain = excluded.domain, status = excluded.status,
                    payload_json = excluded.payload_json, created_at = excluded.created_at,
                    resolved_at = excluded.resolved_at
                """,
                (
                    ticket.ticket_id,
                    ticket.query_id,
                    ticket.subject_id,
                    ticket.domain,
                    ticket.status.value,
                    ticket.model_dump_json(),
                    ticket.created_at,
                    ticket.resolved_at,
                ),
            )
        return ticket

    def commit_escalation_resolution(
        self, ticket: EscalationTicket, decision: DecisionLog,
        *, completed_task: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE escalations SET status = ?, payload_json = ?, resolved_at = ?
                   WHERE ticket_id = ? AND status = ?""",
                (ticket.status.value, ticket.model_dump_json(), ticket.resolved_at,
                 ticket.ticket_id, EscalationStatus.PENDING.value),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only pending tickets can be resolved.")
            self._assert_active_subject(conn, ticket.subject_id)
            if completed_task is not None and not self._update_a2a_task(
                conn, completed_task, expected_state="input-required",
            ):
                raise ValueError("The associated A2A task is no longer pending.")
            self._insert_decision(conn, decision)
            conn.execute(
                """UPDATE prediction_reviews SET outcome_action = ?, outcome_at = ?, answer_source = ?
                   WHERE subject_id = ? AND ticket_id = ? AND outcome_action IS NULL""",
                (decision.chosen_action, ticket.resolved_at, "human_owner", ticket.subject_id, ticket.ticket_id),
            )
            self._append_audit(
                conn, "escalation_resolved", "human_owner", ticket.subject_id, ticket.domain,
                decision.chosen_action, 1.0,
                {"ticket_id": ticket.ticket_id, "new_decision_id": decision.decision_id},
            )

    def get_escalation(self, ticket_id: str) -> EscalationTicket | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM escalations WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return EscalationTicket.model_validate_json(row["payload_json"]) if row else None

    def list_escalations(self, status: EscalationStatus | None = None) -> list[EscalationTicket]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT payload_json FROM escalations WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload_json FROM escalations ORDER BY created_at DESC"
                ).fetchall()
        return [EscalationTicket.model_validate_json(r["payload_json"]) for r in rows]

    def save_cohort(self, profile: CohortProfile) -> CohortProfile:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cohorts (cohort_id, domain, name, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cohort_id) DO UPDATE SET domain = excluded.domain,
                    name = excluded.name, payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.cohort_id,
                    profile.domain,
                    profile.name,
                    profile.model_dump_json(),
                    utc_now_iso(),
                ),
            )
        return profile

    def get_cohort(self, cohort_id: str) -> CohortProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM cohorts WHERE cohort_id = ?",
                (cohort_id,),
            ).fetchone()
        return CohortProfile.model_validate_json(row["payload_json"]) if row else None

    def list_cohorts(self) -> list[CohortProfile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM cohorts ORDER BY name ASC").fetchall()
        return [CohortProfile.model_validate_json(r["payload_json"]) for r in rows]

    def log_audit(
        self,
        event_type: str,
        caller_agent: str,
        subject_or_cohort_id: str,
        domain: str,
        action: str,
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if event_type != "cohort_simulation":
                self._assert_active_subject(conn, subject_or_cohort_id)
            self._append_audit(
                conn, event_type, caller_agent, subject_or_cohort_id, domain,
                action, confidence, details,
            )

    @staticmethod
    def _append_audit(
        conn: sqlite3.Connection, event_type: str, caller_agent: str,
        subject_or_cohort_id: str, domain: str, action: str,
        confidence: float | None, details: dict[str, Any] | None,
    ) -> None:
        head = conn.execute(
            "SELECT entry_hash FROM audit_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous = head["entry_hash"] if head else _AUDIT_GENESIS
        cursor = conn.execute(
            """
            INSERT INTO audit_logs
            (event_type, caller_agent, subject_or_cohort_id, domain, action, confidence, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING *
            """,
            (
                event_type,
                caller_agent,
                subject_or_cohort_id,
                domain,
                action,
                confidence,
                json.dumps(details or {}),
                utc_now_iso(),
            ),
        )

        row = cursor.fetchone()
        conn.execute(
            "UPDATE audit_logs SET entry_hash = ? WHERE id = ?",
            (_audit_hash(previous, row), row["id"]),
        )

    def verify_audit_chain(self, expected_head: str | None = None) -> bool:
        """Verify the chain, optionally against a head retained outside the vault."""
        previous = _AUDIT_GENESIS
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM audit_logs ORDER BY id"):
                current = _audit_hash(previous, row)
                if current != row["entry_hash"]:
                    return False
                previous = current
        return expected_head is None or previous == expected_head

    def list_audit_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result
