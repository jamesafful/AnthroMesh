from __future__ import annotations

import re

from pydantic import BaseModel, Field

from anthromesh.vault.storage import BehavioralVault


class JudgmentSettings(BaseModel):
    subject_id: str = Field(min_length=1, max_length=128)
    domain: str = Field(min_length=1, max_length=80)
    enable_elicitation: bool = False
    allow_training: bool = False
    audit_probability: float = Field(default=0, ge=0, le=1)
    person_type: str = Field(default="unspecified", pattern=r"^[a-z0-9_-]{1,48}$")


def save_judgment_settings(vault: BehavioralVault, settings: JudgmentSettings) -> None:
    if settings.allow_training and not settings.enable_elicitation:
        raise ValueError("Enable elicitation before opting in to training on elicited answers.")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", settings.domain):
        raise ValueError("Use an explicit domain identifier, without a wildcard.")
    with vault._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        vault._assert_active_subject(conn, settings.subject_id)
        conn.execute(
            """INSERT INTO judgment_settings VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(subject_id, domain) DO UPDATE SET
               enable_elicitation = excluded.enable_elicitation, allow_training = excluded.allow_training,
               audit_probability = excluded.audit_probability, person_type = excluded.person_type""",
            (settings.subject_id, settings.domain, int(settings.enable_elicitation),
             int(settings.allow_training), settings.audit_probability, settings.person_type),
        )


def get_judgment_settings(conn, subject: str, domain: str) -> JudgmentSettings | None:
    row = conn.execute(
        "SELECT * FROM judgment_settings WHERE subject_id = ? AND domain = ?", (subject, domain),
    ).fetchone()
    return JudgmentSettings.model_validate(dict(row)) if row else None
