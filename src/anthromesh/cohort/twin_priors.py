from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from anthromesh.models import CohortArchetype, CohortProfile, DecisionLog
from anthromesh.vault.storage import BehavioralVault


def _profile_hash(profile: CohortProfile) -> str:
    return hashlib.sha256(json.dumps(profile.model_dump(mode="json"), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class TwinCohortConsent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cohort_id: str
    subject_id: str
    archetype_id: str
    domain: str
    option_ids: list[str] = Field(min_length=2, max_length=32)
    ttl_seconds: int = Field(default=30 * 86400, ge=1, le=365 * 86400)
    permit_aggregate_prior: Literal[True]


def grant_cohort_consent(vault: BehavioralVault, consent: TwinCohortConsent, *, now: int | None = None) -> None:
    now = int(time.time()) if now is None else now
    if len(set(consent.option_ids)) != len(consent.option_ids):
        raise ValueError("Use a distinct registered set of response options.")
    with vault._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        vault._assert_active_subject(conn, consent.subject_id)
        row = conn.execute("SELECT payload_json FROM cohorts WHERE cohort_id = ?", (consent.cohort_id,)).fetchone()
        if row is None:
            raise KeyError(consent.cohort_id)
        profile = CohortProfile.model_validate_json(row["payload_json"])
        if (profile.domain != consent.domain or profile.twin_prior_weight <= 0
                or consent.archetype_id not in {item.archetype_id for item in profile.archetypes}):
            raise ValueError("This cohort does not accept the requested aggregate contribution.")
        conn.execute(
            """INSERT INTO cohort_twin_consents VALUES (?, ?, ?, ?, ?, ?, ?, 0)
               ON CONFLICT(cohort_id, subject_id) DO UPDATE SET archetype_id = excluded.archetype_id,
               domain = excluded.domain, options_json = excluded.options_json, profile_hash = excluded.profile_hash,
               expires_at = excluded.expires_at, revoked = 0""",
            (consent.cohort_id, consent.subject_id, consent.archetype_id, consent.domain,
             json.dumps(sorted(consent.option_ids), separators=(",", ":")), _profile_hash(profile),
             now + consent.ttl_seconds),
        )
        vault._append_audit(
            conn, "twin_cohort_consent", "human_owner", consent.subject_id, consent.domain, "grant", None,
            {"cohort_id": consent.cohort_id, "expires_at": now + consent.ttl_seconds},
        )


def revoke_cohort_consent(vault: BehavioralVault, subject: str, cohort_id: str) -> None:
    with vault._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = conn.execute(
            "UPDATE cohort_twin_consents SET revoked = 1 WHERE cohort_id = ? AND subject_id = ? AND revoked = 0",
            (cohort_id, subject),
        )
        if result.rowcount:
            vault._append_audit(conn, "twin_cohort_consent", "human_owner", subject, "cohort",
                                "revoke", None, {"cohort_id": cohort_id})


def blended_archetypes(
    vault: BehavioralVault, profile: CohortProfile, options: set[str], *, now: int | None = None,
) -> tuple[list[CohortArchetype], bool]:
    if profile.twin_prior_weight <= 0:
        return profile.archetypes, False
    now = int(time.time()) if now is None else now
    option_signature = json.dumps(sorted(options), separators=(",", ":"))
    with vault._connect() as conn:
        conn.execute("BEGIN")
        rows = conn.execute(
            """SELECT archetype_id, subject_id, payload_json FROM (
                 SELECT c.archetype_id, d.subject_id, d.payload_json, row_number() OVER
                      (PARTITION BY d.subject_id ORDER BY d.created_at DESC, d.decision_id) AS rank
                 FROM cohort_twin_consents c JOIN decisions d
                      ON d.subject_id = c.subject_id AND d.domain = c.domain
                 WHERE c.cohort_id = ? AND c.domain = ? AND c.options_json = ?
                   AND c.profile_hash = ? AND c.revoked = 0 AND c.expires_at > ?
               ) AS bounded_history WHERE rank <= 100""",
            (profile.cohort_id, profile.domain, option_signature, _profile_hash(profile), now),
        ).fetchall()
    by_person = defaultdict(Counter)
    for row in rows:
        decision = DecisionLog.model_validate_json(row["payload_json"])
        if {option.option_id for option in decision.options_considered} == options and decision.chosen_action in options:
            by_person[row["archetype_id"], row["subject_id"]][decision.chosen_action] += 1
    groups = defaultdict(list)
    for (archetype, _), choices in by_person.items():
        total = sum(choices.values())
        groups[archetype].append({option: choices[option] / total for option in options})
    result = []
    used = False
    for archetype in profile.archetypes:
        distributions = groups[archetype.archetype_id]
        survey_total = sum(max(0, archetype.prior_tendencies.get(option, 0)) for option in options)
        if len(distributions) < 5 or survey_total <= 0:
            result.append(archetype)
            continue
        blend = {
            option: ((1 - profile.twin_prior_weight)
                     * max(0, archetype.prior_tendencies.get(option, 0)) / survey_total
                     + profile.twin_prior_weight * sum(row[option] for row in distributions) / len(distributions))
            for option in sorted(options)
        }
        result.append(archetype.model_copy(update={"prior_tendencies": blend}))
        used = True
    return result, used
