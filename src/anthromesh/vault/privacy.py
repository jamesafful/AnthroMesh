from __future__ import annotations

import json
from typing import Any

from .dense import delete_indexed_decisions
from .postgres import PostgresBehavioralVault
from .storage import _AUDIT_GENESIS, _audit_hash, BehavioralVault


def _details(row):
    return json.loads(row["details_json"])


def _matching_tasks(conn, subject: str, audit: list, ticket_ids: set[str],
                    decision_ids: set[str] | None) -> list[dict[str, Any]]:
    queries = {detail.get("query_id") for detail in map(_details, audit)}
    tasks = []
    for row in conn.execute("SELECT payload_json FROM a2a_tasks"):
        task = json.loads(row["payload_json"])
        if task.get("skill") != "twin_decision":
            continue
        if task.get("escalation_ticket_id") in ticket_ids:
            tasks.append(task)
            continue
        if decision_ids is None and task.get("subject_id") == subject:
            tasks.append(task)
            continue
        for artifact in task.get("artifacts", []):
            data = artifact.get("data") or {}
            decision = data.get("recorded_decision") or {}
            ticket = data.get("escalation_ticket") or data.get("ticket") or {}
            citations = (data.get("provenance") or {}).get("evidence_citations", [])
            ids = {citation.get("decision_id") for citation in citations}
            if (data.get("query_id") is not None and data["query_id"] in queries
                    or (decision_ids is None and subject in {decision.get("subject_id"), ticket.get("subject_id")})
                    or (decision_ids is not None and (decision.get("decision_id") in decision_ids
                                                       or ids & decision_ids))):
                tasks.append(task)
                break
    return tasks


class SubjectData:
    """Export or erase subject-linked records and derived data managed by this vault."""

    def __init__(self, vault: BehavioralVault) -> None:
        self.vault = vault

    @staticmethod
    def _audit_rows(conn, subject_id: str) -> list:
        return list(conn.execute(
            """SELECT * FROM audit_logs WHERE subject_or_cohort_id = ?
               AND event_type <> 'cohort_simulation' ORDER BY id""",
            (subject_id,),
        ))

    def export(self, subject_id: str) -> dict[str, Any]:
        if not subject_id.strip():
            raise ValueError("A subject ID is required.")
        with self.vault._connect() as conn:
            conn.execute("BEGIN")
            def payloads(table):
                return [json.loads(row["payload_json"]) for row in conn.execute(
                    f"SELECT payload_json FROM {table} WHERE subject_id = ?", (subject_id,),
                )]

            decisions = payloads("decisions")
            tickets = payloads("escalations")
            calibrations = payloads("calibration_profiles")
            audit = self._audit_rows(conn, subject_id)
            tasks = _matching_tasks(conn, subject_id, audit, {item["ticket_id"] for item in tickets}, None)
            grants = list(conn.execute(
                """SELECT token_id, client_id, scopes_json, expires_at, revoked
                   FROM capability_tokens WHERE subject_id = ?""",
                (subject_id,),
            ))
            elicitation = [
                {"scenario": json.loads(row["payload_json"]),
                 "response": json.loads(row["response_json"]) if row["response_json"] else None}
                for row in conn.execute(
                    "SELECT payload_json, response_json FROM elicitation_cases WHERE subject_id = ?",
                    (subject_id,),
                )
            ]
            panel_profile = conn.execute(
                """SELECT profile_json, verified_by, verification_expires_at, active
                   FROM panel_members WHERE subject_id = ?""", (subject_id,),
            ).fetchone()
            panel_answers = [
                {**dict(row), "answer_json": json.loads(row["answer_json"]) if row["answer_json"] else None}
                for row in conn.execute("SELECT * FROM panel_assignments WHERE subject_id = ?", (subject_id,))
            ]
            compensation = [dict(row) for row in conn.execute(
                "SELECT * FROM panel_compensation WHERE subject_id = ?", (subject_id,),
            )]
            twin_cohort_consents = [
                {**dict(row), "options_json": json.loads(row["options_json"])}
                for row in conn.execute("SELECT * FROM cohort_twin_consents WHERE subject_id = ?", (subject_id,))
            ]
            delegations = [
                {"grant": json.loads(row["payload_json"]), "revoked_at": row["revoked_at"]}
                for row in conn.execute(
                    "SELECT payload_json, revoked_at FROM delegation_grants WHERE subject_id = ?", (subject_id,),
                )
            ]
            judgment_settings = [dict(row) for row in conn.execute(
                "SELECT * FROM judgment_settings WHERE subject_id = ?", (subject_id,),
            )]
            prediction_reviews = [
                {**{key: row[key] for key in row.keys() if key != "prediction_json"},
                 "prediction": json.loads(row["prediction_json"])}
                for row in conn.execute("SELECT * FROM prediction_reviews WHERE subject_id = ?", (subject_id,))
            ]
            streams = list(conn.execute(
                "SELECT stream_id FROM capture_checkpoints WHERE subject_id = ?", (subject_id,),
            ))
        return {
            "schema_version": 1,
            "subject_id": subject_id,
            "decisions": decisions,
            "escalations": tickets,
            "a2a_tasks": tasks,
            "calibration_profiles": calibrations,
            "panel": {
                "profile": ({**dict(panel_profile), "profile_json": json.loads(panel_profile["profile_json"])}
                            if panel_profile else None),
                "assignments": panel_answers, "compensation": compensation,
            },
            "cohort_twin_consents": twin_cohort_consents,
            "delegations": delegations,
            "elicitation": elicitation,
            "judgment_settings": judgment_settings,
            "prediction_reviews": prediction_reviews,
            "audit": [{**{key: row[key] for key in row.keys() if key != "details_json"},
                       "details": _details(row)} for row in audit],
            "capabilities": [{"token_id": row["token_id"], "client_id": row["client_id"],
                              "scopes": json.loads(row["scopes_json"]), "expires_at": row["expires_at"],
                              "revoked": bool(row["revoked"])} for row in grants],
            "capture_stream_ids": [row["stream_id"] for row in streams],
        }

    def erase(self, subject_id: str, *, decision_ids: set[str] | None = None) -> dict[str, Any]:
        if not subject_id.strip() or decision_ids == set():
            raise ValueError("Specify a subject and either no decision filter or at least one decision ID.")
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            all_audit = list(conn.execute("SELECT * FROM audit_logs ORDER BY id"))
            old_head = _AUDIT_GENESIS
            for row in all_audit:
                old_head = _audit_hash(old_head, row)
                if old_head != row["entry_hash"]:
                    raise ValueError("Audit verification failed before erasure.")

            records = conn.execute(
                "SELECT decision_id, payload_json FROM decisions WHERE subject_id = ?", (subject_id,),
            ).fetchall()
            known_ids = {row["decision_id"] for row in records}
            removed_decisions = known_ids if decision_ids is None else decision_ids
            if removed_decisions - known_ids:
                raise ValueError("A selected decision does not belong to this subject.")
            evidence_tickets = {
                doc["metadata"]["resolved_from_ticket"] for row in records
                if row["decision_id"] in removed_decisions
                for doc in [json.loads(row["payload_json"])]
                if (doc.get("metadata") or {}).get("resolved_from_ticket")
            }

            tickets = set()
            for row in conn.execute(
                "SELECT ticket_id, payload_json FROM escalations WHERE subject_id = ?", (subject_id,),
            ):
                ticket = json.loads(row["payload_json"])
                if (decision_ids is None or row["ticket_id"] in evidence_tickets
                        or removed_decisions.intersection(ticket.get("evidence_ids", []))):
                    tickets.add(row["ticket_id"])

            affected_cases = set()
            affected_reviews = set()
            for row in conn.execute("SELECT * FROM prediction_reviews WHERE subject_id = ?", (subject_id,)):
                detail = json.loads(row["prediction_json"])
                if (decision_ids is None or row["ticket_id"] in tickets
                        or removed_decisions.intersection(detail.get("evidence_ids", []))):
                    affected_reviews.add(row["query_id"])
                    if row["case_id"]:
                        affected_cases.add(row["case_id"])
            scenarios = []
            for row in conn.execute("SELECT * FROM elicitation_cases WHERE subject_id = ?", (subject_id,)):
                scenario = json.loads(row["payload_json"])
                answer = json.loads(row["response_json"]) if row["response_json"] else {}
                scenarios.append(scenario)
                if (decision_ids is None or scenario.get("observed_decision_id") in removed_decisions
                        or answer.get("decision_id") in removed_decisions):
                    affected_cases.add(scenario["case_id"])
            while True:
                related = {case["case_id"] for case in scenarios if case.get("parent_case_id") in affected_cases}
                if related.issubset(affected_cases):
                    break
                affected_cases.update(related)
            for case_id in affected_cases:
                conn.execute("DELETE FROM elicitation_cases WHERE case_id = ?", (case_id,))
            for query_id in affected_reviews:
                conn.execute("DELETE FROM prediction_reviews WHERE subject_id = ? AND query_id = ?",
                             (subject_id, query_id))

            audit = []
            for row in self._audit_rows(conn, subject_id):
                details = _details(row)
                if (decision_ids is None or details.get("ticket_id") in tickets
                        or details.get("elicitation_case_id") in affected_cases
                        or details.get("query_id") in affected_reviews
                        or details.get("decision_id") in removed_decisions
                        or details.get("new_decision_id") in removed_decisions
                        or removed_decisions.intersection(details.get("evidence_ids", []))):
                    audit.append(row)
            tasks = _matching_tasks(conn, subject_id, audit, tickets, decision_ids)
            delete_indexed_decisions(
                conn, sorted(removed_decisions), postgres=isinstance(self.vault, PostgresBehavioralVault),
            )
            for identifier in removed_decisions:
                conn.execute("DELETE FROM fast_evidence WHERE decision_id = ?", (identifier,))
                conn.execute("DELETE FROM decisions WHERE decision_id = ?", (identifier,))
            for identifier in tickets:
                conn.execute("DELETE FROM mcp_continuations WHERE ticket_id = ?", (identifier,))
                conn.execute("DELETE FROM escalations WHERE ticket_id = ?", (identifier,))
            for task in tasks:
                conn.execute("DELETE FROM a2a_tasks WHERE task_id = ?", (task["id"],))
            calibration = conn.execute("DELETE FROM calibration_profiles WHERE subject_id = ?", (subject_id,))
            digest = self.vault._subject_digest(conn, subject_id, create=True)
            conn.execute("INSERT INTO calibration_revocations VALUES (?) ON CONFLICT DO NOTHING", (digest,))
            if decision_ids is None:
                conn.execute("INSERT INTO subject_tombstones VALUES (?) ON CONFLICT DO NOTHING", (digest,))
                for table in ("capture_checkpoints", "capture_events", "capability_tokens", "judgment_settings", "delegation_grants", "panel_members", "panel_assignments", "cohort_twin_consents"):
                    conn.execute(f"DELETE FROM {table} WHERE subject_id = ?", (subject_id,))

                conn.execute(
                    "UPDATE panel_compensation SET subject_id = NULL WHERE subject_id = ?", (subject_id,),
                )

            else:
                for identifier in removed_decisions:
                    decision_digest = self.vault._subject_digest(conn, "decision:" + identifier)
                    conn.execute("INSERT INTO decision_tombstones VALUES (?) ON CONFLICT DO NOTHING",
                                 (decision_digest,))

            removed_logs = {row["id"] for row in audit}
            head = _AUDIT_GENESIS
            for row in all_audit:
                if row["id"] in removed_logs:
                    conn.execute("DELETE FROM audit_logs WHERE id = ?", (row["id"],))
                else:
                    head = _audit_hash(head, row)
                    if head != row["entry_hash"]:
                        conn.execute("UPDATE audit_logs SET entry_hash = ? WHERE id = ?", (head, row["id"]))
            counts = {"decisions": len(removed_decisions), "escalations": len(tickets),
                      "tasks": len(tasks), "calibration_profiles": calibration.rowcount,
                      "audit_rows": len(removed_logs), "elicitation_cases": len(affected_cases),
                      "prediction_reviews": len(affected_reviews)}
            self.vault._append_audit(
                conn, "privacy_erasure", "vault_owner", "privacy", "governance", "erase",
                None, {"counts": counts, "previous_chain_head": old_head},
            )
            latest = conn.execute("SELECT entry_hash FROM audit_logs ORDER BY id DESC LIMIT 1").fetchone()
            return {"erased": counts, "previous_chain_head": old_head, "audit_head": latest["entry_hash"]}
