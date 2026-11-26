from __future__ import annotations

import json
import secrets
from collections import defaultdict
from datetime import datetime, timezone

from anthromesh.models import EscalationTicket, GateAction, GateDecision, QueryContext
from anthromesh.vault.storage import BehavioralVault

from .elicitation import ElicitationCase
from .settings import get_judgment_settings


class PredictionMonitor:
    def __init__(self, vault: BehavioralVault) -> None:
        self.vault = vault

    def record(self, query: QueryContext, decision: GateDecision, predicted_action: str | None) -> None:
        if decision.action == GateAction.DENIED:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, query.subject_id)
            settings = get_judgment_settings(conn, query.subject_id, query.domain)
            if not settings or settings.audit_probability <= 0:
                return
            route = "escalation" if decision.action == GateAction.ESCALATE else "unselected"
            sample_cutoff = round(settings.audit_probability * 1_000_000)
            if not sample_cutoff:
                return
            case = None
            ticket = decision.escalation_ticket
            if ticket:
                row = conn.execute(
                    "SELECT payload_json FROM escalations WHERE ticket_id = ? AND subject_id = ?",
                    (ticket.ticket_id, query.subject_id),
                ).fetchone()
                if row:
                    ticket = EscalationTicket.model_validate_json(row["payload_json"])
            if route == "unselected" and secrets.randbelow(1_000_000) < sample_cutoff:
                route = "random_audit"
                case = ElicitationCase(
                    subject_id=query.subject_id, domain=query.domain, kind="blind_prediction_check",
                    context=query.context, candidate_options=query.candidate_options,
                )
            details = {
                "predicted_action": predicted_action,
                "answer_source": "model_simulation",
                "evidence_ids": [item.decision_id for item in decision.provenance.evidence_citations],
                "risk_tier": query.risk_tier.value,
            }
            inserted = conn.execute(
                """INSERT INTO prediction_reviews
                   (subject_id, query_id, domain, person_type, backend, prediction_json,
                    confidence, disposition, audit_probability, route, case_id, ticket_id, created_at,
                    outcome_action, outcome_at, answer_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(subject_id, query_id) DO NOTHING""",
                (query.subject_id, query.query_id, query.domain, settings.person_type,
                 decision.provenance.adapter_used + ":" + decision.provenance.model_used,
                 json.dumps(details, separators=(",", ":")), decision.confidence, decision.action.value,
                 sample_cutoff / 1_000_000 if decision.action == GateAction.APPROVED else 0,
                 route, case.case_id if case else None, ticket.ticket_id if ticket else None, now,
                 ticket.human_chosen_action if ticket else None, ticket.resolved_at if ticket else None,
                 "human_owner" if ticket and ticket.human_chosen_action is not None else None),
            )
            if inserted.rowcount and case:
                conn.execute(
                    """INSERT INTO elicitation_cases
                       (case_id, subject_id, domain, available_at, payload_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (case.case_id, case.subject_id, case.domain, case.available_at.isoformat(),
                     case.model_dump_json()),
                )

    def trust_map(self, domain: str | None = None) -> list[dict]:
        filters = " AND r.domain = ?" if domain is not None else ""
        parameters = (domain,) if domain is not None else ()
        with self.vault._connect() as conn:
            rows = conn.execute(
                """SELECT r.* FROM prediction_reviews r JOIN judgment_settings s
                   ON s.subject_id = r.subject_id AND s.domain = r.domain
                   WHERE s.audit_probability > 0""" + filters, parameters,
            ).fetchall()
        cells = defaultdict(list)
        for row in rows:
            cells[row["domain"], row["person_type"], row["backend"]].append(row)
        report = []
        for (domain, person_type, backend), items in sorted(cells.items()):
            invitations = [row for row in items if row["route"] == "random_audit"]
            answered = [row for row in invitations if row["outcome_action"] is not None]
            if len(answered) < 30 or len({row["subject_id"] for row in answered}) < 5:
                continue
            weighted_total = sum(1 / row["audit_probability"] for row in invitations)
            weighted_answered = 0.0
            weighted_correct = 0.0
            bins = defaultdict(lambda: [0.0, 0.0, 0.0])
            for row in answered:
                weight = 1 / row["audit_probability"]
                is_correct = json.loads(row["prediction_json"])["predicted_action"] == row["outcome_action"]
                weighted_answered += weight
                weighted_correct += is_correct * weight
                bucket = bins[min(9, int(row["confidence"] * 10))]
                bucket[0] += weight
                bucket[1] += is_correct * weight
                bucket[2] += row["confidence"] * weight
            report.append({
                "domain": domain, "person_type": person_type, "backend": backend,
                "scope": "random_audits_of_approved_simulations",
                "approved_predictions": sum(row["disposition"] == "approved" for row in items),
                "audit_invitations": len(invitations), "completed_audits": len(answered),
                "completed_owner_escalations": sum(
                    row["route"] == "escalation" and row["outcome_action"] is not None for row in items
                ),
                "inverse_probability_weighted_accuracy_among_responders": weighted_correct / weighted_answered,
                "inverse_probability_weighted_response_rate": weighted_answered / weighted_total,
                "observed_audit_nonresponse_bounds": [
                    weighted_correct / weighted_total,
                    (weighted_correct + weighted_total - weighted_answered) / weighted_total,
                ],
                "weighted_ece_among_responders": sum(
                    abs(count_correct - expected_correct)
                    for _, count_correct, expected_correct in bins.values()
                ) / weighted_answered,
            })
        return report
