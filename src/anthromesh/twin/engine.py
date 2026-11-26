from __future__ import annotations

from typing import Any, Callable

from anthromesh.adapters.base import BaseModelAdapter
from anthromesh.gate.conformal import ConformalCalibration
from anthromesh.gate.engine import ConfidenceGateEngine
from anthromesh.models import (
    DecisionLog,
    EscalationStatus,
    EscalationTicket,
    GateDecision,
    PolicyVerdict,
    QueryContext,
)
from anthromesh.models.decision import utc_now_iso
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.retrieval import HybridDecisionRetriever
from anthromesh.vault.storage import BehavioralVault


class PersonalDecisionTwin:
    """High-level facade for querying, auditing, and actively training an individual's cognitive twin."""

    def __init__(
        self,
        vault: BehavioralVault,
        policy_engine: PolicyEngine,
        adapter: BaseModelAdapter,
        confidence_threshold: float = 0.72,
        min_evidence_count: int = 2,
        retriever: HybridDecisionRetriever | None = None,
        conformal: ConformalCalibration | None = None,
        fast_enabled: bool = False,
        memory_only: bool = False,
        deliberation_seconds: float | None = None,
        monitor_enabled: bool = False,
    ) -> None:
        self.vault = vault
        self.policy_engine = policy_engine
        self.adapter = adapter
        self.gate = ConfidenceGateEngine(
            vault=vault,
            policy_engine=policy_engine,
            adapter=adapter,
            retriever=retriever,
            conformal=conformal,
            fast_enabled=fast_enabled,
            memory_only=memory_only,
            deliberation_seconds=deliberation_seconds,
            monitor_enabled=monitor_enabled,
            confidence_threshold=confidence_threshold,
            min_evidence_count=min_evidence_count,
        )

    def record_decision(self, decision: DecisionLog) -> DecisionLog:
        saved = self.vault.add_decision(decision)
        self.vault.log_audit(
            event_type="decision_recorded",
            caller_agent="human_owner",
            subject_or_cohort_id=decision.subject_id,
            domain=decision.domain,
            action=decision.chosen_action,
            confidence=1.0,
            details={"decision_id": decision.decision_id},
        )
        return saved

    def check_boundary(self, query: QueryContext) -> PolicyVerdict:
        return self.policy_engine.evaluate(query)

    def query_decision(self, query: QueryContext, top_k: int = 5) -> GateDecision:
        return self.gate.evaluate(query, top_k=top_k)

    def list_pending_escalations(self) -> list[EscalationTicket]:
        return self.vault.list_escalations(status=EscalationStatus.PENDING)

    def resolve_escalation(
        self,
        ticket_id: str,
        human_chosen_action: str,
        human_rationale: str,
        tags: list[str] | None = None,
        *,
        complete_task: Callable[[EscalationTicket, DecisionLog], dict[str, Any]] | None = None,
    ) -> tuple[EscalationTicket, DecisionLog]:
        """Resolves a pending escalation ticket and writes the human decision back into the vault (active learning)."""
        ticket = self.vault.get_escalation(ticket_id)
        if ticket is None:
            raise KeyError(f"Escalation ticket '{ticket_id}' not found.")

        valid_ids = {o.option_id for o in ticket.candidate_options}
        if human_chosen_action not in valid_ids:
            raise ValueError(
                f"Invalid chosen action '{human_chosen_action}'. Must be one of: {sorted(valid_ids)}"
            )

        ticket.status = EscalationStatus.RESOLVED
        ticket.human_chosen_action = human_chosen_action
        ticket.human_rationale = human_rationale
        ticket.resolved_at = utc_now_iso()

        new_decision = DecisionLog(
            subject_id=ticket.subject_id,
            domain=ticket.domain,
            context=ticket.context,
            options_considered=ticket.candidate_options,
            chosen_action=human_chosen_action,
            rationale=human_rationale,
            risk_tier=ticket.risk_tier,
            tags=list(tags or ["active_learning", "escalation_resolved"]),
            metadata={"resolved_from_ticket": ticket.ticket_id, "original_query_id": ticket.query_id},
        )
        completed_task = complete_task(ticket, new_decision) if complete_task else None
        self.vault.commit_escalation_resolution(ticket, new_decision, completed_task=completed_task)
        return ticket, new_decision
