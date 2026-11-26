from __future__ import annotations

import uuid
from enum import Enum
from pydantic import BaseModel, Field, computed_field

from .decision import CandidateOption, ProvenanceTrace, utc_now_iso
from .policy import PolicyVerdict, RiskTier
from .validity import ValidityEnvelope


class EscalationStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class GateAction(str, Enum):
    APPROVED = "approved"
    ESCALATE = "escalate"
    DENIED = "denied"


class EscalationTicket(BaseModel):
    ticket_id: str = Field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:12]}")
    query_id: str
    subject_id: str
    caller_agent: str
    domain: str
    context: str
    candidate_options: list[CandidateOption]
    risk_tier: RiskTier = RiskTier.LOW
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    proposed_action: str | None = None
    confidence: float = 0.0
    status: EscalationStatus = EscalationStatus.PENDING
    human_chosen_action: str | None = None
    human_rationale: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    resolved_at: str | None = None


class GateDecision(BaseModel):
    query_id: str
    action: GateAction
    chosen_action: str | None = None
    rationale: str
    confidence: float
    policy_verdict: PolicyVerdict
    provenance: ProvenanceTrace
    escalation_ticket: EscalationTicket | None = None

    @computed_field
    @property
    def validity_envelope(self) -> ValidityEnvelope:
        citations = self.provenance.evidence_citations
        fields = {
            "domain": self.escalation_ticket.domain if self.escalation_ticket else (
                citations[0].domain if citations else None
            ),
            "evidence_ids": tuple(citation.decision_id for citation in citations),
        }
        if self.action == GateAction.ESCALATE:
            return ValidityEnvelope(status="owner_review", recommended_use="ask_owner", **fields)
        if self.action == GateAction.DENIED:
            return ValidityEnvelope(status="denied", **fields)
        if self.provenance.inference_tier == "fast" and self.provenance.adapter_used == "recorded_preferences":
            return ValidityEnvelope(status="recorded_exact", recommended_use="consult_memory", **fields)
        return ValidityEnvelope(status="experimental", recommended_use="research_only", **fields)
