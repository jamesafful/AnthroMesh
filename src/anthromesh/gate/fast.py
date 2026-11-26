from __future__ import annotations

from datetime import datetime, timedelta, timezone

from anthromesh.models import (
    EvidenceCitation, GateAction, GateDecision, PolicyEffect, PolicyVerdict, ProvenanceTrace,
    QueryContext, RiskTier,
)
from anthromesh.models.decision import preference_key
from anthromesh.vault.storage import BehavioralVault


class FastDecisionGate:
    """A conservative exact-preference tier for repeated low-risk decisions."""

    confidence = 0.85

    def __init__(self, vault: BehavioralVault, minimum_evidence: int, threshold: float) -> None:
        self.vault = vault
        self.minimum_evidence = max(2, minimum_evidence)
        self.threshold = threshold
        vault.initialize_fast_evidence()

    def evaluate(self, query: QueryContext, verdict: PolicyVerdict) -> GateDecision | None:
        if (verdict.effect != PolicyEffect.ALLOW or query.risk_tier != RiskTier.LOW
                or query.monetary_value is not None or query.metadata or self.confidence < self.threshold):
            return None
        snapshot = self.vault.matching_fast_evidence(
            query.subject_id, query.domain, preference_key(query.context, query.candidate_options),
        )
        now = datetime.now(timezone.utc)
        evidence = []
        for item in snapshot:
            try:
                stamp = datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if stamp.tzinfo and timedelta(0) <= now - stamp <= timedelta(days=90):
                evidence.append(item)
        if len(evidence) < self.minimum_evidence:
            return None
        chosen = evidence[0].chosen_action
        if (chosen not in {option.option_id for option in query.candidate_options}
                or any(item.chosen_action != chosen for item in evidence)):
            return None
        if not self.vault.audit_fast_decision(query, chosen, snapshot, self.confidence):
            return None
        citations = [
            EvidenceCitation(decision_id=item.decision_id, similarity=1, chosen_action=chosen,
                             rationale=item.rationale, domain=item.domain)
            for item in evidence[:5]
        ]
        return GateDecision(
            query_id=query.query_id, action=GateAction.APPROVED, chosen_action=chosen,
            rationale="Recent recorded decisions agree for this exact low-risk context.",
            confidence=self.confidence, policy_verdict=verdict,
            provenance=ProvenanceTrace(
                evidence_citations=citations, evidence_coverage_score=1, agreement_score=1,
                calibration_confidence=self.confidence, adapter_used="recorded_preferences",
                model_used="exact_match", inference_tier="fast",
            ),
        )
