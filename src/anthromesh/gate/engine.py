from __future__ import annotations

from datetime import datetime, timedelta, timezone

from anthromesh.adapters.base import BaseModelAdapter, inference_budget
from anthromesh.judgment.monitor import PredictionMonitor
from anthromesh.models import (
    DecisionLog,
    EscalationTicket,
    EvidenceCitation,
    GateAction,
    GateDecision,
    PolicyEffect,
    PolicyVerdict,
    ProvenanceTrace,
    QueryContext,
)
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.retrieval import HybridDecisionRetriever, tokenize
from anthromesh.vault.storage import BehavioralVault
from .conformal import ConformalCalibration, normalize_scores
from .fast import FastDecisionGate
from .calibration import (
    compute_calibration_confidence,
    compute_evidence_coverage,
    compute_historical_agreement,
)


class ConfidenceGateEngine:
    """Core AnthroMesh decision gate: Policy -> Retrieval -> Inference -> Calibration -> Escalation -> Audit."""

    def __init__(
        self,
        vault: BehavioralVault,
        policy_engine: PolicyEngine,
        adapter: BaseModelAdapter,
        retriever: HybridDecisionRetriever | None = None,
        confidence_threshold: float = 0.72,
        min_evidence_count: int = 2,
        conformal: ConformalCalibration | None = None,
        fast_enabled: bool = False,
        memory_only: bool = False,
        deliberation_seconds: float | None = None,
        monitor_enabled: bool = False,
    ) -> None:
        self.vault = vault
        self.monitor = PredictionMonitor(vault) if monitor_enabled else None
        self.policy_engine = policy_engine
        self.adapter = adapter
        self.retriever = retriever or HybridDecisionRetriever()
        self.confidence_threshold = confidence_threshold
        self.min_evidence_count = min_evidence_count
        self.conformal = conformal
        if deliberation_seconds is not None and not 0 < deliberation_seconds <= 30:
            raise ValueError("Deliberation budget must be positive and at most thirty seconds.")
        self.deliberation_seconds = deliberation_seconds
        self.memory_only = memory_only
        self.fast_gate = (
            FastDecisionGate(vault, min_evidence_count, confidence_threshold)
            if fast_enabled or memory_only else None
        )

    def select_evidence(
        self, query: QueryContext, history: list[DecisionLog], top_k: int = 5,
    ) -> tuple[list[tuple[DecisionLog, float]], list[str]]:
        retrieved = self.retriever.retrieve(query, history, top_k=top_k)
        now = datetime.now(timezone.utc)

        def timestamp(item: DecisionLog) -> datetime:
            try:
                value = datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
                return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            except ValueError:
                return datetime.min.replace(tzinfo=timezone.utc)

        evidence = []
        stale_ids = []
        overrides = [
            item for item, score in retrieved
            if score >= 0.18 and item.metadata.get("resolved_from_ticket")
            and timedelta(0) <= now - timestamp(item) <= timedelta(days=90)
        ]
        for item, score in retrieved:
            if score < 0.18:
                continue
            old_terms = set(tokenize(item.context))
            old_options = {option.option_id for option in item.options_considered}
            for override in overrides:
                if timestamp(override) <= timestamp(item) or override.chosen_action == item.chosen_action:
                    continue
                if old_options != {option.option_id for option in override.options_considered}:
                    continue
                recent_terms = set(tokenize(override.context))
                if len(old_terms & recent_terms) / max(len(old_terms | recent_terms), 1) >= 0.5:
                    stale_ids.append(item.decision_id)
                    score *= 0.5
                    break
            if score >= 0.18:
                evidence.append((item, score))
        return evidence, stale_ids

    def _deny(self, query: QueryContext, verdict: PolicyVerdict) -> GateDecision:
        trace = ProvenanceTrace(
            adapter_used=self.adapter.adapter_name,
            model_used=self.adapter.model,
        )
        decision = GateDecision(
            query_id=query.query_id,
            action=GateAction.DENIED,
            chosen_action=None,
            rationale=verdict.reason,
            confidence=1.0,
            policy_verdict=verdict,
            provenance=trace,
        )
        self.vault.log_audit(
            event_type="gate_decision",
            caller_agent=query.caller_agent,
            subject_or_cohort_id=query.subject_id,
            domain=query.domain,
            action=GateAction.DENIED.value,
            confidence=1.0,
            details={"query_id": query.query_id, "rule": verdict.matched_rule_id, "reason": verdict.reason},
        )
        return decision

    def evaluate_fast(self, query: QueryContext) -> GateDecision | None:
        self.vault.assert_active_subject(query.subject_id)
        verdict = self.policy_engine.evaluate(query)
        if verdict.effect == PolicyEffect.DENY:
            return self._deny(query, verdict)
        if self.fast_gate is None or self.conformal is not None:
            return None
        decision = self.fast_gate.evaluate(query, verdict)
        if decision is not None and self.monitor is not None:
            self.monitor.record(query, decision, decision.chosen_action)
        return decision

    def evaluate(self, query: QueryContext, top_k: int = 5) -> GateDecision:
        self.vault.assert_active_subject(query.subject_id)
        verdict = self.policy_engine.evaluate(query)

        if verdict.effect == PolicyEffect.DENY:
            return self._deny(query, verdict)
        if self.fast_gate is not None and self.conformal is None:
            quick = self.fast_gate.evaluate(query, verdict)
            if quick is not None:
                if self.monitor is not None:
                    self.monitor.record(query, quick, quick.chosen_action)
                return quick

        relevant_evidence = []
        stale_ids = []
        prediction = None
        if not self.memory_only:
            try:
                with inference_budget(self.deliberation_seconds):
                    history = self.vault.list_decisions(subject_id=query.subject_id, domain=query.domain, limit=500)
                    relevant_evidence, stale_ids = self.select_evidence(query, history, top_k)
                    prediction = self.adapter.predict_decision(query, relevant_evidence)
            except TimeoutError:
                pass

        coverage = compute_evidence_coverage(relevant_evidence, self.min_evidence_count)
        agreement = 0.0
        calibrated = 0.0
        prediction_set = None
        if prediction is None:
            uncertainty_gate = True
        else:
            agreement = compute_historical_agreement(prediction.chosen_option, relevant_evidence)
            if self.conformal is None:
                calibrated = compute_calibration_confidence(
                    evidence_coverage=coverage,
                    agreement_score=agreement,
                    model_confidence=prediction.self_reported_confidence,
                    risk_severity=query.risk_tier.severity,
                )
                uncertainty_gate = calibrated < self.confidence_threshold
            else:
                probabilities = normalize_scores(
                    prediction.option_scores, {option.option_id for option in query.candidate_options}
                )
                prediction_set = self.conformal.prediction_set(
                    query.subject_id, query.domain, self.adapter.adapter_name, self.adapter.model,
                    probabilities,
                )
                if self.conformal.externally_managed and self.vault.file_calibration_revoked(query.subject_id):
                    prediction_set = []
                uncertainty_gate = prediction_set != [prediction.chosen_option]
                calibrated = probabilities.get(prediction.chosen_option, 0.0)

        citations = [
            EvidenceCitation(
                decision_id=dec.decision_id,
                similarity=sim,
                chosen_action=dec.chosen_action,
                rationale=dec.rationale,
                domain=dec.domain,
            )
            for dec, sim in relevant_evidence
        ]
        trace = ProvenanceTrace(
            evidence_citations=citations,
            evidence_coverage_score=coverage,
            agreement_score=agreement,
            calibration_confidence=calibrated,
            conformal_prediction_set=prediction_set,
            stale_evidence_ids=stale_ids,
            adapter_used=self.adapter.adapter_name,
            model_used=self.adapter.model,
        )

        needs_escalation = (
            verdict.effect == PolicyEffect.ESCALATE
            or uncertainty_gate
            or len(relevant_evidence) < self.min_evidence_count
        )

        if needs_escalation:
            reasons: list[str] = []
            if verdict.effect == PolicyEffect.ESCALATE:
                reasons.append(verdict.reason)
            if self.memory_only:
                reasons.append("No eligible exact preference match. Owner review is required.")
            elif len(relevant_evidence) < self.min_evidence_count:
                reasons.append(
                    f"Insufficient empirical evidence ({len(relevant_evidence)} < {self.min_evidence_count})."
                )
            if prediction is None and not self.memory_only:
                reasons.append("Deliberative inference exceeded its deadline.")
            elif prediction is not None and uncertainty_gate and prediction_set is not None:
                reasons.append("Conformal calibration did not support a single predicted action.")
            elif prediction is not None and uncertainty_gate:
                reasons.append(
                    f"Calibrated confidence ({calibrated:.2f}) below threshold ({self.confidence_threshold:.2f})."
                )
            reason_str = " ".join(reasons)

            ticket = EscalationTicket(
                query_id=query.query_id,
                subject_id=query.subject_id,
                caller_agent=query.caller_agent,
                domain=query.domain,
                context=query.context,
                candidate_options=query.candidate_options,
                risk_tier=query.risk_tier,
                reason=reason_str,
                evidence_ids=[citation.decision_id for citation in citations],
                proposed_action=prediction.chosen_option if prediction else None,
                confidence=calibrated,
            )
            self.vault.save_escalation(ticket)
            decision = GateDecision(
                query_id=query.query_id,
                action=GateAction.ESCALATE,
                chosen_action=None,
                rationale=reason_str,
                confidence=calibrated,
                policy_verdict=verdict,
                provenance=trace,
                escalation_ticket=ticket,
            )
        else:
            assert prediction is not None
            decision = GateDecision(
                query_id=query.query_id,
                action=GateAction.APPROVED,
                chosen_action=prediction.chosen_option,
                rationale=prediction.rationale,
                confidence=calibrated,
                policy_verdict=verdict,
                provenance=trace,
            )

        self.vault.log_audit(
            event_type="gate_decision",
            caller_agent=query.caller_agent,
            subject_or_cohort_id=query.subject_id,
            domain=query.domain,
            action=decision.action.value,
            confidence=decision.confidence,
            details={
                "query_id": query.query_id,
                "chosen_action": decision.chosen_action,
                "evidence_ids": [c.decision_id for c in citations],
                "ticket_id": decision.escalation_ticket.ticket_id if decision.escalation_ticket else None,
            },
        )
        if self.monitor is not None:
            self.monitor.record(query, decision, prediction.chosen_option if prediction is not None else None)
        return decision
