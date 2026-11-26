from __future__ import annotations

import tempfile
from pathlib import Path

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.models import (
    CandidateOption,
    DecisionLog,
    EscalationStatus,
    GateAction,
    QueryContext,
    RiskTier,
)
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def test_twin_approve_escalate_and_active_learning() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = BehavioralVault(db_path=str(Path(tmpdir) / "twin.db"))
        twin = PersonalDecisionTwin(
            vault=vault,
            policy_engine=PolicyEngine(),
            adapter=HeuristicAdapter(),
            confidence_threshold=0.68,
            min_evidence_count=2,
        )
        opts = [
            CandidateOption(option_id="request_changes", label="Request Changes"),
            CandidateOption(option_id="approve", label="Approve"),
        ]

        for idx in range(2):
            twin.record_decision(
                DecisionLog(
                    subject_id="user_1",
                    domain="code_review",
                    context=f"PR #{idx} modifies numerical solver loop without unit tests",
                    options_considered=opts,
                    chosen_action="request_changes",
                    rationale="Always require unit tests on numerical changes.",
                )
            )

        q_in = QueryContext(
            subject_id="user_1",
            caller_agent="pr_bot",
            domain="code_review",
            context="PR modifies solver loop and omits unit tests",
            candidate_options=opts,
            risk_tier=RiskTier.LOW,
        )
        res_in = twin.query_decision(q_in)
        assert res_in.action == GateAction.APPROVED
        assert res_in.chosen_action == "request_changes"
        assert len(res_in.provenance.evidence_citations) == 2

        q_ood = QueryContext(
            subject_id="user_1",
            caller_agent="legal_bot",
            domain="legal_contracts",
            context="Vendor demands perpetual IP rights over fine-tuned models",
            candidate_options=[
                CandidateOption(option_id="redline", label="Redline Clause"),
                CandidateOption(option_id="sign", label="Sign As-Is"),
            ],
            risk_tier=RiskTier.HIGH,
        )
        res_ood = twin.query_decision(q_ood)
        assert res_ood.action == GateAction.ESCALATE
        assert res_ood.escalation_ticket is not None
        assert len(twin.list_pending_escalations()) == 1

        ticket, new_dec = twin.resolve_escalation(
            ticket_id=res_ood.escalation_ticket.ticket_id,
            human_chosen_action="redline",
            human_rationale="Never concede model weight IP ownership.",
        )
        assert ticket.status == EscalationStatus.RESOLVED
        assert new_dec.chosen_action == "redline"
        assert len(twin.list_pending_escalations()) == 0
        assert len(vault.list_decisions(subject_id="user_1", domain="legal_contracts")) == 1
