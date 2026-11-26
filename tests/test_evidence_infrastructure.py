import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.cohort.human_anchor import AnchorRow, AnchorStudy, ForecastRow, analyze_anchors
from anthromesh.models import CandidateOption, DecisionLog, GateAction, QueryContext
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def test_memory_only_exact_approval_and_owner_route(tmp_path):
    class NoModel(HeuristicAdapter):
        def predict_decision(self, *args, **kwargs):
            raise AssertionError("Memory-only consultation must not invoke inference")

    vault = BehavioralVault(str(tmp_path / "vault.db"))
    twin = PersonalDecisionTwin(vault, PolicyEngine(), NoModel(), memory_only=True)
    options = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]
    for subject, timestamp in (("owner", None), ("expired", "2020-01-01T00:00:00+00:00")):
        for _ in range(2):
            fields = {"created_at": timestamp} if timestamp else {}
            vault.add_decision(DecisionLog(
                subject_id=subject, domain="purchase", context="Buy it for $8.",
                options_considered=options, chosen_action="yes", rationale="Recorded choice", **fields,
            ))

    def query(subject, context):
        return twin.query_decision(QueryContext(
            subject_id=subject, domain="purchase", context=context, candidate_options=options,
        ))

    exact = query("owner", "Buy it for $8.")
    envelope = exact.model_dump()["validity_envelope"]
    assert envelope["status"] == "recorded_exact"
    assert len(envelope["evidence_ids"]) == 2
    assert envelope["observed_accuracy"] is None and not envelope["grants_authority"]
    for subject, context in (("owner", "Buy it for $18."), ("expired", "Buy it for $8.")):
        result = query(subject, context)
        assert result.action == GateAction.ESCALATE
        assert result.validity_envelope.recommended_use == "ask_owner"
        assert result.chosen_action is None and result.escalation_ticket.proposed_action is None
        assert len(twin.list_pending_escalations()) >= 1


def test_anchor_rectifies_bias_and_suppresses_small_groups():
    anchors = [AnchorRow(unit=f"a-{i}", group="large", outcome=i % 2, probability=.9) for i in range(50)]
    predictions = [ForecastRow(unit=f"p-{i}", group="large", probability=.9) for i in range(200)]
    predictions.append(ForecastRow(unit="small", group="small", probability=.9))
    study = AnchorStudy(outcome_id="q1", model_revision="model-1", sampling_design="independent synthetic draws",
                        anchors=anchors, predictions=predictions)
    result = analyze_anchors(study)
    assert result["population"]["rectified_proportion"] == pytest.approx(.5)
    assert result["population"]["synthetic_mean"] == pytest.approx(.9)
    lower, upper = result["population"]["rectified_interval"]
    assert lower <= .5 <= upper
    assert result["suppressed_subgroups"] == ["small"]
    study.predictions.append(ForecastRow(unit="a-0", group="large", probability=.5))
    with pytest.raises(ValueError, match="once"):
        analyze_anchors(study)
