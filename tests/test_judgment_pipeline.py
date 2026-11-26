from datetime import datetime, timedelta, timezone

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.judgment.elicitation import Elicitation, ElicitationCase
from anthromesh.judgment.monitor import PredictionMonitor
from anthromesh.judgment.settings import JudgmentSettings, save_judgment_settings
from anthromesh.models import CandidateOption, DecisionLog, QueryContext
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.privacy import SubjectData
from anthromesh.vault.storage import BehavioralVault


_OPTIONS = [CandidateOption(option_id="keep", label="Keep"), CandidateOption(option_id="move", label="Move")]


def test_elicitation_tradeoff_and_blinded_followups(tmp_path):
    vault = BehavioralVault(str(tmp_path / "judgment.db"))
    intake = Elicitation(vault)
    case = ElicitationCase(subject_id="owner", domain="calendar", kind="tradeoff",
                           context="Keep the planning meeting or free up the afternoon?",
                           candidate_options=_OPTIONS, share_with_twin=True)
    with pytest.raises(ValueError, match="not enabled"):
        intake.assign(case)
    save_judgment_settings(vault, JudgmentSettings(
        subject_id="owner", domain="calendar", enable_elicitation=True, allow_training=True,
    ))
    intake.assign(case)
    assert intake.pending("other") == []
    with pytest.raises(ValueError, match="tradeoff"):
        intake.respond("owner", case.case_id, "keep")
    outcome = intake.respond("owner", case.case_id, "keep", "Continuity matters", "Urgent deadline")
    assert vault.get_decision(outcome.decision_id).chosen_action == "keep"
    assert intake.respond(
        "owner", case.case_id, "keep", "Continuity matters", "Urgent deadline"
    ) == outcome

    follow = ElicitationCase(subject_id="owner", domain="calendar", kind="counterfactual",
                             context="An urgent deadline now overlaps the planning meeting.",
                             candidate_options=_OPTIONS, parent_case_id=case.case_id,
                             changed_condition="An urgent deadline overlaps.")
    intake.assign(follow)
    assert intake.pending("owner")[0]["changed_condition"] == "An urgent deadline overlaps."
    assert "chosen_action" not in intake.pending("owner")[0]

    retest = ElicitationCase(subject_id="owner", domain="calendar", kind="retest",
                             context=case.context, candidate_options=_OPTIONS, parent_case_id=case.case_id,
                             available_at=datetime.now(timezone.utc) + timedelta(hours=25))
    intake.assign(retest)
    assert {row["case_id"] for row in intake.pending("owner")} == {follow.case_id}

    result = SubjectData(vault).erase("owner", decision_ids={outcome.decision_id})
    assert result["erased"]["elicitation_cases"] == 3
    assert not intake.pending("owner")


def test_random_audits_are_separate_and_accumulate_a_trust_cell(tmp_path):
    vault = BehavioralVault(str(tmp_path / "audit.db"))
    twin = PersonalDecisionTwin(vault, PolicyEngine(), HeuristicAdapter(),
                                fast_enabled=True, monitor_enabled=True)
    intake = Elicitation(vault)
    for i in range(5):
        subject = f"subject-{i}"
        save_judgment_settings(vault, JudgmentSettings(
            subject_id=subject, domain="calendar", audit_probability=1, person_type="group",
        ))
        for _ in range(2):
            vault.add_decision(DecisionLog(
                subject_id=subject, domain="calendar", context="Weekly planning meeting",
                options_considered=_OPTIONS, chosen_action="keep", rationale="Keep schedules stable.",
            ))
        for j in range(6):
            query = QueryContext(
                subject_id=subject, domain="calendar", context="Weekly planning meeting",
                candidate_options=_OPTIONS,
            )
            decision = twin.query_decision(query)
            assert decision.action.value == "approved"
            assert decision.provenance.answer_source == "model_simulation"
            [item] = intake.pending(subject)
            assert item["kind"] == "blind_prediction_check"
            assert "chosen_action" not in item
            outcome = intake.respond(subject, item["case_id"], "move" if i == j == 0 else "keep")
            assert not outcome.decision_id
        assert len(vault.list_decisions(subject_id=subject)) == 2
        if i < 4:
            assert PredictionMonitor(vault).trust_map() == []
    [cell] = PredictionMonitor(vault).trust_map("calendar")
    assert cell["completed_audits"] == 30
    assert cell["inverse_probability_weighted_accuracy_among_responders"] == pytest.approx(29 / 30)
    novel = QueryContext(subject_id="subject-0", domain="novel", context="New case", candidate_options=_OPTIONS)
    save_judgment_settings(vault, JudgmentSettings(
        subject_id="subject-0", domain="novel", audit_probability=1, person_type="group",
    ))
    pending = twin.query_decision(novel)
    twin.resolve_escalation(pending.escalation_ticket.ticket_id, "move", "New choice")
    review = [row for row in SubjectData(vault).export("subject-0")["prediction_reviews"]
              if row["query_id"] == novel.query_id][0]
    assert review["route"] == "escalation" and review["outcome_action"] == "move"
    assert review["answer_source"] == "human_owner"
    # An owner may resolve the ticket before the monitoring write acquires the vault lock.
    earlier = QueryContext(subject_id="subject-0", domain="novel", context="Another case",
                           candidate_options=_OPTIONS)
    unmonitored = PersonalDecisionTwin(vault, PolicyEngine(), HeuristicAdapter())
    earlier_pending = unmonitored.query_decision(earlier)
    unmonitored.resolve_escalation(earlier_pending.escalation_ticket.ticket_id, "keep", "Earlier reply")
    PredictionMonitor(vault).record(earlier, earlier_pending, predicted_action="move")
    [earlier_review] = [
        row for row in SubjectData(vault).export("subject-0")["prediction_reviews"]
        if row["query_id"] == earlier.query_id
    ]
    assert earlier_review["outcome_action"] == "keep"
    assert earlier_review["answer_source"] == "human_owner"
    SubjectData(vault).erase("subject-0")
    assert PredictionMonitor(vault).trust_map("calendar") == []
