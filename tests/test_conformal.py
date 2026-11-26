from anthromesh.gate.conformal import CalibrationProfile, ConformalCalibration, normalize_scores


def test_conformal_prediction_is_scoped_and_abstains_for_small_calibration_sample() -> None:
    profile = CalibrationProfile(
        subject_id="user", domain="calendar", adapter="local", model="model",
        alpha=0.1, scores=[0.2] * 20,
    )
    calibrator = ConformalCalibration([profile])
    scores = normalize_scores({"accept": 0.9, "decline": 0.1}, {"accept", "decline"})
    assert calibrator.prediction_set("user", "calendar", "local", "model", scores) == ["accept"]
    assert calibrator.prediction_set("other", "calendar", "local", "model", scores) == []
    small_sample = ConformalCalibration([profile.model_copy(update={"scores": [0.2]})])
    assert small_sample.prediction_set("user", "calendar", "local", "model", scores) == [
        "accept", "decline",
    ]


def test_fit_excludes_future_vault_records(tmp_path) -> None:
    from anthromesh.adapters.heuristic import HeuristicAdapter
    from anthromesh.gate.fit import fit_profiles
    from anthromesh.models import CandidateOption, DecisionLog
    from anthromesh.twin.engine import PersonalDecisionTwin
    from anthromesh.vault.policy_engine import PolicyEngine
    from anthromesh.vault.storage import BehavioralVault

    class RecordingAdapter(HeuristicAdapter):
        def predict_decision(self, query, evidence):
            self.seen = [item.decision_id for item, _ in evidence]
            return super().predict_decision(query, evidence)

    vault = BehavioralVault(str(tmp_path / "fit.db"))
    adapter = RecordingAdapter()
    twin = PersonalDecisionTwin(vault, PolicyEngine(), adapter)

    options = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]
    template = {"subject_id": "owner", "domain": "calendar", "context": "Move weekly planning",
                "options_considered": options, "chosen_action": "yes", "rationale": "Weekly planning"}
    for identifier, date in [("past", "2026-01-01"), ("future", "2026-01-03")]:
        vault.add_decision(DecisionLog(
            **template, decision_id=identifier, created_at=f"{date}T13:00:00Z",
        ))
    held_out = DecisionLog(**template, decision_id="holdout", created_at="2026-01-02T13:00:00Z")
    profiles = fit_profiles(twin, [held_out], alpha=0.1)
    assert len(profiles[0].scores) == 1
    assert adapter.seen == ["past"]
