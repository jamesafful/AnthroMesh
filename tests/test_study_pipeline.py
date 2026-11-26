import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.benchmarks.replay import forecast_study, load_dataset
from anthromesh.benchmarks.study import StudyDataset, load_registration, prepare_study, register_study
from anthromesh.benchmarks.study_report import evaluate_study


def _fixture(tmp_path: Path):
    plan = {
        "study_id": "pilot", "domains": ["calendar"], "backends": ["heuristic:deterministic-v1"],
        "minimum_participants": 2, "minimum_training_cases": 2,
        "minimum_calibration_cases": 3, "minimum_test_cases": 3,
        "minimum_retests_per_domain": 1, "alpha": 0.25, "bootstrap_draws": 500,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    registration_path = tmp_path / "registered.json"
    registration = register_study(plan_path, registration_path)
    beginning = datetime.now(timezone.utc) - timedelta(days=90)
    consents = []
    cases = []
    for person, choice in (("p1", "keep"), ("p2", "move")):
        consents.append({
            "participant_id": person, "study_id": "pilot",
            "consented_at": beginning.isoformat(), "approved_backends": plan["backends"],
        })
        for index in range(10):
            cases.append({
                "case_id": f"case-{index}", "scenario_id": f"scenario-{index}",
                "participant_id": person, "domain": "calendar",
                "context": f"A weekly planning meeting needs a scheduling decision on day {index}.",
                "options": [{"option_id": "keep", "label": "Keep"},
                            {"option_id": "move", "label": "Move"}],
                "chosen_action": choice, "recorded_at": (beginning + timedelta(days=index)).isoformat(),
            })
        cases.append({**cases[-1], "case_id": "repeat", "retest_of": "case-9",
                      "recorded_at": (beginning + timedelta(days=15)).isoformat()})
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"source": "synthetic", "participants": consents, "cases": cases}),
                       encoding="utf-8")
    return registration, registration_path, dataset


def test_full_study_rehearsal_case_isolation_and_report(tmp_path):
    registration, registration_path, dataset = _fixture(tmp_path)
    assert load_registration(registration_path) == registration
    prepared = prepare_study(registration, load_dataset(dataset)[0])
    assert len(prepared.train) == 8 and len(prepared.calibration) == 6 and len(prepared.test) == 6
    assert len(prepared.retests) == 2
    output = tmp_path / "forecasts.json"
    run = forecast_study(registration, dataset, HeuristicAdapter(), output)
    assert len(run.forecasts) == 24
    result = evaluate_study(registration, dataset, [output])
    assert result["fidelity_gate"] == "pilot_only"
    assert result["backends"][0]["self_consistency"]["human_retest_agreement"] == 1
    assert result["backends"][0]["invalid_or_unavailable_predictions"]["population_majority"] == 0
    dataset.write_bytes(dataset.read_bytes() + b" ")
    with pytest.raises(ValueError, match="exact data"):
        evaluate_study(registration, dataset, [output])


def test_study_excludes_revoked_consent(tmp_path):
    registration, _, path = _fixture(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["participants"][0]["revoked_at"] = datetime.now(timezone.utc).isoformat()
    with pytest.raises(ValueError, match="current consent"):
        prepare_study(registration, StudyDataset.model_validate(data))
