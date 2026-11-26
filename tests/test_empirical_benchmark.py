import json

import pytest

from anthromesh.benchmarks.empirical import score_study


def test_offline_study_scores_selective_accuracy_and_ordinal_distributions(tmp_path) -> None:
    file = tmp_path / "study.jsonl"
    base = {"backend": "local", "system": "anthromesh"}
    cases = [
        {**base, "participant_id": "u1", "case_id": "a", "expected_action": "yes",
         "predicted_action": "yes", "confidence": 0.9, "escalated": False,
         "out_of_distribution": False},
        {**base, "participant_id": "u1", "case_id": "b", "expected_action": "yes",
         "predicted_action": "no", "confidence": 0.3, "escalated": True,
         "out_of_distribution": True},
    ]
    file.write_text("\n".join(map(json.dumps, cases)), encoding="utf-8")
    result = score_study(file, "personal")[0]
    assert result["selective_accuracy"] == 1
    assert result["ood_escalation_f1"] == 1
    assert result["avoided_errors"] == 1
    assert result["escalation_rate"] == 0.5

    survey = {**base, "question_id": "q1", "observed": {"low": 0.95, "high": 0.05},
              "predicted": {"low": 0.9, "high": 0.1}, "ordered_options": ["low", "high"]}
    file.write_text(json.dumps(survey), encoding="utf-8")
    result = score_study(file, "survey")[0]
    assert result["mean_jsd"] > 0
    assert result["mean_ordinal_wasserstein"] == pytest.approx(0.05)
    assert result["long_tail_recall_at_half_observed_mass"] == 1


def test_paired_survey_bootstrap_measures_improvement(tmp_path) -> None:
    file = tmp_path / "paired.jsonl"
    rows = []
    for question in ("q1", "q2"):
        for system, predicted in (
            ("baseline", {"yes": 0.5, "no": 0.5}),
            ("candidate", {"yes": 0.9, "no": 0.1}),
        ):
            rows.append({"backend": "local", "system": system, "question_id": question,
                         "observed": {"yes": 0.9, "no": 0.1}, "predicted": predicted})
    file.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    paired = score_study(file, "survey", baseline="baseline")[1]["paired_improvement"]
    assert paired["cluster_count"] == 2
    assert paired["confidence_interval_95"][0] > 0
