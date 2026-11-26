import csv
import json

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.benchmarks.public_twin import evaluate_public_twin


class EvidenceCheckingAdapter(HeuristicAdapter):
    def predict_decision(self, query, evidence):
        assert all(record.context.startswith("History") for record, _ in evidence)
        return super().predict_decision(query, evidence)

    def predict_persona(self, query, persona):
        assert "Paired" not in persona
        return super().predict_persona(query, persona)


def test_public_replay_uses_independent_participants_and_questions(tmp_path):
    paired = [f"paired_{index}" for index in range(6)]
    history = ["history_0", "history_1"]
    questions = [
        {
            "QuestionType": "MC",
            "QuestionText": f"{prefix} scheduling preferences {column}",
            "Options": ["Keep the appointment", "Move the appointment"],
            "Settings": {"Selector": "SAVR"},
            "csv_columns": [column],
        }
        for prefix, columns in [("Paired", paired), ("History", history)]
        for column in columns
    ]
    (tmp_path / "question_catalog.json").write_text(json.dumps(questions), encoding="utf-8")
    for filename, columns, value in (
        ("wave1_3_response.csv", paired + history, 1),
        ("wave4_response.csv", paired, 2),
    ):
        with (tmp_path / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["pid", *columns])
            for index in range(16):
                writer.writerow([f"participant-{index}", *[value] * len(columns)])
    result = evaluate_public_twin(tmp_path, EvidenceCheckingAdapter(), seed="fixture", participants=4)
    assert result["paired_evaluation_answers"] == 12
    assert result["evaluation_participants"] == 4
    assert result["calibration_participants"] + result["reference_participants"] == 12
    assert not (set(result["evaluation_question_columns"]) & set(result["calibration_question_columns"]))
    assert result["personal"]["human_retest_agreement"] == 0
    assert result["personal"]["earlier_wave_population_majority_accuracy"] == 0
    assert result["personal"]["offline_owner_resolutions_replayed"] == 3
    assert result["cohort_question_count"] == 3
