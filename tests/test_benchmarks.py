from __future__ import annotations

from anthromesh.benchmarks.runner import BenchmarkRunner


def test_benchmark_runner_beats_baselines() -> None:
    report = BenchmarkRunner().run_all()
    assert report.twin_anthromesh["in_distribution_accuracy"] == 1.0
    assert report.twin_anthromesh["ood_escalation_recall"] == 1.0
    assert report.twin_anthromesh["expected_calibration_error"] < report.twin_zero_shot_baseline["expected_calibration_error"]

    assert report.cohort_anthromesh["jensen_shannon_divergence"] < report.cohort_mode_collapsed_baseline["jensen_shannon_divergence"]
    assert report.cohort_anthromesh["normalized_entropy"] > report.cohort_mode_collapsed_baseline["normalized_entropy"]
