import numpy as np

from anthromesh.benchmarks.public_validation import (
    PopulationAdjustedTwin, _case_predictions, _cohort_values, _fit_confidence, _roc_auc,
)


def test_population_floor_uses_the_prior_without_personal_weight():
    twin = PopulationAdjustedTwin.__new__(PopulationAdjustedTwin)
    twin.priors = [np.array([0.65, 0.35])]
    forecasts = {32: [np.array([[0.35, 0.65], [0.65, 0.35]])]}
    truth = np.array([[0], [1]])
    floor, _, _ = _case_predictions(twin.blend(forecasts, (0, 0), 2), truth)
    blended, _, _ = _case_predictions(twin.blend(forecasts, (32, 0.25), 2), truth)
    assert floor.tolist() == blended.tolist() == [[True], [False]]


def test_calibration_and_minority_metrics_distinguish_correctness_and_false_positives():
    confidence = np.array([[0.8, 0.8, 0.2, 0.2]])
    correct = np.array([[True, True, False, False]])
    bins = _fit_confidence(confidence, correct, np.ones_like(correct))
    assert bins == sorted(bins)
    assert bins[2] < bins[8]
    assert _roc_auc(confidence.ravel(), correct.ravel()) == 1
    _, accurate, called, observed = _cohort_values(np.array([0.95, 0.05, 0.0]), np.array([0.9, 0.05, 0.05]))
    assert (accurate, called, observed) == (1, 2, 1)
