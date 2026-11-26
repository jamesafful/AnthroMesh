import pytest

from anthromesh.cohort.ingest import build_empirical_profile


def test_raking_preserves_empirical_response_priors() -> None:
    rows = [
        {"region": "north", "answer": "yes", "weight": 3},
        {"region": "north", "answer": "no", "weight": 1},
        {"region": "south", "answer": "no", "weight": 1},
    ]
    result = build_empirical_profile(
        rows, cohort_id="survey", domain="research", traits=["region"], response="answer",
        weight="weight", margins={"region": {"north": 0.4, "south": 0.6}},
    )
    north, south = result.archetypes
    assert north.weight == pytest.approx(0.4)
    assert south.weight == pytest.approx(0.6)
    assert north.prior_tendencies == {"yes": 0.75, "no": 0.25}
