from collections import defaultdict

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.cohort.clustering import build_clustered_profile
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.cohort.twin_priors import TwinCohortConsent, blended_archetypes, grant_cohort_consent, revoke_cohort_consent
from anthromesh.models import CandidateOption, CohortArchetype, CohortProfile, CohortQuery, DecisionLog
from anthromesh.vault.storage import BehavioralVault


_OPTIONS = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]


def test_weighted_clustering_preserves_registered_population_margins():
    records = [
        {"region": region, "habit": habit, "response": answer}
        for region in ("north", "south")
        for habit, answer in (("routine", "yes"), ("change", "no"))
        for _ in range(5)
    ]
    profile = build_clustered_profile(
        records, cohort_id="panel", domain="calendar", strata=["region"], features=["habit"],
        response="response", clusters=2, margins={"region": {"north": 0.75, "south": 0.25}},
    )
    regional_mass = defaultdict(float)
    for stratum in profile.archetypes:
        regional_mass[stratum.traits["region"]] += stratum.weight
        assert stratum.prior_tendencies == ({"yes": 1} if stratum.traits["habit"] == "routine" else {"no": 1})
    assert dict(regional_mass) == pytest.approx({"north": 0.75, "south": 0.25})


def test_cohort_aggregate_recomputes_after_revocation_and_hides_small_cells(tmp_path):
    vault = BehavioralVault(str(tmp_path / "consent.db"))
    archetypes = [
        CohortArchetype(archetype_id=key, label=key, weight=0.5, prior_tendencies={"yes": 1, "no": 0})
        for key in ("a", "b")
    ]
    profile = CohortProfile(cohort_id="test", domain="calendar", name="test",
                            twin_prior_weight=0.25, archetypes=archetypes)
    vault.save_cohort(profile)
    for index in range(5):
        subject = f"participant-{index}"
        vault.add_decision(DecisionLog(subject_id=subject, domain="calendar",
                                       context="Would you prefer a change?",
                                       options_considered=_OPTIONS, chosen_action="no", rationale="Private rationale"))
        grant_cohort_consent(vault, TwinCohortConsent(
            subject_id=subject, cohort_id="test", archetype_id="a", domain="calendar",
            option_ids=["yes", "no"], permit_aggregate_prior=True,
        ))
        if index < 4:
            assert not blended_archetypes(vault, profile, {"yes", "no"})[1]
    result, used = blended_archetypes(vault, profile, {"yes", "no"})
    assert used and result[0].prior_tendencies == {"yes": 0.75, "no": 0.25}
    assert "Private rationale" not in result[0].model_dump_json()
    oracle = ResearchCohortOracle(vault, HeuristicAdapter())
    query = CohortQuery(cohort_id="test", stimulus="Would you prefer a change?", options=_OPTIONS)
    simulation = oracle.simulate(query)
    assert simulation.answer_source == "model_simulation"
    assert "consented_twin_aggregates" in simulation.prior_sources
    assert not blended_archetypes(vault, profile.model_copy(update={"name": "different"}), {"yes", "no"})[1]
    revoke_cohort_consent(vault, "participant-0", "test")
    assert "consented_twin_aggregates" not in oracle.simulate(query).prior_sources
