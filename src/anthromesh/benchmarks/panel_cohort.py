from __future__ import annotations

from collections import defaultdict

from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.judgment.panel import HumanPanel
from anthromesh.models import CohortArchetype, CohortQuery

from .empirical import SurveyObservation, _paired_improvement, _survey


def evaluate_against_panel(
    oracle: ResearchCohortOracle, cohort_id: str, question_ids: list[str], *, sample_size: int = 500,
) -> dict:
    if len(set(question_ids)) != len(question_ids) or not question_ids:
        raise ValueError("Provide distinct, held-out panel question IDs.")
    cohort = oracle.vault.get_cohort(cohort_id)
    if cohort is None:
        raise KeyError(cohort_id)
    panel = HumanPanel(oracle.vault)
    prepared = []
    for identifier in question_ids:
        question = panel.question(identifier)
        if question.topic != cohort.domain or question.repeat_of:
            raise ValueError("Use held-out non-repeated questions from the cohort's domain.")
        observed = panel.report(identifier)
        if observed["option_counts"] is None or observed["quality_reviewed_answers"] < question.answer_count:
            raise ValueError("All requested panel responses must be reviewed before evaluation.")
        prepared.append((question, observed["option_counts"]))

    grouped = defaultdict(list)
    twin_priors_used = []
    backend = f"{oracle.adapter.adapter_name}:{oracle.adapter.model}"
    for question, counts in prepared:
        query = CohortQuery(
            cohort_id=cohort_id, stimulus=question.stimulus, options=question.options,
            sample_size=sample_size, caller_agent="panel_cohort_evaluation",
        )
        survey_only = oracle.simulate(query.model_copy(update={"use_twin_priors": False, "simulation_id": f"base-{query.simulation_id}"}))
        integrated = oracle.simulate(query)
        generic = oracle.adapter.simulate_cohort_archetype(
            question.stimulus, question.options,
            CohortArchetype(archetype_id="general", label="Typical respondent", weight=1),
        )
        if "consented_twin_aggregates" in integrated.prior_sources:
            twin_priors_used.append(question.question_id)
        persona = {option.option_id: generic.option_scores.get(option.option_id, 0.0)
                   for option in question.options}
        for name, prediction in (
            ("persona", persona),
            ("survey_prior", survey_only.metrics.option_distribution),
            ("consented_twin_prior", integrated.metrics.option_distribution),
        ):
            grouped[name].append(SurveyObservation(
                backend=backend, system=name, question_id=question.question_id,
                observed=counts, predicted=prediction,
            ))
    evaluations = {name: _survey(rows) for name, rows in grouped.items()}
    for name in ("survey_prior", "consented_twin_prior"):
        evaluations[name]["paired_improvement_over_persona"] = _paired_improvement(
            grouped["persona"], grouped[name], "survey",
        )
    return {
        "cohort_id": cohort_id, "backend": backend, "ground_truth_source": "human_panel",
        "questions_using_consented_twin_priors": twin_priors_used,
        "simulations": evaluations,
    }
