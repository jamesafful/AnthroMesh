from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from anthromesh.gate.calibration import compute_ece

from .replay import SYSTEMS, Forecast, ForecastRun, load_dataset
from .study import Registration, StudyCase, canonical_hash, prepare_study


def _interval(values: list[float], *, draws: int, seed: int, tail: float) -> list[float] | None:
    if len(values) < 2:
        return None
    sampler = random.Random(seed)
    statistics = sorted(fmean(sampler.choices(values, k=len(values))) for _ in range(draws))
    lower = int(tail * (draws - 1))
    upper = int((1 - tail) * (draws - 1))
    return [statistics[lower], statistics[upper]]


def _ratio(count: int, total: int) -> float | None:
    return count / total if total else None


def _check_forecasts(run: ForecastRun, cases: dict[tuple[str, str], StudyCase]):
    seen = {}
    for row in run.forecasts:
        key = (row.participant_id, row.case_id)
        case = cases.get(key)
        if (case is None or row.system not in SYSTEMS or (key, row.system) in seen
                or row.predicted_action not in case.option_ids() | {None}):
            raise ValueError("Forecast records must be unique and use registered cases, systems and options.")
        if row.system == "twin" and row.predicted_action is None and not row.escalated:
            raise ValueError("The twin cannot approve an invalid prediction.")
        if row.system != "twin" and row.escalated:
            raise ValueError("Baseline forecasts are forced choices, independent of their matched selector.")
        seen[key, row.system] = row
    if len(seen) != len(cases) * len(SYSTEMS):
        raise ValueError("Every backend must forecast every case using every registered system.")
    return seen


def _score_backend(run: ForecastRun, prepared, plan, adjusted_tail: float) -> dict:
    cases = {(case.participant_id, case.case_id): case for case in prepared.test}
    rows = _check_forecasts(run, cases)
    groups = defaultdict(list)
    for key, case in cases.items():
        groups[(case.participant_id, case.domain)].append(key)
    correct = {(key, system): rows[key, system].predicted_action == case.chosen_action
               for key, case in cases.items() for system in SYSTEMS}
    invalid = {system: 0 for system in SYSTEMS}
    forced = {system: [] for system in SYSTEMS}
    for key in cases:
        for system in SYSTEMS:
            forced[system].append(correct[key, system])
            invalid[system] += rows[key, system].predicted_action is None

    acceptance = {}
    comparisons = defaultdict(list)
    selected_outcomes = {system: [] for system in SYSTEMS}
    for (person, domain), keys in sorted(groups.items()):
        accepted = [key for key in keys if not rows[key, "twin"].escalated]
        acceptance[person, domain] = len(accepted) / len(keys)
        k = len(accepted)
        if not k:
            continue
        accuracy = fmean(correct[key, "twin"] for key in accepted)
        selected_outcomes["twin"].extend(correct[key, "twin"] for key in accepted)
        for system in SYSTEMS[1:]:
            ranked = sorted(keys, key=lambda key: (
                rows[key, system].predicted_action is None,
                -rows[key, system].confidence,
                canonical_hash([plan.seed, *key]),
            ))
            selected_outcomes[system].extend(correct[key, system] for key in ranked[:k])
            comparisons[system, person].append(
                accuracy - fmean(correct[key, system] for key in ranked[:k])
            )

    paired = {}
    for system in SYSTEMS[1:]:
        person_deltas = [fmean(values) for (name, _), values in sorted(comparisons.items())
                         if name == system]
        limits = _interval(person_deltas, draws=plan.bootstrap_draws, seed=plan.seed, tail=adjusted_tail)
        paired[system] = {
            "participant_mean_accuracy_gain": fmean(person_deltas) if person_deltas else None,
            "adjusted_interval": limits,
            "participants": len(person_deltas),
            "clears_registered_margin": (limits is not None and limits[0] >= plan.accuracy_margin
                                         and len(person_deltas) == len(prepared.participants)),
        }

    retest_by_person = defaultdict(list)
    human_agreements, twin_agreements = [], []
    for first, repeated in prepared.retests:
        human = first.chosen_action == repeated.chosen_action
        twin = correct[(first.participant_id, first.case_id), "twin"]
        human_agreements.append(human)
        twin_agreements.append(twin)
        retest_by_person[first.participant_id].append(int(human) - int(twin))
    gap = [fmean(values) for _, values in sorted(retest_by_person.items())]
    gap_interval = _interval(gap, draws=plan.bootstrap_draws, seed=plan.seed, tail=adjusted_tail)
    consistent = bool(gap_interval and gap_interval[1] <= plan.self_consistency_gap)
    coverage_by_domain = {domain: fmean(value for (_, group), value in acceptance.items() if group == domain)
                          for domain in plan.domains}
    coverage_ok = all(value >= plan.minimum_coverage for value in coverage_by_domain.values())

    holds = {key: rows[key, "twin"].escalated for key in cases}
    tp = sum(holds[key] and case.out_of_distribution for key, case in cases.items())
    fp = sum(holds[key] and not case.out_of_distribution for key, case in cases.items())
    fn = sum(not holds[key] and case.out_of_distribution for key, case in cases.items())
    return {
        "backend": run.backend,
        "forced_choice_accuracy": {system: fmean(values) for system, values in forced.items()},
        "invalid_or_unavailable_predictions": invalid,
        "selective_accuracy_at_matched_case_counts": {
            system: fmean(values) if values else None for system, values in selected_outcomes.items()
        },
        "coverage_by_domain": coverage_by_domain,
        "escalation_rate": fmean(holds.values()),
        "ood_escalation_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "ece_by_system": {
            system: compute_ece([rows[key, system].confidence for key in cases], forced[system])
            for system in SYSTEMS
        },
        "accuracy_gain_by_baseline": paired,
        "self_consistency": {
            "pairs": len(human_agreements), "human_retest_agreement": fmean(human_agreements),
            "twin_agreement_on_same_first_cases": fmean(twin_agreements),
            "human_minus_twin_adjusted_interval": gap_interval,
            "clears_registered_gap": consistent,
        },
        "clears_numeric_gates": coverage_ok and consistent and all(
            row["clears_registered_margin"] for row in paired.values()
        ),
    }


def evaluate_study(registration: Registration, dataset_path: Path, forecast_paths: list[Path]) -> dict:
    plan = registration.plan
    dataset, digest = load_dataset(dataset_path)
    prepared = prepare_study(registration, dataset)
    runs = {}
    for path in forecast_paths:
        run = ForecastRun.model_validate_json(path.read_text(encoding="utf-8"))
        if run.plan_hash != registration.plan_hash or run.dataset_hash != digest or run.backend in runs:
            raise ValueError("Forecast files must match the registered plan and exact data, once per backend.")
        runs[run.backend] = run
    if set(runs) != set(plan.backends):
        raise ValueError("Reports require exactly the preregistered backends.")
    tail = 0.05 / (2 * len(plan.backends) * len(SYSTEMS))
    reports = [_score_backend(runs[name], prepared, plan, tail) for name in plan.backends]
    return {
        "study_id": plan.study_id,
        "purpose": plan.purpose,
        "dataset_source": dataset.source,
        "plan_hash": registration.plan_hash,
        "dataset_hash": digest,
        "participants": len(prepared.participants),
        "training_cases": len(prepared.train),
        "calibration_cases": len(prepared.calibration),
        "test_cases": len(prepared.test),
        "backends": reports,
        "fidelity_gate": ("pass" if all(row["clears_numeric_gates"] for row in reports) else "fail")
        if plan.purpose == "confirmatory" and dataset.source == "consented" else "pilot_only",
    }
