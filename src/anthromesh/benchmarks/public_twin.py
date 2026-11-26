"""Retrospective Twin-2K-500 evaluation using the publisher's standardized CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from urllib.error import URLError
from urllib.request import urlopen

from anthromesh.adapters.base import BaseModelAdapter, StructuredPrediction, extract_json_object
from anthromesh.benchmarks.study_report import _interval
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.gate.calibration import compute_ece, jensen_shannon_divergence
from anthromesh.gate.conformal import CalibrationProfile, ConformalCalibration, normalize_scores
from anthromesh.gate.engine import ConfidenceGateEngine
from anthromesh.models import (
    CandidateOption, CohortArchetype, CohortProfile, CohortQuery, DecisionLog, GateAction, QueryContext,
)
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


DATASET = "LLM-Digital-Twin/Twin-2K-500"
REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
PUBLIC_FILES = {
    "question_catalog.json": "question_catalog_and_human_response_csv",
    "wave1_3_response.csv": "question_catalog_and_human_response_csv",
    "wave4_response.csv": "question_catalog_and_human_response_csv",
    "wave4_formatted_to_catalog_mapping.json": "LLM_simulation_results",
    "responses_wave4_formatted.csv": "LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human",
    "responses_llm_imputed_formatted.csv": "LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human",
}
FILE_HASHES = {
    "question_catalog.json": "e72e305f0c7c83bb990e97de4c0cf2c718ead18cc197239b7feeb354c9ac1bef",
    "wave1_3_response.csv": "fc227b1682e2654fd4b2a8152294b68118c0e3be4d83f4a9af5f3e96df56d3f0",
    "wave4_response.csv": "09ec1a88bdd48fec95963da3102df0ca467378bf729cbc1c0b869b0f1e4fb942",
    "wave4_formatted_to_catalog_mapping.json": "345f7f6645f80dc23bcb423a9bb7c5ae77dbbd17fe572472fdfeb252bdcfc2c7",
    "responses_wave4_formatted.csv": "ace9258a2029f36283a61b6066aa8c9d59458fedefaac0afcbbe7797f4a3aaea",
    "responses_llm_imputed_formatted.csv": "22937217115db30c064489125e39ceb0539c461b678400d8b3e0827962e5436c",
}


def download_public_twin(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "revision.txt"
    if marker.is_file() and marker.read_text(encoding="utf-8-sig").strip() != REVISION:
        raise ValueError("Choose a directory for the pinned Twin-2K-500 revision.")
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
    for filename, directory in PUBLIC_FILES.items():
        destination = folder / filename
        expected = FILE_HASHES[filename]
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{filename} does not match the pinned release.")
            continue
        for attempt in range(3):
            try:
                suffix = "?download=true" if attempt % 2 == 0 else ""
                with tempfile.TemporaryDirectory(dir=folder) as temporary:
                    staged = Path(temporary) / filename
                    with (urlopen(f"{base}/{directory}/{filename}{suffix}", timeout=30) as source,
                          staged.open("wb") as output):
                        shutil.copyfileobj(source, output)
                    if hashlib.sha256(staged.read_bytes()).hexdigest() != expected:
                        raise ValueError(f"Publisher checksum mismatch for {filename}.")
                    staged.replace(destination)
                break
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as output:
            output.write(REVISION + "\n")


@dataclass(frozen=True)
class Item:
    column: str
    text: str
    options: tuple[CandidateOption, ...]

    def answer(self, row: dict[str, str]) -> str | None:
        raw = row.get(self.column, "").strip()
        if not raw:
            return None
        try:
            position = float(raw)
        except ValueError as error:
            raise ValueError(f"Non-numeric answer for {self.column}") from error
        if not position.is_integer() or not 1 <= position <= len(self.options):
            raise ValueError(f"Answer code outside catalog for {self.column}")
        return self.options[int(position) - 1].option_id


class _EvaluationPredictor:
    def __init__(self, adapter: BaseModelAdapter) -> None:
        self.adapter = adapter
        self.adapter_name = adapter.adapter_name
        self.model = adapter.model
        self.prediction: StructuredPrediction | None = None

    def predict_decision(self, query, evidence):
        prediction = self.adapter.predict_decision(query, evidence)
        self.prediction = prediction
        return prediction


def _choice(prediction: StructuredPrediction | None, item: Item) -> str | None:
    if prediction is None or prediction.chosen_option not in {o.option_id for o in item.options}:
        return None
    if prediction.raw_text:
        try:
            if extract_json_object(prediction.raw_text).get("chosen_option") != prediction.chosen_option:
                return None
        except (AttributeError, ValueError):
            return None
    return prediction.chosen_option


def _rank(*values: str) -> str:
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def _table(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = csv.DictReader(stream)
        columns = set(table.fieldnames or [])
        if "pid" not in columns or len(columns) != len(table.fieldnames or []):
            raise ValueError(f"{path.name} must be the publisher CSV with unique fields and a pid column.")
        for row in table:
            pid = row.pop("pid")
            if not pid or pid in rows or None in row or None in row.values():
                raise ValueError(f"Malformed or duplicate participant row in {path.name}.")
            rows[pid] = row
    return columns, rows


def _items(catalog: list[dict]) -> dict[str, Item]:
    items = {}
    for entry in catalog:
        if entry["QuestionType"] != "MC" or entry["Settings"].get("Selector") not in {"SAVR", "SAHR"}:
            continue
        labels = [label.strip() for label in entry.get("Options", [])]
        text = entry["QuestionText"].strip()
        [column] = entry["csv_columns"]
        if not text or len(set(labels)) != len(labels) or not 2 <= len(labels) <= 20:
            continue
        if column in items:
            raise ValueError(f"Ambiguous catalog column {column}")
        options = tuple(CandidateOption(option_id=_rank(label.casefold())[:16], label=label) for label in labels)
        if len({option.option_id for option in options}) == len(options):
            items[column] = Item(column, text, options)
    return items


def _query(pid: str, item: Item) -> QueryContext:
    return QueryContext(query_id=f"{pid}:{item.column}", subject_id=pid, caller_agent="public-benchmark",
                        domain="survey", context=item.text, candidate_options=list(item.options))


def _distribution(votes: Counter, options: tuple[CandidateOption, ...]) -> dict[str, float]:
    total = sum(votes.values())
    return {option.option_id: votes[option.option_id] / total for option in options}


def _reference_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pid = row["TWIN_ID"]
            # The publisher includes two Qualtrics metadata rows.
            if not pid.isdecimal():
                continue
            if pid in rows:
                raise ValueError(f"Duplicate participant in {path.name}.")
            rows[pid] = row
    return rows


def _compare_published(folder: Path, cases: list, items: dict[str, Item]) -> dict:
    paths = [folder / name for name in ("wave4_formatted_to_catalog_mapping.json",
                                        "responses_wave4_formatted.csv",
                                        "responses_llm_imputed_formatted.csv")]
    mapping = {row["catalog_csv_column"]: row["formatted_column"]
               for row in json.loads(paths[0].read_text(encoding="utf-8-sig"))}
    human = _reference_rows(paths[1])
    model = _reference_rows(paths[2])
    paired = []
    mismatches = 0
    for pid, column, truth, twin in cases:
        if column not in mapping or pid not in human or pid not in model:
            continue
        formatted = mapping[column]
        item = items[column]
        try:
            published_truth = item.answer({column: human[pid].get(formatted, "")})
            published_choice = item.answer({column: model[pid].get(formatted, "")})
        except ValueError:
            mismatches += 1
            continue
        if published_truth != truth:
            mismatches += 1
        elif published_choice is not None:
            paired.append((twin, published_choice == truth))
    return {
        "matched_answers": len(paired),
        "excluded_invalid_or_unaligned_answers": mismatches,
        "anthromesh_ungated_accuracy": fmean(row[0] for row in paired) if paired else None,
        "publisher_gpt_4_1_mini_accuracy": fmean(row[1] for row in paired) if paired else None,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "same_cases_different_persona_inputs": True,
    }


def evaluate_public_twin(folder: Path, adapter: BaseModelAdapter, *, seed: str, participants: int) -> dict:
    if participants < 4:
        raise ValueError("Choose at least four evaluation participants.")
    catalog_path = folder / "question_catalog.json"
    earlier_path = folder / "wave1_3_response.csv"
    later_path = folder / "wave4_response.csv"
    earlier_columns, earlier = _table(earlier_path)
    later_columns, later = _table(later_path)
    items = _items(json.loads(catalog_path.read_text(encoding="utf-8-sig")))
    paired = sorted(items.keys() & earlier_columns & later_columns, key=lambda key: _rank(seed, "question", key))
    cut = len(paired) // 2
    calibration_items, test_items = paired[:cut], paired[cut:]
    people = sorted(earlier.keys() & later.keys(), key=lambda pid: _rank(seed, "participant", pid))
    if cut < 2 or len(people) < participants + 8:
        raise ValueError("Insufficient paired questions or independent calibration participants.")
    test_people = people[:participants]
    calibration_people = people[participants:participants + min(32, (len(people) - participants) // 2)]
    reference_people = people[participants + len(calibration_people):]
    if not reference_people:
        raise ValueError("No participants remain for the independent population prior.")
    history_items = sorted((items.keys() & earlier_columns) - later_columns)
    # Exclude verbatim duplicates even if the publisher used a different column ID.
    paired_text = {items[key].text.casefold() for key in paired}
    history_items = [key for key in history_items if items[key].text.casefold() not in paired_text]
    domain = "survey"
    records = defaultdict(list)
    confidences, correctness, answered = [], [], []
    aggregates = defaultdict(Counter)
    repeat_correct, majority_correct, persona_correct = [], [], []
    paired_cases = []
    majority_deltas = defaultdict(list)
    owner_replays = {}
    cohort = []
    with tempfile.TemporaryDirectory(prefix="anthromesh-public-replay-") as temp:
        vault = BehavioralVault(str(Path(temp) / "benchmark.db"))
        replay = _EvaluationPredictor(adapter)
        gate = ConfidenceGateEngine(vault, PolicyEngine(), replay)
        for pid in test_people + calibration_people:
            for column in history_items:
                item = items[column]
                action = item.answer(earlier[pid])
                if action is None:
                    continue
                record = DecisionLog(
                    decision_id=f"{pid}:{column}", subject_id=pid, domain=domain, context=item.text,
                    options_considered=list(item.options), chosen_action=action,
                    rationale=next(option.label for option in item.options if option.option_id == action),
                )
                records[pid].append(record)
        with vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for pid in test_people:
                for record in records[pid]:
                    vault._insert_decision(conn, record)
        scores = []
        for pid in calibration_people:
            for column in calibration_items:
                item = items[column]
                truth = item.answer(later[pid])
                if truth is None:
                    continue
                query = _query(pid, item)
                evidence, _ = gate.select_evidence(query, records[pid])
                prediction = adapter.predict_decision(query, evidence)
                probabilities = normalize_scores(prediction.option_scores, {o.option_id for o in item.options})
                scores.append(1 - probabilities.get(truth, 0) if _choice(prediction, item) else 1)
        if len(scores) < 9:
            raise ValueError("Insufficient observed calibration answers for the 90% prediction set.")
        gate.conformal = ConformalCalibration(
            CalibrationProfile(subject_id=pid, domain=domain, adapter=adapter.adapter_name,
                               model=adapter.model, alpha=0.1, scores=scores)
            for pid in test_people
        )
        oracle = ResearchCohortOracle(vault, adapter)
        for column in test_items:
            item = items[column]
            strata = [Counter(answer for pid in reference_people[group::2]
                              if (answer := item.answer(earlier[pid])) is not None) for group in range(2)]
            prior = strata[0] + strata[1]
            if not prior:
                continue
            prior_distribution = _distribution(prior, item.options)
            common = max(prior_distribution, key=prior_distribution.__getitem__)
            for pid in test_people:
                early = item.answer(earlier[pid])
                truth = item.answer(later[pid])
                if early is None or truth is None:
                    continue
                query = _query(pid, item)
                replay.prediction = None
                result = gate.evaluate(query)
                chosen = _choice(replay.prediction, item)
                persona_summary = "\n".join(
                    f"{record.context} — {record.rationale}" for record in records[pid][:8]
                )[:2000]
                persona = adapter.predict_persona(query, persona_summary)
                is_correct = chosen == truth
                correctness.append(is_correct)
                confidences.append(result.confidence)
                persona_correct.append(_choice(persona, item) == truth)
                majority_correct.append(common == truth)
                repeat_correct.append(early == truth)
                majority_deltas[pid].append(int(is_correct) - int(common == truth))
                paired_cases.append((pid, column, truth, is_correct))
                aggregates[column][truth] += 1
                if result.action == GateAction.APPROVED:
                    answered.append(is_correct)
                elif result.escalation_ticket:
                    owner_replays.setdefault(column, (result.escalation_ticket.ticket_id, truth))
            if not aggregates[column] or not all(strata):
                continue
            profile = CohortProfile(
                cohort_id=f"public-{column}", name="Independent earlier-wave respondents", domain=domain,
                population_weighted=True,
                archetypes=[
                    CohortArchetype(archetype_id=f"reference-{index}", label="Reference sample",
                                    weight=sum(votes.values()) / sum(prior.values()),
                                    prior_tendencies=_distribution(votes, item.options))
                    for index, votes in enumerate(strata)
                ],
            )
            oracle.register_cohort(profile)
            distribution = oracle.simulate(CohortQuery(
                cohort_id=profile.cohort_id, stimulus=item.text, options=list(item.options),
                sample_size=2000, random_seed=int(_rank(seed, column)[:8], 16), use_twin_priors=False,
            )).metrics.option_distribution
            observed = _distribution(aggregates[column], item.options)
            generic = adapter.predict_persona(_query("generic", item), "")
            generic_distribution = normalize_scores(generic.option_scores, set(observed))
            cohort.append((observed, distribution, prior_distribution, generic_distribution))
        # Resolve one sampled ticket per question after all forecasts are frozen so
        # held-out labels cannot enter the histories used to predict later cases.
        twin = PersonalDecisionTwin(vault, PolicyEngine(), adapter)
        for ticket_id, choice in owner_replays.values():
            twin.resolve_escalation(ticket_id, choice, "Published Wave 4 outcome used for offline replay.",
                                    tags=["offline_study_replay"])
        if not vault.verify_audit_chain():
            raise RuntimeError("The study vault audit chain failed after offline escalation replay.")
    if not correctness:
        raise ValueError("No paired evaluation answers matched the question catalog.")
    revision_path = folder / "revision.txt"
    compare = (folder / "wave4_formatted_to_catalog_mapping.json").is_file() and all(
        (folder / filename).is_file() for filename in
        ("responses_wave4_formatted.csv", "responses_llm_imputed_formatted.csv")
    )
    cohort_metrics = {}
    for index, name in enumerate(("oracle", "earlier_wave_empirical_prior", "generic_prompt"), start=1):
        tails = [row[index].get(option, 0) >= probability / 2
                 for row in cohort for option, probability in row[0].items() if 0 < probability < 0.10]
        cohort_metrics[name] = {
            "mean_jsd": fmean(jensen_shannon_divergence(row[0], row[index]) for row in cohort) if cohort else None,
            "minority_recall_at_half_observed_mass": fmean(tails) if tails else None,
        }
    return {
        "dataset": DATASET,
        "license": "CC BY 4.0",
        "dataset_revision": revision_path.read_text(encoding="utf-8-sig").strip() if revision_path.is_file() else None,
        "design": "retrospective_single_select_public_benchmark",
        "adapter": f"{adapter.adapter_name}:{adapter.model}",
        "seed": seed,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (catalog_path, earlier_path, later_path)},
        "evaluation_participants": len(test_people),
        "participants_with_paired_answers": len(majority_deltas),
        "reference_participants": len(reference_people),
        "calibration_participants": len(calibration_people),
        "calibration_question_columns": calibration_items,
        "evaluation_question_columns": test_items,
        "paired_evaluation_answers": len(correctness),
        "personal": {
            "ungated_retrieval_accuracy": fmean(correctness),
            "gate_coverage": len(answered) / len(correctness),
            "gate_selective_accuracy": fmean(answered) if answered else None,
            "escalations_requiring_owner": len(correctness) - len(answered),
            "offline_owner_resolutions_replayed": len(owner_replays),
            "incorrect_approvals": len(answered) - sum(answered),
            "expected_calibration_error": compute_ece(confidences, correctness),
            "persona_prompt_accuracy": fmean(persona_correct),
            "earlier_wave_population_majority_accuracy": fmean(majority_correct),
            "human_retest_agreement": fmean(repeat_correct),
            "participant_bootstrap_95pct_accuracy_difference_vs_majority": _interval(
                [fmean(deltas) for deltas in majority_deltas.values()],
                draws=2000, seed=int(_rank(seed, "bootstrap")[:8], 16), tail=0.025,
            ),
        },
        "published_comparison": _compare_published(folder, paired_cases, items) if compare else None,
        "cohort_question_count": len(cohort),
        "cohort_distribution": cohort_metrics,
    }
