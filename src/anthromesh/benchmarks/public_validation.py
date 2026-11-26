"""Development and held-out validation of population-adjusted personal forecasts."""
from __future__ import annotations

import hashlib
import html
import json
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.benchmarks.public_twin import FILE_HASHES, Item, _items, _query, _rank, _reference_rows, _table
from anthromesh.benchmarks.study_report import _interval
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.gate.calibration import compute_ece, jensen_shannon_divergence
from anthromesh.models import CohortArchetype, CohortProfile, CohortQuery, DecisionLog
from anthromesh.vault.retrieval import HybridDecisionRetriever
from anthromesh.vault.storage import BehavioralVault

SEED = "anthromesh-public-validation-v2"
GRID = ((0, 0.0), *((k, strength) for strength in (0.25, 0.5, 0.75, 1.0) for k in (32, 64, 128, 256)))


@dataclass(frozen=True)
class PublicValidation:
    items: dict[str, Item]
    questions: list[str]
    earlier: dict
    later: dict
    exploratory: list[str]
    selection: list[str]
    calibration: list[str]
    sealed: list[str]
    reference: list[str]
    features: list[str]
    fingerprint: str


def load_public_validation(folder: Path, original_report: Path) -> PublicValidation:
    for filename in ("question_catalog.json", "wave1_3_response.csv", "wave4_response.csv"):
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != FILE_HASHES[filename]:
            raise ValueError(f"Unexpected dataset content for {filename}.")
    catalog = json.loads((folder / "question_catalog.json").read_text(encoding="utf-8-sig"))
    report = json.loads(original_report.read_text(encoding="utf-8-sig"))
    if report["seed"] != "anthromesh-public-v1" or report["evaluation_participants"] != 60:
        raise ValueError("Use the original 60-person AnthroMesh report.")
    items = _items(catalog)
    earlier_cols, earlier = _table(folder / "wave1_3_response.csv")
    later_cols, later = _table(folder / "wave4_response.csv")
    paired = sorted(items.keys() & earlier_cols & later_cols,
                    key=lambda key: _rank(report["seed"], "question", key))
    questions = paired[len(paired) // 2:]
    if questions != report["evaluation_question_columns"]:
        raise ValueError("The reported question split does not match the pinned catalog.")
    people = sorted(earlier.keys() & later.keys(),
                    key=lambda pid: _rank(report["seed"], "participant", pid))
    fresh = sorted(people[92:], key=lambda pid: _rank(SEED, pid))
    if len(fresh) < 500:
        raise ValueError("The full pinned dataset is required.")
    repeated_text = {items[key].text.casefold() for key in paired}
    features = [key for key in sorted((items.keys() & earlier_cols) - later_cols)
                if items[key].text.casefold() not in repeated_text]
    fingerprint = _rank(SEED, repr(questions), repr(features), repr(fresh), repr(GRID), repr(FILE_HASHES))
    return PublicValidation(items, questions, earlier, later, people[:60], fresh[:100], fresh[100:200],
                            fresh[200:400], fresh[400:], features, fingerprint)


def _choices(data: PublicValidation, people: list[str], columns: list[str], wave: dict) -> np.ndarray:
    values = []
    for column in columns:
        item = data.items[column]
        indexes = {option.option_id: i for i, option in enumerate(item.options)}
        values.append([indexes.get(item.answer(wave[person]), -1) for person in people])
    return np.asarray(values, dtype=np.int16).T


def _similar_history(targets: np.ndarray, reference: np.ndarray) -> np.ndarray:
    order = []
    for target in targets:
        overlap = (reference >= 0) & (target >= 0)
        matched = np.count_nonzero(overlap & (reference == target), axis=1)
        similarity = matched / (np.count_nonzero(overlap, axis=1) + 12)
        order.append(np.argsort(-similarity, kind="stable"))
    return np.asarray(order)


class PopulationAdjustedTwin:
    """A cross-question survey predictor with opt-in evidence adjustment."""

    def __init__(self, data: PublicValidation) -> None:
        self.data = data
        self.reference = _choices(data, data.reference, data.features, data.earlier)
        self.early_choices = _choices(data, data.reference, data.questions, data.earlier)
        self.priors = []
        for j, column in enumerate(data.questions):
            observed = self.early_choices[:, j]
            counts = np.bincount(observed[observed >= 0], minlength=len(data.items[column].options))
            self.priors.append((counts + 0.5) / (counts.sum() + 0.5 * len(counts)))

    def forecast(self, people: list[str], sizes: set[int], *, shuffle: bool = False) -> dict:
        if not sizes:
            return {}
        features = _choices(self.data, people, self.data.features, self.data.earlier)
        if shuffle:
            permutation = sorted(range(len(people)), key=lambda i: _rank(SEED, "shuffle", people[i]))
            features = features[permutation]
        neighbors = _similar_history(features, self.reference)
        forecasts = {k: [] for k in sizes}
        for j, prior in enumerate(self.priors):
            labels = self.early_choices[:, j]
            labels = labels[neighbors]
            for k in sizes:
                votes = np.asarray([
                    np.bincount(row[row >= 0][:k], minlength=len(prior))
                    for row in labels
                ])
                forecasts[k].append((votes + 20 * prior) / (votes.sum(axis=1, keepdims=True) + 20))
        return forecasts

    def blend(self, forecasts: dict, setting: tuple[int, float], count: int) -> list[np.ndarray]:
        neighbors, strength = setting
        if strength == 0:
            return [np.broadcast_to(prior, (count, len(prior))) for prior in self.priors]
        return [(1 - strength) * prior + strength * votes
                for prior, votes in zip(self.priors, forecasts[neighbors])]


def _case_predictions(distributions: list[np.ndarray], truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = np.column_stack([dist.argmax(axis=1) for dist in distributions])
    confidence = np.column_stack([dist.max(axis=1) for dist in distributions])
    return chosen == truth, confidence, chosen


def _means(correct: np.ndarray, present: np.ndarray) -> list[float]:
    return [float(row[mask].mean()) for row, mask in zip(correct, present) if mask.any()]


def _ranking(confidence: np.ndarray, mask: np.ndarray, people: list[str], questions: list[str]):
    return sorted(zip(*np.where(mask)),
                  key=lambda ij: (-confidence[ij], _rank(SEED, people[ij[0]], questions[ij[1]])))


def _roc_auc(scores: np.ndarray, correct: np.ndarray) -> float | None:
    counts = Counter(zip(scores.tolist(), correct.tolist()))
    ordered = sorted(set(scores.tolist()))
    negatives = concordant = 0
    for score in ordered:
        bad = counts[score, False]
        concordant += counts[score, True] * (negatives + bad / 2)
        negatives += bad
    positives = int(correct.sum())
    return concordant / (positives * negatives) if positives and negatives else None


def _fit_confidence(confidence: np.ndarray, correct: np.ndarray, present: np.ndarray) -> list[float]:
    bins = np.minimum(9, (confidence[present] * 10).astype(int))
    correct = correct[present]
    blocks = []
    for bin_id in range(10):
        mask = bins == bin_id
        blocks.append([int(correct[mask].sum()) + 1, int(mask.sum()) + 2, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1] = [left + value for left, value in zip(blocks[-1], right)]
    return [correct / total for correct, total, slots in blocks for _ in range(slots)]


def _stage_fit(data: PublicValidation, model: PopulationAdjustedTwin) -> dict:
    people = data.selection
    truth = _choices(data, people, data.questions, data.later)
    present = truth >= 0
    cached = model.forecast(people, {k for k, _ in GRID if k})
    results = []
    for setting in GRID:
        correct, _, _ = _case_predictions(model.blend(cached, setting, len(people)), truth)
        results.append((fmean(_means(correct, present)), setting))
    # GRID starts with the majority and is ordered by increasing blend strength.
    accuracy, selected = max(results, key=lambda row: row[0])
    people = data.calibration
    truth = _choices(data, people, data.questions, data.later)
    cached = model.forecast(people, {selected[0]} - {0})
    correct, confidence, _ = _case_predictions(model.blend(cached, selected, len(people)), truth)
    ranking = _ranking(confidence, truth >= 0, people, data.questions)
    cutoff = ranking[max(0, round(len(ranking) * 0.25) - 1)]
    return {
        "protocol": SEED, "data_fingerprint": data.fingerprint,
        "neighbors": selected[0], "personal_weight": selected[1],
        "selection_accuracy": accuracy, "selection_majority_accuracy": results[0][0],
        "operating_cutoff": [float(confidence[cutoff]), _rank(SEED, people[cutoff[0]], data.questions[cutoff[1]])],
        "calibrated_bins": _fit_confidence(confidence, correct, truth >= 0),
    }


def _bound(data: PublicValidation, correct: np.ndarray, majority: np.ndarray, present: np.ndarray) -> list | None:
    differences = _means(correct.astype(int) - majority.astype(int), present)
    return _interval(differences, draws=5000, seed=int(_rank(SEED, data.fingerprint)[:8], 16), tail=0.025)


def _score(data: PublicValidation, model: PopulationAdjustedTwin, fitted: dict) -> tuple[dict, list[np.ndarray], np.ndarray]:
    if fitted["data_fingerprint"] != data.fingerprint or fitted["protocol"] != SEED:
        raise ValueError("Development artifact does not match the held-out protocol.")
    setting = (fitted["neighbors"], fitted["personal_weight"])
    if setting not in GRID:
        raise ValueError("Frozen model is outside the registered grid.")
    people = data.sealed
    early = _choices(data, people, data.questions, data.earlier)
    truth = _choices(data, people, data.questions, data.later)
    present = (truth >= 0) & (early >= 0)
    cached = model.forecast(people, {setting[0]} - {0})
    distributions = model.blend(cached, setting, len(people))
    correct, confidence, _ = _case_predictions(distributions, truth)
    majority, _, _ = _case_predictions(model.blend({}, (0, 0.0), len(people)), truth)
    shuffled = model.forecast(people, {setting[0]} - {0}, shuffle=True)
    shuffled_correct, _, _ = _case_predictions(model.blend(shuffled, setting, len(people)), truth)
    rank = _ranking(confidence, present, people, data.questions)
    numeric_cutoff, tie_cutoff = fitted["operating_cutoff"]
    operating = [ij for ij in rank if confidence[ij] > numeric_cutoff or (
        confidence[ij] == numeric_cutoff and _rank(SEED, people[ij[0]], data.questions[ij[1]]) <= tie_cutoff
    )]
    curve = {}
    for fraction in (0.1, 0.25, 0.5):
        retained = rank[:max(1, round(len(rank) * fraction))]
        curve[str(fraction)] = {"accuracy": fmean(bool(correct[ij]) for ij in retained),
                                 "majority_accuracy_on_same_cases": fmean(bool(majority[ij]) for ij in retained)}
    interval = _bound(data, correct, majority, present)
    model_score = fmean(_means(correct, present))
    majority_score = fmean(_means(majority, present))
    calibrated = np.asarray(fitted["calibrated_bins"])[np.minimum(9, (confidence[present] * 10).astype(int))]
    oracle_dist = [
        np.eye(len(data.items[column].options))[np.maximum(truth[:, j], 0)]
        for j, column in enumerate(data.questions)
    ]
    oracle_correct, _, _ = _case_predictions(oracle_dist, truth)
    return ({
        "participants": len(people), "paired_answers": len(rank),
        "oracle_accuracy": fmean(_means(oracle_correct, present)),
        "copy_earlier_answer_accuracy": fmean(_means(early == truth, present)),
        "majority_accuracy": majority_score, "personal_accuracy": model_score,
        "shuffled_history_accuracy": fmean(_means(shuffled_correct, present)),
        "paired_bootstrap_95pct": interval,
        "clears_predictive_gate": bool(interval and interval[0] > 0 and model_score - majority_score >= 0.03),
        "confidence_roc_auc": _roc_auc(confidence[present], correct[present]),
        "confidence_deciles": np.quantile(confidence[present], np.arange(11) / 10).tolist(),
        "raw_ece": compute_ece(confidence[present].tolist(), correct[present].tolist()),
        "calibrated_ece": compute_ece(calibrated.tolist(), correct[present].tolist()),
        "risk_coverage": curve,
        "by_question_family": {
            family: {"cases": int(mask.sum()), "personal_accuracy": fmean(_means(correct, mask)),
                     "majority_accuracy": fmean(_means(majority, mask))}
            for family, columns in (
                ("product_pricing", [column.startswith("QID9_") for column in data.questions]),
                ("other_single_select", [not column.startswith("QID9_") for column in data.questions]),
            )
            if (mask := present & np.asarray(columns)).any()
        },
        "frozen_gate": {"coverage": len(operating) / len(rank), "accuracy":
                        fmean(bool(correct[ij]) for ij in operating) if operating else None},
    }, distributions, present)


def _cohort(data: PublicValidation, folder: Path, model: PopulationAdjustedTwin,
            distributions: list[np.ndarray], present: np.ndarray, oracle: ResearchCohortOracle) -> dict:
    people = data.sealed
    early = _choices(data, people, data.questions, data.earlier)
    late = _choices(data, people, data.questions, data.later)
    mapping = {row["catalog_csv_column"]: row["formatted_column"] for row in
               json.loads((folder / "wave4_formatted_to_catalog_mapping.json").read_text(encoding="utf-8-sig"))}
    published = _reference_rows(folder / "responses_llm_imputed_formatted.csv")
    published_human = _reference_rows(folder / "responses_wave4_formatted.csv")
    metrics = {}
    for j, column in enumerate(data.questions):
        item = data.items[column]
        indexes = {option.option_id: i for i, option in enumerate(item.options)}
        selected = [i for i in np.flatnonzero(present[:, j]) if (
            column in mapping and people[i] in published and people[i] in published_human
            and item.answer({column: published_human[people[i]].get(mapping[column], "")})
            == item.options[late[i, j]].option_id
            and item.answer({column: published[people[i]].get(mapping[column], "")}) is not None
        )]
        if len(selected) < 4:
            continue
        formatted = mapping[column]
        published_codes = [indexes[item.answer({column: published[people[i]][formatted]})] for i in selected]
        size = len(item.options)
        def mass(codes) -> np.ndarray:
            return np.bincount(codes, minlength=size) / len(codes)

        observed = mass(late[selected, j])
        profile = CohortProfile(
            cohort_id=column, name="Earlier-wave reference sample", domain="survey", population_weighted=True,
            archetypes=[
                CohortArchetype(
                    archetype_id=f"group-{index}", label="Reference sample", weight=0.5,
                    prior_tendencies=dict(zip(indexes, model.priors[j])),
                )
                for index in range(2)
            ],
        )
        oracle.register_cohort(profile)
        simulation = oracle.simulate(CohortQuery(
            cohort_id=column, stimulus=item.text, options=list(item.options), sample_size=2000,
            random_seed=int(_rank(SEED, column)[:8], 16), use_twin_priors=False,
        ))
        forecasts = {
            "population_prior": model.priors[j],
            "population_adjusted_twin": distributions[j][selected].mean(axis=0),
            "cohort_oracle": np.asarray([simulation.metrics.option_distribution[key] for key in indexes]),
            "uniform": np.full(size, 1 / size),
            "publisher_persona": mass(published_codes),
            "early_vs_late_same_people": mass(early[selected, j]),
        }
        halves = sorted(selected, key=lambda i: _rank(SEED, "noise", column, people[i]))
        noise_left, noise_right = mass(late[halves[::2], j]), mass(late[halves[1::2], j])
        metrics.setdefault("same_wave_split_half", []).append(_cohort_values(noise_left, noise_right))
        for name, predicted in forecasts.items():
            metrics.setdefault(name, []).append(_cohort_values(observed, predicted))
    return {
        name: {"mean_jsd": fmean(row[0] for row in rows),
               "minority_precision": sum(row[1] for row in rows) / sum(row[2] for row in rows)
               if sum(row[2] for row in rows) else None,
               "minority_recall": sum(row[1] for row in rows) / sum(row[3] for row in rows)
               if sum(row[3] for row in rows) else None,
               "questions": len(rows)}
        for name, rows in metrics.items()
    }


def _cohort_values(observed: np.ndarray, predicted: np.ndarray) -> tuple:
    # A 1% floor suppresses numerical dust. A forecast must call an option
    # a minority (1% to <10%) before it earns credit for recovering it.
    truth = (observed >= 0.01) & (observed < 0.10)
    calls = (predicted >= 0.01) & (predicted < 0.10)
    return (jensen_shannon_divergence(dict(enumerate(observed)), dict(enumerate(predicted))),
            int((truth & calls).sum()), int(calls.sum()), int(truth.sum()))


def run_public_validation(folder: Path, original_report: Path, frozen: Path, *, phase: str) -> dict:
    data = load_public_validation(folder, original_report)
    model = PopulationAdjustedTwin(data)
    if phase == "development":
        outcome = _stage_fit(data, model)
        with frozen.open("x", encoding="utf-8") as file:
            file.write(json.dumps(outcome, indent=2) + "\n")
        return outcome
    if phase != "sealed":
        raise ValueError("Use the development or sealed phase.")
    fitted = json.loads(frozen.read_text(encoding="utf-8"))
    results, distributions, present = _score(data, model, fitted)
    with tempfile.TemporaryDirectory(prefix="anthromesh-cohort-validation-") as temporary:
        oracle = ResearchCohortOracle(BehavioralVault(str(Path(temporary) / "cohort.db")), HeuristicAdapter())
        cohort = _cohort(data, folder, model, distributions, present, oracle)
    return {"protocol": SEED, "development_artifact_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "frozen_settings": fitted, "sealed_validation": results,
            "cohort": cohort}


def audit_original_public_run(folder: Path, original_report: Path) -> dict:
    data = load_public_validation(folder, original_report)
    retriever = HybridDecisionRetriever()
    adapter = HeuristicAdapter()
    people = data.exploratory
    source_people = sorted(people, key=lambda pid: _rank(SEED, "original-shuffle", pid))
    histories = {}
    for pid in people:
        histories[pid] = []
        for column in data.features:
            item = data.items[column]
            choice = item.answer(data.earlier[pid])
            if choice is not None:
                histories[pid].append(DecisionLog(
                    decision_id=f"{pid}:{column}", subject_id=pid, domain="survey", context=item.text,
                    options_considered=list(item.options), chosen_action=choice,
                    rationale=next(option.label for option in item.options if option.option_id == choice),
                    created_at="2025-01-01T00:00:00+00:00",
                ))
    cases = defaultdict(list)
    mapping = {row["catalog_csv_column"]: row["formatted_column"] for row in
               json.loads((folder / "wave4_formatted_to_catalog_mapping.json").read_text(encoding="utf-8-sig"))}
    formatted = _reference_rows(folder / "responses_wave4_formatted.csv")
    labeled = {}
    for wave, name in (("earlier", "wave1_3_response_label.csv"), ("later", "wave4_response_label.csv")):
        path = folder / name
        if path.is_file():
            labeled[wave] = _table(path)[1]
    verified_labels = Counter()
    for pid, shuffled_pid in zip(people, source_people):
        for column in data.questions:
            item = data.items[column]
            actual = item.answer(data.later[pid])
            early = item.answer(data.earlier[pid])
            if actual is None or early is None:
                continue
            options = [option.option_id for option in item.options]
            reference = item.answer({column: formatted[pid][mapping[column]]})
            if actual != reference:
                raise ValueError(f"Official response alignment failed for {column}.")
            for wave, code in (("earlier", early), ("later", actual)):
                if wave in labeled:
                    expected_label = item.options[options.index(code)].label
                    if _normalized_label(labeled[wave][pid][column]) != _normalized_label(expected_label):
                        raise ValueError(f"Official {wave} text label does not match the code for {column}.")
                    verified_labels[wave] += 1
            query = _query(pid, item)
            original = adapter.predict_decision(query, retriever.retrieve(query, histories[pid]))
            shuffled = adapter.predict_decision(query, retriever.retrieve(query, histories[shuffled_pid]))
            cases[pid].append((options.index(actual), original.chosen_option == actual,
                               shuffled.chosen_option == actual, early == actual))
    # Feed one-hot ground truth through the same argmax scorer as sealed forecasts.
    oracle_cases = [row[0] for rows in cases.values() for row in rows]
    oracle_truth = np.asarray(oracle_cases).reshape(-1, 1)
    oracle_scored, _, _ = _case_predictions([np.eye(max(oracle_cases) + 1)[oracle_cases]], oracle_truth)
    rows = [row for participant_rows in cases.values() for row in participant_rows]
    expected = json.loads(original_report.read_text(encoding="utf-8-sig"))["paired_evaluation_answers"]
    if len(rows) != expected or not oracle_scored.all():
        raise ValueError("Original case reconstruction or score oracle failed.")
    return {
        "cases": len(rows), "participants": len(cases), "oracle_accuracy": float(oracle_scored.mean()),
        "matched_official_human_codes": len(rows), "question_type": "single_select",
        "verified_text_labels": dict(verified_labels),
        "deterministic_heuristic_replay_accuracy": fmean(row[1] for row in rows),
        "shuffled_history_accuracy": fmean(row[2] for row in rows),
        "copy_earlier_answer_accuracy": fmean(row[3] for row in rows),
        "paired_participant_bootstrap_95pct_linked_minus_shuffled": _interval(
            [fmean(int(row[1]) - int(row[2]) for row in participant_rows)
             for participant_rows in cases.values()], draws=5000, seed=20261123, tail=0.025,
        ),
    }

def _normalized_label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", html.unescape(value)).split()).casefold()
