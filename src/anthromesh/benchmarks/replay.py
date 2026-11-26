from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from anthromesh.adapters.base import BaseModelAdapter, StructuredPrediction, extract_json_object
from anthromesh.gate.conformal import CalibrationProfile, ConformalCalibration, normalize_scores
from anthromesh.gate.engine import ConfidenceGateEngine
from anthromesh.models import GateAction, QueryContext
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault

from .study import Registration, StudyCase, StudyDataset, prepare_study


SYSTEMS = ("twin", "ungated_rag", "persona", "population_majority")


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    participant_id: str
    case_id: str
    system: str
    predicted_action: str | None
    confidence: float = Field(ge=0, le=1)
    escalated: bool


class ForecastRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_hash: str
    dataset_hash: str
    backend: str
    forecasts: list[Forecast]


def load_dataset(path: Path) -> tuple[StudyDataset, str]:
    raw = path.read_bytes()
    return StudyDataset.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def _valid_choice(prediction: StructuredPrediction, case: StudyCase) -> str | None:
    if prediction.chosen_option not in case.option_ids():
        return None
    if prediction.raw_text:
        try:
            original = extract_json_object(prediction.raw_text).get("chosen_option")
            if original not in case.option_ids() or original != prediction.chosen_option:
                return None
        except (TypeError, ValueError, AttributeError):
            return None
    return prediction.chosen_option


class _RecordedPredictor:
    """Replay one already-recorded prediction through the production gate."""
    def __init__(self, adapter: BaseModelAdapter) -> None:
        self.adapter_name = adapter.adapter_name
        self.model = adapter.model
        self.query_id: str | None = None
        self.prediction: StructuredPrediction | None = None

    def predict_decision(self, query: QueryContext, evidence):
        if query.query_id != self.query_id or self.prediction is None:
            raise ValueError("Replay prediction does not match the requested case.")
        return self.prediction


def _population_choice(case: StudyCase, items: list[StudyCase]) -> tuple[str | None, float]:
    votes = defaultdict(Counter)
    for item in items:
        if (item.participant_id != case.participant_id and item.recorded_at < case.recorded_at
                and item.option_ids() == case.option_ids()):
            votes[item.participant_id][item.chosen_action] += 1
    if not votes:
        return None, 0
    mass = defaultdict(float)
    for counts in votes.values():
        total = sum(counts.values())
        for option, count in counts.items():
            mass[option] += count / total
    chosen = min(mass, key=lambda option: (-mass[option], option))
    return chosen, mass[chosen] / len(votes)


def forecast_study(
    registration: Registration, dataset_path: Path, adapter: BaseModelAdapter,
    destination: Path, *, allow_remote: bool = False, remote_backend: bool = False,
    vault_key: str | None = None,
) -> ForecastRun:
    if destination.exists():
        raise FileExistsError(destination)
    plan = registration.plan
    dataset, digest = load_dataset(dataset_path)
    prepared = prepare_study(registration, dataset)
    backend = f"{adapter.adapter_name}:{adapter.model}"
    if backend not in plan.backends:
        raise ValueError("The configured model is absent from the registered plan.")
    if remote_backend and not allow_remote:
        raise ValueError("Enable explicit remote inference for a participant-approved backend.")
    if plan.purpose == "confirmatory":
        if adapter.adapter_name == "heuristic":
            raise ValueError("The deterministic rehearsal adapter does not qualify for confirmatory evaluation.")
        if not vault_key:
            raise ValueError("Confirmatory replay requires ANTHROMESH_VAULT_KEY for the temporary vault.")
        if any(not record.persona.strip() for record in prepared.participants.values()):
            raise ValueError("Every confirmatory participant needs a short contemporaneous persona.")
    forecast_rows = []
    train_by_domain = defaultdict(list)
    trained = defaultdict(list)
    for case in prepared.train:
        train_by_domain[case.domain].append(case)
        trained[(case.participant_id, case.domain)].append(case.as_decision())
    with tempfile.TemporaryDirectory(prefix="anthromesh-study-") as temporary:
        vault = BehavioralVault(str(Path(temporary) / "replay.db"), encryption_key=vault_key)
        for records in trained.values():
            for record in records:
                vault.add_decision(record)
        replay = _RecordedPredictor(adapter)
        gate = ConfidenceGateEngine(vault, PolicyEngine(), replay)
        calibration = defaultdict(list)
        for case in prepared.calibration:
            query = case.query()
            evidence, _ = gate.select_evidence(query, trained[(case.participant_id, case.domain)][-500:])
            model_output = adapter.predict_decision(query, evidence)
            probabilities = normalize_scores(model_output.option_scores, case.option_ids())
            choice = _valid_choice(model_output, case)
            calibration[(case.participant_id, case.domain)].append(
                1 - probabilities.get(case.chosen_action, 0) if choice is not None else 1.0
            )
        gate.conformal = ConformalCalibration(
            CalibrationProfile(subject_id=person, domain=domain, adapter=adapter.adapter_name,
                               model=adapter.model, alpha=plan.alpha, scores=scores)
            for (person, domain), scores in calibration.items()
        )

        for case in prepared.test:
            query = case.query()
            evidence, _ = gate.select_evidence(query, trained[(case.participant_id, case.domain)][-500:])
            rag = adapter.predict_decision(query, evidence)
            raw_choice = _valid_choice(rag, case)
            replay.query_id, replay.prediction = query.query_id, rag
            if raw_choice is not None:
                result = gate.evaluate(query)
                twin_choice, confidence, escalated = raw_choice, result.confidence, result.action != GateAction.APPROVED
            else:
                twin_choice, confidence, escalated = None, 0, True
            persona = adapter.predict_persona(query, prepared.participants[case.participant_id].persona)
            majority, majority_confidence = _population_choice(case, train_by_domain[case.domain])
            if plan.purpose == "confirmatory" and majority is None:
                raise ValueError("A confirmatory test case has no prior cross-participant majority baseline.")
            for system, chosen, certainty, hold in (
                ("twin", twin_choice, confidence, escalated),
                ("ungated_rag", raw_choice, rag.self_reported_confidence, False),
                ("persona", _valid_choice(persona, case), persona.self_reported_confidence, False),
                ("population_majority", majority, majority_confidence, False),
            ):
                forecast_rows.append(Forecast(
                    participant_id=case.participant_id, case_id=case.case_id, system=system,
                    predicted_action=chosen, confidence=certainty, escalated=hold,
                ))

    report = ForecastRun(plan_hash=registration.plan_hash, dataset_hash=digest, backend=backend,
                         forecasts=forecast_rows)
    with destination.open("x", encoding="utf-8") as output:
        output.write(report.model_dump_json(indent=2) + "\n")
    return report
