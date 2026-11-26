from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anthromesh.models import CandidateOption, DecisionLog, QueryContext, RiskTier


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


class StudyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    purpose: Literal["pilot", "confirmatory"] = "pilot"
    domains: tuple[str, ...] = Field(min_length=1)
    backends: tuple[str, ...] = Field(min_length=1)
    minimum_participants: int = Field(default=30, ge=1)
    minimum_training_cases: int = Field(default=20, ge=2)
    minimum_calibration_cases: int = Field(default=10, ge=1)
    minimum_test_cases: int = Field(default=10, ge=1)
    minimum_retests_per_domain: int = Field(default=5, ge=1)
    calibration_fraction: float = Field(default=0.20, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.20, gt=0, lt=0.5)
    alpha: float = Field(default=0.10, gt=0, lt=0.5)
    accuracy_margin: float = Field(default=0.05, gt=0, lt=1)
    self_consistency_gap: float = Field(default=0.10, ge=0, lt=1)
    minimum_coverage: float = Field(default=0.60, gt=0, le=1)
    minimum_retest_delay_hours: int = Field(default=24, ge=1)
    bootstrap_draws: int = Field(default=2000, ge=500, le=10000)
    seed: int = 42

    @field_validator("domains", "backends")
    @classmethod
    def unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values) or any(not value.strip() for value in values):
            raise ValueError("Study domains and backends must be distinct and nonempty.")
        return values

    def validate_design(self) -> None:
        if self.purpose == "confirmatory" and (
            self.minimum_participants < 30 or len(self.domains) < 2 or len(self.backends) < 3
        ):
            raise ValueError("A confirmatory run needs at least 30 participants, two domains, and three backends.")
        if self.minimum_calibration_cases < math.ceil((1 - self.alpha) / self.alpha):
            raise ValueError("The calibration minimum is too small for a finite conformal cutoff.")


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    registered_at: datetime
    plan: StudyPlan
    plan_hash: str


def register_study(plan_path: Path, destination: Path) -> Registration:
    plan = StudyPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    plan.validate_design()
    registration = Registration(
        registered_at=datetime.now(timezone.utc), plan=plan,
        plan_hash=canonical_hash(plan.model_dump(mode="json")),
    )
    with destination.open("x", encoding="utf-8") as output:
        output.write(registration.model_dump_json(indent=2) + "\n")
    return registration


def load_registration(path: Path) -> Registration:
    registration = Registration.model_validate_json(path.read_text(encoding="utf-8"))
    registration.plan.validate_design()
    if canonical_hash(registration.plan.model_dump(mode="json")) != registration.plan_hash:
        raise ValueError("The registered plan has changed.")
    if registration.registered_at.tzinfo is None:
        raise ValueError("The registration timestamp must include a timezone.")
    return registration


class ParticipantConsent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    study_id: str
    consented_at: datetime
    revoked_at: datetime | None = None
    approved_backends: list[str]
    persona: str = Field(default="", max_length=2000)

    @field_validator("consented_at", "revoked_at")
    @classmethod
    def aware_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("Consent timestamps must include a timezone.")
        return value


class StudyCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    participant_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    scenario_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    domain: str = Field(min_length=1)
    context: str = Field(min_length=1, max_length=12000)
    options: list[CandidateOption] = Field(min_length=2, max_length=20)
    chosen_action: str
    recorded_at: datetime
    rationale: str = Field(default="", max_length=4000)
    out_of_distribution: bool = False
    risk_tier: RiskTier = RiskTier.LOW
    retest_of: str | None = None

    @field_validator("recorded_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Case timestamps must include a timezone.")
        return value

    def option_ids(self) -> set[str]:
        return {option.option_id for option in self.options}

    def query(self) -> QueryContext:
        return QueryContext(
            query_id=f"{self.participant_id}:{self.case_id}", subject_id=self.participant_id,
            caller_agent="study", domain=self.domain, context=self.context,
            candidate_options=self.options, risk_tier=self.risk_tier,
        )

    def as_decision(self) -> DecisionLog:
        return DecisionLog(
            decision_id=f"{self.participant_id}:{self.case_id}", subject_id=self.participant_id,
            domain=self.domain, context=self.context, created_at=self.recorded_at.isoformat(),
            options_considered=self.options, chosen_action=self.chosen_action,
            rationale=self.rationale, risk_tier=self.risk_tier,
        )


class StudyDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["synthetic", "consented"]
    participants: list[ParticipantConsent] = Field(min_length=1)
    cases: list[StudyCase] = Field(min_length=1)


class PreparedStudy(BaseModel):
    train: list[StudyCase]
    calibration: list[StudyCase]
    test: list[StudyCase]
    retests: list[tuple[StudyCase, StudyCase]]
    participants: dict[str, ParticipantConsent]


def prepare_study(registration: Registration, dataset: StudyDataset) -> PreparedStudy:
    plan = registration.plan
    if dataset.source == "synthetic" and plan.purpose != "pilot":
        raise ValueError("Synthetic data cannot satisfy a confirmatory study.")
    participants = {item.participant_id: item for item in dataset.participants}
    if len(participants) != len(dataset.participants) or len(participants) < plan.minimum_participants:
        raise ValueError("Duplicate participants or insufficient consented participants.")
    for record in participants.values():
        if (record.study_id != plan.study_id or record.revoked_at is not None
                or record.consented_at > datetime.now(timezone.utc)):
            raise ValueError("Every participant must have current consent for this study.")
        if not set(plan.backends).issubset(record.approved_backends):
            raise ValueError("Participant consent must cover every registered backend.")

    by_id = {}
    primary: dict[tuple[str, str], list[StudyCase]] = defaultdict(list)
    repeated: list[StudyCase] = []
    for case in dataset.cases:
        identifier = (case.participant_id, case.case_id)
        consent = participants.get(case.participant_id)
        if consent is None or case.domain not in plan.domains:
            raise ValueError("A case is outside the registered participant or domain scope.")
        if identifier in by_id or len(case.option_ids()) != len(case.options):
            raise ValueError("Case and option identifiers must be unique.")
        if case.chosen_action not in case.option_ids():
            raise ValueError("A recorded choice does not match any candidate option.")
        by_id[identifier] = case
        if case.retest_of is None:
            primary[(case.participant_id, case.domain)].append(case)
        else:
            repeated.append(case)

    train, calibration, test = [], [], []
    expected_groups = {(person, domain) for person in participants for domain in plan.domains}
    if set(primary) != expected_groups:
        raise ValueError("Every included participant must have primary cases in every registered domain.")
    for group in sorted(primary):
        cases = sorted(primary[group], key=lambda case: (case.recorded_at, case.case_id))
        if len({case.scenario_id for case in cases}) != len(cases):
            raise ValueError("Repeated scenarios must be explicitly marked as retests.")
        ordinary = [case for case in cases if not case.out_of_distribution]
        n_test = max(plan.minimum_test_cases, math.ceil(len(ordinary) * plan.test_fraction))
        n_calibration = max(plan.minimum_calibration_cases, math.ceil(len(ordinary) * plan.calibration_fraction))
        if len(ordinary) - n_test - n_calibration < plan.minimum_training_cases:
            raise ValueError("Insufficient cases for the registered temporal split.")
        split_train = ordinary[:-(n_test + n_calibration)]
        split_calibration = ordinary[-(n_test + n_calibration):-n_test]
        split_test = ordinary[-n_test:]
        if (split_train[-1].recorded_at >= split_calibration[0].recorded_at
                or split_calibration[-1].recorded_at >= split_test[0].recorded_at):
            raise ValueError("Training, calibration, and evaluation must have disjoint timestamps.")
        if any(case.recorded_at < split_test[0].recorded_at for case in cases if case.out_of_distribution):
            raise ValueError("OOD cases must be collected in the held-out period.")
        train.extend(split_train)
        calibration.extend(split_calibration)
        test.extend([*split_test, *(case for case in cases if case.out_of_distribution)])

    if plan.purpose == "confirmatory" and any(
        case.recorded_at <= max(registration.registered_at, participants[case.participant_id].consented_at)
        for case in test
    ):
        raise ValueError("Confirmatory test responses must be collected after registration and consent.")
    test_ids = {(case.participant_id, case.case_id) for case in test}
    retests = []
    seen_pairs = set()
    for second in repeated:
        source_key = (second.participant_id, second.retest_of)
        first = by_id.get(source_key)
        if (source_key not in test_ids or first is None or source_key in seen_pairs
                or first.domain != second.domain or first.scenario_id != second.scenario_id
                or first.context != second.context or first.option_ids() != second.option_ids()
                or first.risk_tier != second.risk_tier
                or {option.option_id: (option.label, option.description) for option in first.options}
                != {option.option_id: (option.label, option.description) for option in second.options}
                or (second.recorded_at - first.recorded_at).total_seconds() < plan.minimum_retest_delay_hours * 3600):
            raise ValueError("Each retest must independently repeat one held-out case after the registered delay.")
        seen_pairs.add(source_key)
        retests.append((first, second))
    counts = Counter((first.participant_id, first.domain) for first, _ in retests)
    if any(counts[group] < plan.minimum_retests_per_domain for group in expected_groups):
        raise ValueError("The dataset lacks the preregistered retest count.")
    return PreparedStudy(train=train, calibration=calibration, test=test,
                         retests=retests, participants=participants)
