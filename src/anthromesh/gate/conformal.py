from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field


def normalize_scores(scores: dict[str, float], options: set[str]) -> dict[str, float]:
    if (set(scores) != options or not scores
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values())):
        return {}
    total = sum(scores.values())
    return {key: value / total for key, value in scores.items()} if total else {}


class CalibrationProfile(BaseModel):
    subject_id: str
    domain: str
    adapter: str
    model: str
    alpha: float = Field(gt=0, lt=1)
    scores: list[float] = Field(min_length=1)

    def cutoff(self) -> float:
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in self.scores):
            raise ValueError("Nonconformity scores must be between zero and one.")
        index = math.ceil((len(self.scores) + 1) * (1 - self.alpha))
        return sorted(self.scores)[index - 1] if index <= len(self.scores) else math.inf


class ConformalCalibration:
    externally_managed = True
    def __init__(self, profiles: Iterable[CalibrationProfile]) -> None:
        self.cutoffs: dict[tuple[str, str, str, str], float] = {}
        for profile in profiles:
            key = (profile.subject_id, profile.domain, profile.adapter, profile.model)
            if key in self.cutoffs:
                raise ValueError("Duplicate conformal calibration profile.")
            self.cutoffs[key] = profile.cutoff()

    @classmethod
    def from_file(cls, path: Path) -> ConformalCalibration:
        records = json.loads(path.read_text(encoding="utf-8"))
        return cls(CalibrationProfile.model_validate(record) for record in records)

    def prediction_set(
        self, subject_id: str, domain: str, adapter: str, model: str,
        probabilities: dict[str, float],
    ) -> list[str]:
        cutoff = self.cutoffs.get((subject_id, domain, adapter, model))
        if cutoff is None:
            return []
        return sorted(option for option, mass in probabilities.items() if 1 - mass <= cutoff)


class VaultConformalCalibration(ConformalCalibration):
    externally_managed = False

    def __init__(self, vault) -> None:
        self.vault = vault

    def prediction_set(
        self, subject_id: str, domain: str, adapter: str, model: str,
        probabilities: dict[str, float],
    ) -> list[str]:
        record = self.vault.load_calibration_profile(subject_id, domain, adapter, model)
        if record is None:
            return []
        cutoff = CalibrationProfile.model_validate(record).cutoff()
        return sorted(option for option, mass in probabilities.items() if 1 - mass <= cutoff)
