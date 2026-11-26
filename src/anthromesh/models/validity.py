"""Machine-readable limits on how a result can be used."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ValidityEnvelope(BaseModel):
    version: Literal["1"] = "1"
    status: Literal["recorded_exact", "experimental", "owner_review", "denied", "unassessed"] = "unassessed"
    recommended_use: Literal["consult_memory", "research_only", "ask_owner", "do_not_use"] = "do_not_use"
    domain: str | None = None
    evidence_ids: tuple[str, ...] = ()
    observed_accuracy: float | None = None
    accuracy_interval: tuple[float, float] | None = None
    grants_authority: Literal[False] = False
