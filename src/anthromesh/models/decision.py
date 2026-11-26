from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .policy import RiskTier


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CandidateOption(BaseModel):
    option_id: str
    label: str
    description: str = ""


def preference_key(context: str, options: list[CandidateOption]) -> str:
    payload = json.dumps([context, [option.model_dump() for option in options]],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DecisionLog(BaseModel):
    decision_id: str = Field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:12]}")
    subject_id: str
    domain: str
    context: str
    options_considered: list[CandidateOption] = Field(min_length=2)
    chosen_action: str
    rejected_options: list[str] = Field(default_factory=list)
    rationale: str
    risk_tier: RiskTier = RiskTier.LOW
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    def model_post_init(self, __context: Any) -> None:
        if not self.rejected_options:
            self.rejected_options = [
                opt.option_id for opt in self.options_considered if opt.option_id != self.chosen_action
            ]


class QueryContext(BaseModel):
    query_id: str = Field(default_factory=lambda: f"qry_{uuid.uuid4().hex[:12]}")
    subject_id: str
    caller_agent: str = "unknown_agent"
    domain: str
    context: str
    candidate_options: list[CandidateOption] = Field(min_length=2)
    risk_tier: RiskTier = RiskTier.LOW
    monetary_value: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceCitation(BaseModel):
    decision_id: str
    similarity: float
    chosen_action: str
    rationale: str
    domain: str


class ProvenanceTrace(BaseModel):
    evidence_citations: list[EvidenceCitation] = Field(default_factory=list)
    evidence_coverage_score: float = 0.0
    agreement_score: float = 0.0
    calibration_confidence: float = 0.0
    conformal_prediction_set: list[str] | None = None
    stale_evidence_ids: list[str] = Field(default_factory=list)
    adapter_used: str = "heuristic"
    model_used: str = "deterministic"
    inference_tier: str = "deliberative"
    answer_source: str = "model_simulation"
