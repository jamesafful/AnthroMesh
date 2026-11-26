from __future__ import annotations

import uuid
from typing import Any
from pydantic import BaseModel, Field

from .validity import ValidityEnvelope

from .decision import CandidateOption


class CohortArchetype(BaseModel):
    archetype_id: str
    label: str
    weight: float = Field(gt=0.0, le=1.0)
    is_long_tail: bool = False
    traits: dict[str, Any] = Field(default_factory=dict)
    prior_tendencies: dict[str, float] = Field(default_factory=dict)
    sample_rationales: list[str] = Field(default_factory=list)


class CohortProfile(BaseModel):
    cohort_id: str
    name: str
    description: str = ""
    domain: str
    population_weighted: bool = False
    twin_prior_weight: float = Field(default=0.0, ge=0, le=0.25)
    archetypes: list[CohortArchetype] = Field(min_length=2)


class CohortQuery(BaseModel):
    simulation_id: str = Field(default_factory=lambda: f"sim_{uuid.uuid4().hex[:12]}")
    cohort_id: str
    caller_agent: str = "research_agent"
    stimulus: str
    options: list[CandidateOption] = Field(min_length=2)
    sample_size: int = Field(default=60, ge=5, le=2000)
    preserve_long_tail: bool = True
    use_twin_priors: bool = True
    random_seed: int | None = 42


class SimulatedResponse(BaseModel):
    respondent_id: str
    archetype_id: str
    archetype_label: str
    is_long_tail: bool
    chosen_option: str
    rationale: str
    confidence: float


class DistributionMetrics(BaseModel):
    option_distribution: dict[str, float]
    shannon_entropy: float
    normalized_entropy: float
    long_tail_representation_rate: float
    distinct_options_selected: int
    effective_sample_size: int


class CohortSimulationResult(BaseModel):
    simulation_id: str
    cohort_id: str
    stimulus: str
    sample_size: int
    adapter_used: str
    metrics: DistributionMetrics
    archetype_breakdown: dict[str, dict[str, float]]
    answer_source: str = "model_simulation"
    prior_sources: list[str] = Field(default_factory=lambda: ["cohort_profile"])
    sample_responses: list[SimulatedResponse] = Field(default_factory=list)
    validity_envelope: ValidityEnvelope = Field(default_factory=lambda: ValidityEnvelope(
        status="experimental", recommended_use="research_only",
    ))
