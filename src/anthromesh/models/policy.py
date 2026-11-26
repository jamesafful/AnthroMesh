from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def severity(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


class PolicyRule(BaseModel):
    rule_id: str
    name: str
    description: str = ""
    effect: PolicyEffect
    domains: list[str] = Field(default_factory=lambda: ["*"])
    allowed_callers: list[str] = Field(default_factory=lambda: ["*"])
    blocked_keywords: list[str] = Field(default_factory=list)
    max_risk_tier: RiskTier | None = None
    max_monetary_value: float | None = None
    priority: int = 100


class PolicyVerdict(BaseModel):
    effect: PolicyEffect
    matched_rule_id: str | None = None
    matched_rule_name: str | None = None
    reason: str
