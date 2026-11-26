from __future__ import annotations

import json
from pathlib import Path

from anthromesh.models import PolicyEffect, PolicyRule, PolicyVerdict, QueryContext


class PolicyEngine:
    """Deterministic boundary, consent, and risk-tier evaluator executed prior to LLM inference."""

    def __init__(
        self, rules: list[PolicyRule] | None = None, *, organization: PolicyEngine | None = None,
    ) -> None:
        self.rules: list[PolicyRule] = sorted(rules or [], key=lambda r: r.priority)
        self.organization = organization

    @classmethod
    def from_file(
        cls, path: str | Path, *, organization: PolicyEngine | None = None,
    ) -> "PolicyEngine":
        file_path = Path(path)
        raw_text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".json":
            data = json.loads(raw_text)
        else:
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(raw_text)
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required to load YAML policy files. Install `anthromesh[protocols]` or provide JSON."
                ) from exc
        raw_rules = data.get("rules", data) if isinstance(data, dict) else data
        rules = [PolicyRule.model_validate(item) for item in raw_rules]
        return cls(rules=rules, organization=organization)

    def add_rule(self, rule: PolicyRule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def evaluate(self, query: QueryContext) -> PolicyVerdict:
        if self.organization is None:
            return self._evaluate_local(query)
        inherited = self.organization.evaluate(query)
        if inherited.effect == PolicyEffect.DENY:
            return inherited
        local = self._evaluate_local(query)
        if inherited.effect == PolicyEffect.ESCALATE and local.effect != PolicyEffect.DENY:
            return inherited
        return local

    def _evaluate_local(self, query: QueryContext) -> PolicyVerdict:
        text_blob = f"{query.context} " + " ".join(
            f"{o.option_id} {o.label} {o.description}" for o in query.candidate_options
        )
        text_lower = text_blob.lower()

        for rule in self.rules:
            if "*" not in rule.domains and query.domain not in rule.domains:
                continue

            if "*" not in rule.allowed_callers:
                if query.caller_agent not in rule.allowed_callers:
                    return PolicyVerdict(
                        effect=PolicyEffect.DENY,
                        matched_rule_id=rule.rule_id,
                        matched_rule_name=rule.name,
                        reason=f"Caller '{query.caller_agent}' is not authorized for domain '{query.domain}' under rule '{rule.name}'.",
                    )
                if not rule.blocked_keywords and rule.max_monetary_value is None and rule.max_risk_tier is None:
                    continue

            if rule.blocked_keywords:
                hit = next((kw for kw in rule.blocked_keywords if kw.lower() in text_lower), None)
                if hit:
                    return PolicyVerdict(
                        effect=rule.effect,
                        matched_rule_id=rule.rule_id,
                        matched_rule_name=rule.name,
                        reason=f"Triggered keyword '{hit}' under policy rule '{rule.name}'.",
                    )
                continue

            if rule.max_monetary_value is not None:
                if query.monetary_value is not None and query.monetary_value > rule.max_monetary_value:
                    return PolicyVerdict(
                        effect=rule.effect,
                        matched_rule_id=rule.rule_id,
                        matched_rule_name=rule.name,
                        reason=(
                            f"Monetary value {query.monetary_value} exceeds policy limit "
                            f"{rule.max_monetary_value} under rule '{rule.name}'."
                        ),
                    )
                continue

            if rule.max_risk_tier is not None:
                if query.risk_tier.severity > rule.max_risk_tier.severity:
                    return PolicyVerdict(
                        effect=rule.effect,
                        matched_rule_id=rule.rule_id,
                        matched_rule_name=rule.name,
                        reason=(
                            f"Risk tier '{query.risk_tier.value}' exceeds maximum autonomous tier "
                            f"'{rule.max_risk_tier.value}' under rule '{rule.name}'."
                        ),
                    )
                continue

            if rule.effect in (PolicyEffect.DENY, PolicyEffect.ESCALATE):
                return PolicyVerdict(
                    effect=rule.effect,
                    matched_rule_id=rule.rule_id,
                    matched_rule_name=rule.name,
                    reason=rule.description or f"Domain '{query.domain}' matched policy rule '{rule.name}'.",
                )

        return PolicyVerdict(
            effect=PolicyEffect.ALLOW,
            reason="Passed all deterministic boundary and consent policies.",
        )
