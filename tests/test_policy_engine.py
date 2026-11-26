from __future__ import annotations

from pathlib import Path

from anthromesh.models import CandidateOption, PolicyEffect, PolicyRule, QueryContext, RiskTier
from anthromesh.vault.policy_engine import PolicyEngine


def test_policy_engine_deny_escalate_allow() -> None:
    policies_path = Path(__file__).resolve().parents[1] / "examples" / "sample_policies.json"
    engine = PolicyEngine.from_file(policies_path)
    opts = [
        CandidateOption(option_id="yes", label="Yes"),
        CandidateOption(option_id="no", label="No"),
    ]

    q_deny = QueryContext(
        subject_id="u1",
        caller_agent="random_bot",
        domain="finance",
        context="Check investment allocation",
        candidate_options=opts,
    )
    assert engine.evaluate(q_deny).effect == PolicyEffect.DENY

    q_kw = QueryContext(
        subject_id="u1",
        caller_agent="trusted_finance_agent",
        domain="finance",
        context="Execute wire transfer to vendor",
        candidate_options=opts,
    )
    assert engine.evaluate(q_kw).effect == PolicyEffect.ESCALATE

    q_money = QueryContext(
        subject_id="u1",
        caller_agent="shopping_bot",
        domain="procurement",
        context="Order replacement monitor",
        monetary_value=450.0,
        candidate_options=opts,
    )
    assert engine.evaluate(q_money).effect == PolicyEffect.ESCALATE

    q_ok = QueryContext(
        subject_id="u1",
        caller_agent="pr_bot",
        domain="code_review",
        context="Review minor documentation PR",
        risk_tier=RiskTier.LOW,
        candidate_options=opts,
    )
    assert engine.evaluate(q_ok).effect == PolicyEffect.ALLOW


def test_inherited_organization_policy_cannot_be_weakened():
    options = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]
    query = QueryContext(subject_id="owner", domain="calendar", context="Schedule the meeting",
                         candidate_options=options)
    organization = PolicyEngine([
        PolicyRule(rule_id="org-review", name="Organization review", effect=PolicyEffect.ESCALATE),
    ])
    local = PolicyEngine(organization=organization)
    assert local.evaluate(query).matched_rule_id == "org-review"
    local.add_rule(PolicyRule(rule_id="local-deny", name="Owner denial", effect=PolicyEffect.DENY))
    assert local.evaluate(query).matched_rule_id == "local-deny"
