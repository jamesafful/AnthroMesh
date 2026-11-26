from __future__ import annotations

from anthromesh.models import (
    CandidateOption,
    CohortArchetype,
    CohortProfile,
    DecisionLog,
    QueryContext,
    RiskTier,
)


def build_twin_benchmark_suite() -> tuple[list[DecisionLog], list[tuple[QueryContext, str, bool]]]:
    opts_pr = [
        CandidateOption(option_id="request_changes", label="Request Changes", description="Block merge until tests and verification pass"),
        CandidateOption(option_id="approve_with_nits", label="Approve with Nits", description="Approve PR with minor follow-up comments"),
        CandidateOption(option_id="merge_immediately", label="Merge Immediately", description="Fast-track merge without extra checks"),
    ]
    opts_cal = [
        CandidateOption(option_id="decline_async", label="Decline & Go Async", description="Convert status meeting into a written update"),
        CandidateOption(option_id="accept_meeting", label="Accept Meeting", description="Attend live synchronous call"),
        CandidateOption(option_id="propose_15min", label="Propose 15-Min Huddle", description="Shorten meeting to focused 15 minutes"),
    ]
    opts_contract = [
        CandidateOption(option_id="sign_standard", label="Sign Standard Terms", description="Accept vendor agreement as written"),
        CandidateOption(option_id="redline_ip", label="Redline IP Clause", description="Negotiate intellectual property ownership"),
        CandidateOption(option_id="reject_vendor", label="Reject Vendor", description="Walk away from vendor"),
    ]

    history = [
        DecisionLog(
            decision_id="hist_pr_01",
            subject_id="user_alex",
            domain="code_review",
            context="PR modifies numerical solver convergence loop without unit tests or regression benchmarks.",
            options_considered=opts_pr,
            chosen_action="request_changes",
            rationale="Never merge numerical or core logic changes without deterministic regression tests.",
            risk_tier=RiskTier.MEDIUM,
            tags=["verification", "testing"],
        ),
        DecisionLog(
            decision_id="hist_pr_02",
            subject_id="user_alex",
            domain="code_review",
            context="PR updates physics kernel boundary condition and skips integration test suite.",
            options_considered=opts_pr,
            chosen_action="request_changes",
            rationale="Missing integration tests on boundary conditions is a hard blocker.",
            risk_tier=RiskTier.MEDIUM,
            tags=["verification", "testing"],
        ),
        DecisionLog(
            decision_id="hist_pr_03",
            subject_id="user_alex",
            domain="code_review",
            context="PR fixes typo in README documentation and updates badge URL, all CI checks green.",
            options_considered=opts_pr,
            chosen_action="approve_with_nits",
            rationale="Low-risk doc fix with passing CI does not need blocking.",
            risk_tier=RiskTier.LOW,
            tags=["docs", "low_risk"],
        ),
        DecisionLog(
            decision_id="hist_cal_01",
            subject_id="user_alex",
            domain="calendar",
            context="Recurring 60-minute weekly status sync with no agenda during deep-work morning block.",
            options_considered=opts_cal,
            chosen_action="decline_async",
            rationale="Protect deep-work mornings; status updates belong in async docs.",
            risk_tier=RiskTier.LOW,
            tags=["deep_work", "async"],
        ),
        DecisionLog(
            decision_id="hist_cal_02",
            subject_id="user_alex",
            domain="calendar",
            context="One-hour project status check-in during morning focus hours with no decision items.",
            options_considered=opts_cal,
            chosen_action="decline_async",
            rationale="No agenda and no decisions required; request written summary instead.",
            risk_tier=RiskTier.LOW,
            tags=["deep_work", "async"],
        ),
    ]

    eval_cases: list[tuple[QueryContext, str, bool]] = [
        (
            QueryContext(
                subject_id="user_alex",
                caller_agent="pr_bot",
                domain="code_review",
                context="PR refactors solver convergence loop and removes unit tests for speed.",
                candidate_options=opts_pr,
                risk_tier=RiskTier.LOW,
            ),
            "request_changes",
            False,
        ),
        (
            QueryContext(
                subject_id="user_alex",
                caller_agent="cal_bot",
                domain="calendar",
                context="60-minute weekly status sync with no agenda scheduled during morning deep-work block.",
                candidate_options=opts_cal,
                risk_tier=RiskTier.LOW,
            ),
            "decline_async",
            False,
        ),
        (
            QueryContext(
                subject_id="user_alex",
                caller_agent="legal_bot",
                domain="legal_contracts",
                context="Vendor MSA includes perpetual derivative IP license on fine-tuned weights.",
                candidate_options=opts_contract,
                risk_tier=RiskTier.HIGH,
            ),
            "redline_ip",
            True,
        ),
        (
            QueryContext(
                subject_id="user_alex",
                caller_agent="ops_bot",
                domain="production_ops",
                context="Database schema migration requires 15-minute downtime during peak customer traffic.",
                candidate_options=[
                    CandidateOption(option_id="run_now", label="Run Now"),
                    CandidateOption(option_id="schedule_weekend", label="Schedule Weekend"),
                ],
                risk_tier=RiskTier.HIGH,
            ),
            "schedule_weekend",
            True,
        ),
    ]
    return history, eval_cases


def build_cohort_benchmark_suite() -> tuple[CohortProfile, list[CandidateOption], str, dict[str, float]]:
    options = [
        CandidateOption(option_id="adopt_now", label="Adopt Immediately", description="Deploy AI assistant broadly"),
        CandidateOption(option_id="pilot_guarded", label="Guarded Pilot", description="Run limited sandbox with audit logs"),
        CandidateOption(option_id="local_only", label="Local-Only Self-Hosted", description="Require air-gapped local models"),
        CandidateOption(option_id="reject_ai", label="Reject Outright", description="Refuse AI integration due to trust/IP"),
    ]

    profile = CohortProfile(
        cohort_id="enterprise_eng_leaders_v1",
        name="Enterprise Engineering & Research Leaders",
        description="Empirical cohort modeling technology adoption across mainstream and long-tail security-first archetypes.",
        domain="ai_adoption",
        archetypes=[
            CohortArchetype(
                archetype_id="pragmatic_evaluators",
                label="Pragmatic Evaluators",
                weight=0.45,
                is_long_tail=False,
                traits={"risk_appetite": "moderate", "priority": "ROI and auditability"},
                prior_tendencies={"pilot_guarded": 0.68, "adopt_now": 0.18, "local_only": 0.12, "reject_ai": 0.02},
                sample_rationales=["We need a bounded pilot with measurable accuracy before rolling out."],
            ),
            CohortArchetype(
                archetype_id="velocity_optimists",
                label="Velocity-First Builders",
                weight=0.25,
                is_long_tail=False,
                traits={"risk_appetite": "high", "priority": "shipping speed"},
                prior_tendencies={"adopt_now": 0.66, "pilot_guarded": 0.28, "local_only": 0.04, "reject_ai": 0.02},
                sample_rationales=["Speed wins; adopt now and iterate on guardrails in production."],
            ),
            CohortArchetype(
                archetype_id="sovereign_infra",
                label="Local-First / Air-Gapped Architects",
                weight=0.20,
                is_long_tail=True,
                traits={"risk_appetite": "low", "priority": "data sovereignty and local control"},
                prior_tendencies={"local_only": 0.74, "pilot_guarded": 0.16, "reject_ai": 0.08, "adopt_now": 0.02},
                sample_rationales=["Proprietary data cannot leave our perimeter; self-hosted local inference only."],
            ),
            CohortArchetype(
                archetype_id="compliance_skeptics",
                label="Strict Regulatory Skeptics",
                weight=0.10,
                is_long_tail=True,
                traits={"risk_appetite": "minimal", "priority": "zero liability and IP protection"},
                prior_tendencies={"reject_ai": 0.65, "local_only": 0.25, "pilot_guarded": 0.08, "adopt_now": 0.02},
                sample_rationales=["Unproven behavioral liability outweighs productivity gains right now."],
            ),
        ],
    )

    ground_truth = {
        "pilot_guarded": 0.416,
        "adopt_now": 0.252,
        "local_only": 0.237,
        "reject_ai": 0.095,
    }
    stimulus = (
        "Your organization is considering deploying an autonomous AI agent middleware that accesses "
        "internal engineering decision logs to assist with code reviews and vendor evaluations."
    )
    return profile, options, stimulus, ground_truth
