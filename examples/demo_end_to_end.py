from __future__ import annotations

import json
import tempfile
from pathlib import Path

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.benchmarks.runner import BenchmarkRunner
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.models import CandidateOption, CohortProfile, CohortQuery, DecisionLog
from anthromesh.protocols.a2a_server import A2AService
from anthromesh.protocols.mcp_server import MCPServer
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def run_demo() -> None:
    examples_dir = Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = BehavioralVault(db_path=str(Path(tmpdir) / "demo_vault.db"))
        policy_engine = PolicyEngine.from_file(examples_dir / "sample_policies.json")
        adapter = HeuristicAdapter()

        twin = PersonalDecisionTwin(vault=vault, policy_engine=policy_engine, adapter=adapter, confidence_threshold=0.68)
        oracle = ResearchCohortOracle(vault=vault, adapter=adapter)

        for item in json.loads((examples_dir / "sample_decisions.json").read_text(encoding="utf-8")):
            twin.record_decision(DecisionLog.model_validate(item))
        oracle.register_cohort(CohortProfile.model_validate_json((examples_dir / "sample_cohort.json").read_text(encoding="utf-8")))

        mcp = MCPServer(twin=twin, oracle=oracle)
        resp_ok = mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "twin_query_decision",
                "arguments": {
                    "subject_id": "user_alex",
                    "caller_agent": "github_pr_agent",
                    "domain": "code_review",
                    "context": "PR modifies solver convergence loop and omits unit tests.",
                    "candidate_options": [
                        {"option_id": "request_changes", "label": "Request Changes"},
                        {"option_id": "approve_with_nits", "label": "Approve with Nits"},
                    ],
                },
            },
        })
        print("=== 1. MCP Personal Twin (In-Distribution) ===")
        print(json.dumps(resp_ok["result"]["structuredContent"], indent=2))

        a2a = A2AService(twin=twin, oracle=oracle)
        task = a2a.send_task({
            "skill": "twin_decision",
            "input": {
                "subject_id": "user_alex",
                "caller_agent": "procurement_agent",
                "domain": "procurement",
                "context": "Approve $1,200 cloud GPU reservation for experiment.",
                "monetary_value": 1200.0,
                "risk_tier": "medium",
                "candidate_options": [
                    {"option_id": "approve_spend", "label": "Approve Spend"},
                    {"option_id": "defer_to_local", "label": "Run on Local Cluster"},
                ],
            },
        })
        print("\n=== 2. A2A Task Triggering Human-in-the-Loop Escalation ===")
        print(f"Task ID: {task['id']} | State: {task['status']['state']} | Ticket: {task['escalation_ticket_id']}")

        resolved = a2a.resolve_task(
            task_id=task["id"],
            human_chosen_action="defer_to_local",
            human_rationale="Use existing cluster allocation before approving cloud GPU spend.",
        )
        print(f"Resolved State: {resolved['status']['state']} | Message: {resolved['status']['message']}")

        sim = oracle.simulate(
            CohortQuery(
                cohort_id="enterprise_eng_leaders_v1",
                stimulus="Deploy autonomous AI behavioral middleware for internal engineering workflows.",
                options=[
                    CandidateOption(option_id="adopt_now", label="Adopt Immediately"),
                    CandidateOption(option_id="pilot_guarded", label="Guarded Pilot"),
                    CandidateOption(option_id="local_only", label="Local-Only Self-Hosted"),
                    CandidateOption(option_id="reject_ai", label="Reject Outright"),
                ],
                sample_size=100,
            )
        )
        print("\n=== 3. Research Cohort Oracle Simulation ===")
        print(json.dumps(sim.metrics.model_dump(), indent=2))

        print("\n=== 4. Benchmark Comparison (AnthroMesh vs Zero-Shot Baselines) ===")
        report = BenchmarkRunner().run_all()
        print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    run_demo()
