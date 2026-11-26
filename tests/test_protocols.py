from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.benchmarks.datasets import build_cohort_benchmark_suite
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.protocols.a2a_server import A2AService
from anthromesh.protocols.mcp_server import CAPABILITIES_META_KEY, MCP_PROTOCOL_VERSION, PROTOCOL_META_KEY, MCPServer
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def test_mcp_and_a2a_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = BehavioralVault(db_path=str(Path(tmpdir) / "proto.db"))
        adapter = HeuristicAdapter()
        twin = PersonalDecisionTwin(vault=vault, policy_engine=PolicyEngine(), adapter=adapter)
        oracle = ResearchCohortOracle(vault=vault, adapter=adapter)

        profile, options, stimulus, _ = build_cohort_benchmark_suite()
        oracle.register_cohort(profile)

        mcp = MCPServer(twin=twin, oracle=oracle)
        init_resp = mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp["result"]["serverInfo"]["name"] == "anthromesh-mcp"

        discovery = mcp.handle_jsonrpc({
            "jsonrpc": "2.0", "id": 4, "method": "server/discover",
            "params": {"_meta": {PROTOCOL_META_KEY: MCP_PROTOCOL_VERSION, CAPABILITIES_META_KEY: {}}},
        })
        assert discovery["result"]["supportedVersions"][0] == MCP_PROTOCOL_VERSION
        assert discovery["result"]["cacheScope"] == "private"
        modern = {
            "jsonrpc": "2.0", "id": 5, "method": "tools/list",
            "params": {"_meta": {PROTOCOL_META_KEY: MCP_PROTOCOL_VERSION, CAPABILITIES_META_KEY: {}}},
        }
        assert mcp.handle_jsonrpc(modern)["result"]["ttlMs"] == 0
        assert mcp.handle_jsonrpc(modern, transport_version="2025-11-25")["error"]
        assert mcp.handle_jsonrpc({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

        tools_resp = mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert {"twin_query_decision", "cohort_simulate", "twin_resolve_escalation"} <= tool_names

        esc_resp = mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "twin_query_decision",
                "arguments": {
                    "subject_id": "u1",
                    "domain": "new_domain",
                    "context": "Brand new scenario with zero history",
                    "candidate_options": [
                        {"option_id": "a", "label": "A"},
                        {"option_id": "b", "label": "B"},
                    ],
                },
            },
        })
        assert esc_resp["result"]["inputRequired"] is True
        assert esc_resp["result"]["ticketId"] is not None

        a2a = A2AService(twin=twin, oracle=oracle)
        card = a2a.get_agent_card()
        assert card["protocolVersion"] == "1.0"

        sim_task = a2a.send_task({
            "skill": "cohort_simulation",
            "input": {
                "cohort_id": profile.cohort_id,
                "stimulus": stimulus,
                "options": [o.model_dump() for o in options],
                "sample_size": 50,
            },
        })
        assert sim_task["status"]["state"] == "completed"

        twin_task = a2a.send_task({
            "skill": "twin_decision",
            "input": {
                "subject_id": "u1",
                "domain": "unseen_ops",
                "context": "Approve risky migration",
                "risk_tier": "high",
                "candidate_options": [
                    {"option_id": "wait", "label": "Wait"},
                    {"option_id": "proceed", "label": "Proceed"},
                ],
            },
        })
        assert twin_task["status"]["state"] == "input-required"
        reopened_vault = BehavioralVault(db_path=vault.db_path)
        reopened_twin = PersonalDecisionTwin(
            vault=reopened_vault, policy_engine=PolicyEngine(), adapter=adapter
        )
        reopened_oracle = ResearchCohortOracle(vault=reopened_vault, adapter=adapter)
        restarted = A2AService(twin=reopened_twin, oracle=reopened_oracle)
        assert restarted.get_task(twin_task["id"]) == twin_task
        assert restarted.get_task(sim_task["id"]) == sim_task
        assert restarted.send_task({"id": twin_task["id"]}) == twin_task

        resolved = restarted.resolve_task(
            task_id=twin_task["id"],
            human_chosen_action="wait",
            human_rationale="Never migrate during peak hours.",
        )
        assert resolved["status"]["state"] == "completed"
        assert a2a.get_task(twin_task["id"]) == resolved


def test_a2a_agent_card_routes(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    vault = BehavioralVault(str(tmp_path / "card.db"))
    adapter = HeuristicAdapter()
    twin = PersonalDecisionTwin(vault=vault, policy_engine=PolicyEngine(), adapter=adapter)
    oracle = ResearchCohortOracle(vault=vault, adapter=adapter)
    service = A2AService(twin=twin, oracle=oracle)
    with TestClient(service.create_fastapi_app()) as client:
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        assert response.json()["supportedInterfaces"][0]["protocolVersion"] == "1.0"
        assert client.get("/.well-known/agent.json").json() == service.get_agent_card()

        def rpc(method: str, params: dict) -> dict:
            response = client.post("/a2a", json={
                "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
            })
            assert response.status_code == 200
            return response.json()

        sent = rpc("SendMessage", {"message": {
            "role": "ROLE_USER", "messageId": "message-1", "parts": [{"data": {
                "skill": "twin_decision", "input": {
                    "subject_id": "user", "domain": "calendar", "context": "Cancel a meeting?",
                    "candidate_options": [
                        {"option_id": "cancel", "label": "Cancel"},
                        {"option_id": "keep", "label": "Keep"},
                    ],
                },
            }}],
        }})
        task = sent["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        assert rpc("GetTask", {"id": task["id"]})["result"] == task
        listed = rpc("ListTasks", {
            "pageSize": 1, "status": "TASK_STATE_INPUT_REQUIRED",
        })["result"]
        assert [item["id"] for item in listed["tasks"]] == [task["id"]]
        canceled = rpc("CancelTask", {"id": task["id"]})["result"]
        assert canceled["status"]["state"] == "TASK_STATE_CANCELED"
        assert rpc("SubscribeToTask", {"id": task["id"]})["error"]["code"] == -32004


def test_mcp_continuation_requires_local_resolution(tmp_path: Path) -> None:
    vault = BehavioralVault(str(tmp_path / "mcp.db"))
    adapter = HeuristicAdapter()
    twin = PersonalDecisionTwin(vault=vault, policy_engine=PolicyEngine(), adapter=adapter)
    oracle = ResearchCohortOracle(vault=vault, adapter=adapter)
    mcp = MCPServer(twin=twin, oracle=oracle)
    arguments = {
        "subject_id": "user", "domain": "calendar", "context": "Move a meeting?",
        "candidate_options": [
            {"option_id": "move", "label": "Move"},
            {"option_id": "keep", "label": "Keep"},
        ],
    }
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "twin_query_decision", "arguments": arguments,
        "_meta": {PROTOCOL_META_KEY: MCP_PROTOCOL_VERSION, CAPABILITIES_META_KEY: {}},
    }}
    pending = mcp.handle_jsonrpc(request, allow_owner_tools=False)["result"]
    assert pending["resultType"] == "input_required"
    request["params"]["requestState"] = pending["requestState"]
    assert mcp.handle_jsonrpc(request, allow_owner_tools=False)["result"] == pending
    ticket_id = pending["_meta"]["org.anthromesh/escalationTicket"]
    twin.resolve_escalation(ticket_id, "keep", "Avoid moving this meeting.")
    complete = mcp.handle_jsonrpc(request, allow_owner_tools=False)["result"]
    assert complete["structuredContent"]["chosen_action"] == "keep"
    arguments["subject_id"] = "different-user"
    assert mcp.handle_jsonrpc(request, allow_owner_tools=False)["error"]
