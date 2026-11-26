import json
import threading
import urllib.error
import urllib.request

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.protocols.capabilities import CapabilityTokens
from anthromesh.protocols.mcp_server import MCP_PROTOCOL_VERSION, MCPServer
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def test_capability_rate_limit_and_revocation(tmp_path, monkeypatch) -> None:
    from anthromesh.protocols import capabilities

    vault = BehavioralVault(str(tmp_path / "tokens.db"))
    tokens = CapabilityTokens(vault)
    current = 120
    monkeypatch.setattr(capabilities.time, "time", lambda: current)
    identifier, secret = tokens.issue("owner", "agent", ["twin:read:calendar"], 100, 1)
    assert tokens.authenticate(f"Bearer {secret}") is not None
    assert tokens.authenticate(f"Bearer {secret}") is None
    current = 180
    assert tokens.authenticate(f"Bearer {secret}") is not None
    assert vault.revoke_capability(identifier)
    current = 210
    assert tokens.authenticate(f"Bearer {secret}") is None


def test_mcp_http_checks_scope_headers_before_tool_execution(tmp_path) -> None:
    pytest.importorskip("httpx")
    vault = BehavioralVault(str(tmp_path / "scoped.db"))
    tokens = CapabilityTokens(vault)
    _, secret = tokens.issue("owner", "trusted-agent", ["twin:read:calendar"], 60, 20)
    adapter = HeuristicAdapter()
    twin = PersonalDecisionTwin(vault=vault, policy_engine=PolicyEngine(), adapter=adapter)
    oracle = ResearchCohortOracle(vault=vault, adapter=adapter)
    service = MCPServer(twin, oracle, capabilities=tokens)
    with service.create_http_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/mcp"
            arguments = {
                "subject_id": "owner", "domain": "finance", "context": "Decision",
                "caller_agent": "spoofed-agent",
                "candidate_options": [{"option_id": "yes", "label": "Yes"},
                                      {"option_id": "no", "label": "No"}],
            }
            body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "twin_query_decision", "arguments": arguments,
                               "_meta": {"io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
                                         "io.modelcontextprotocol/clientCapabilities": {}}}}
            headers = {"Content-Type": "application/json", "Mcp-Protocol-Version": MCP_PROTOCOL_VERSION,
                       "Mcp-Method": "tools/call", "Mcp-Name": "twin_query_decision"}

            def send(overrides: dict):
                request = urllib.request.Request(
                    endpoint, data=json.dumps(body).encode(), headers={**headers, **overrides},
                )
                return urllib.request.urlopen(request, timeout=2)

            with pytest.raises(urllib.error.HTTPError) as no_auth:
                send({})
            assert no_auth.value.code == 401
            authorization = {"Authorization": f"Bearer {secret}"}
            with pytest.raises(urllib.error.HTTPError) as wrong_scope:
                send(authorization)
            assert wrong_scope.value.code == 403
            arguments["domain"] = "calendar"
            with pytest.raises(urllib.error.HTTPError) as wrong_name:
                send({**authorization, "Mcp-Name": "twin_check_boundary"})
            assert wrong_name.value.code == 400
            with send(authorization) as response:
                assert json.load(response)["result"]["resultType"] == "input_required"
            assert vault.list_escalations()[0].caller_agent == "trusted-agent"

            from anthromesh.client import DecisionClient, langgraph_node

            decision_client = DecisionClient(endpoint, token=secret)
            graph_node = langgraph_node(decision_client)
            first = graph_node({"query": arguments})["decision"]
            assert first["action"] == "escalate"
            repeated = graph_node({
                "query": arguments, "request_state": first["request_state"],
            })["decision"]
            assert repeated == first
        finally:
            server.shutdown()
            thread.join(timeout=2)
