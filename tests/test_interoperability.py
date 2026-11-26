import asyncio
import threading

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.protocols.a2a_server import A2AService
from anthromesh.protocols.mcp_server import MCPServer
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


_QUERY = {
    "subject_id": "smoke", "domain": "calendar", "context": "Check boundary",
    "candidate_options": [{"option_id": "yes", "label": "Yes"},
                          {"option_id": "no", "label": "No"}],
}


@pytest.fixture
def runtime(tmp_path):
    vault = BehavioralVault(str(tmp_path / "interop.db"))
    adapter = HeuristicAdapter()
    return PersonalDecisionTwin(vault, PolicyEngine(), adapter), ResearchCohortOracle(vault, adapter)


def test_official_mcp_sdk(runtime):
    mcp = pytest.importorskip("mcp")
    if not hasattr(mcp, "Client"):
        pytest.skip("Requires MCP Python SDK 2.x")

    async def smoke(url):
        async with mcp.Client(url, mode="2026-07-28") as client:
            tools = await client.list_tools()
            assert any(tool.name == "twin_check_boundary" for tool in tools.tools)
            result = await client.call_tool("twin_check_boundary", _QUERY)
            assert result.structured_content["effect"] == "allow"
            assert client.protocol_version == "2026-07-28"

    with MCPServer(*runtime).create_http_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            asyncio.run(smoke(f"http://127.0.0.1:{server.server_port}/mcp"))
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_official_a2a_sdk(runtime):
    pytest.importorskip("a2a")
    pytest.importorskip("fastapi")
    import httpx
    from a2a.client import A2ACardResolver, ClientConfig, create_client
    from a2a.types import GetTaskRequest, Message, SendMessageRequest, TaskState
    from google.protobuf.json_format import ParseDict

    async def smoke():
        app = A2AService(*runtime).create_fastapi_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1:8082") as http:
            card = await A2ACardResolver(http, str(http.base_url)).get_agent_card()
            client = await create_client(
                card, client_config=ClientConfig(streaming=False, httpx_client=http),
            )
            try:
                message = ParseDict({"messageId": "msg-1", "role": "ROLE_USER", "parts": [
                    {"data": {"skill": "twin_decision", "input": _QUERY}},
                ]}, Message())
                received = []
                async for chunk in client.send_message(SendMessageRequest(message=message)):
                    task = chunk[0] if isinstance(chunk, tuple) else chunk.task
                    received.append(task)
                    assert task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
                    found = await client.get_task(GetTaskRequest(id=task.id))
                    assert found.id == task.id
                assert len(received) == 1
            finally:
                await client.close()

    asyncio.run(smoke())
