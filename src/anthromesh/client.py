from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from anthromesh.models import CandidateOption, QueryContext

_PROTOCOL_VERSION = "2026-07-28"


class DecisionClient:
    """Client for AnthroMesh request and continuation results over MCP HTTP."""

    def __init__(self, endpoint: str, *, token: str | None = None, timeout: float = 5) -> None:
        url = urlsplit(endpoint)
        local = url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"}
        if ((url.scheme != "https" and not local) or not url.hostname or url.username
                or url.password or url.fragment or url.query):
            raise ValueError("Use an HTTPS or loopback AnthroMesh MCP endpoint.")
        if not local and not token:
            raise ValueError("Remote AnthroMesh requests require a bearer token.")
        if timeout <= 0:
            raise ValueError("Timeout must be positive.")
        self.endpoint = endpoint
        self.timeout = timeout
        self._token = token

    def consult(self, query: QueryContext, *, request_state: str | None = None) -> dict[str, Any]:
        return self._call("twin_query_decision", query, request_state)

    def consult_fast(self, query: QueryContext) -> dict[str, Any]:
        return self._call("twin_query_fast", query, None)

    def _call(self, tool: str, query: QueryContext, request_state: str | None) -> dict[str, Any]:
        import httpx

        identifier = uuid4().hex
        params = {
            "name": tool,
            "_meta": {"io.modelcontextprotocol/protocolVersion": _PROTOCOL_VERSION,
                      "io.modelcontextprotocol/clientCapabilities": {}},
            "arguments": query.model_dump(mode="json", exclude={"query_id"}),
        }
        if request_state is not None:
            params["requestState"] = request_state
        headers = {
            "Mcp-Protocol-Version": _PROTOCOL_VERSION, "Mcp-Method": "tools/call",
            "Mcp-Name": tool,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        with httpx.Client(follow_redirects=False, timeout=self.timeout, trust_env=False) as connection:
            response = connection.post(self.endpoint, headers=headers, json={
                "jsonrpc": "2.0", "id": identifier, "method": "tools/call", "params": params,
            })
            response.raise_for_status()
            output = response.json()
        if output.get("id") != identifier or output.get("jsonrpc") != "2.0":
            raise ValueError("Invalid AnthroMesh response envelope.")
        result = output.get("result")
        if not isinstance(result, dict):
            raise ValueError("AnthroMesh did not return a gate result.")
        if result.get("resultType") == "input_required" and isinstance(result.get("requestState"), str):
            return {"action": "escalate", "request_state": result["requestState"]}
        if result.get("resultType") == "complete" and isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        raise ValueError("Invalid AnthroMesh decision result.")


def langgraph_node(client: DecisionClient) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a LangGraph node that reads query and writes decision in graph state."""
    def consult(state: Mapping[str, Any]) -> dict[str, Any]:
        query = QueryContext.model_validate(state["query"])
        return {"decision": client.consult(query, request_state=state.get("request_state"))}

    return consult


def openai_agents_tool(client: DecisionClient, *, subject_id: str, domain: str) -> Any:
    """Build an Agents SDK tool with an owner and domain bound by the application."""
    from agents import function_tool

    async def consult_human_preferences(
        context: str, options: list[CandidateOption],
    ) -> dict[str, Any]:
        """Consult the configured person's behavioral model for a decision."""
        query = QueryContext(
            subject_id=subject_id, domain=domain, context=context, candidate_options=options,
        )
        decision = await asyncio.to_thread(client.consult, query)
        return {key: decision[key] for key in ("action", "status", "chosen_action", "confidence")
                if key in decision}

    return function_tool(consult_human_preferences, strict_mode=False)
