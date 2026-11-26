from __future__ import annotations

import hashlib
import json
import secrets
import sys
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.judgment.monitor import PredictionMonitor
from anthromesh.models import (
    CandidateOption,
    CohortQuery,
    DecisionLog,
    EscalationStatus,
    GateAction,
    QueryContext,
    RiskTier,
)
from anthromesh.protocols.capabilities import CapabilityTokens
from anthromesh.protocols.oauth import OAuthResource, Principal
from anthromesh.twin.engine import PersonalDecisionTwin


MCP_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-11-25"
PROTOCOL_META_KEY = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"
SERVER_INFO = {"name": "anthromesh-mcp", "version": "0.1.0"}
CACHE_HINT = {"ttlMs": 0, "cacheScope": "private"}
OWNER_TOOLS = {"twin_record_decision", "twin_resolve_escalation", "vault_audit_log"}


class MCPServer:
    """MCP JSON-RPC 2.0 server exposing Personal Decision Twin and Research Cohort Oracle tools."""

    def __init__(
        self, twin: PersonalDecisionTwin, oracle: ResearchCohortOracle,
        *, oauth: OAuthResource | None = None, capabilities: CapabilityTokens | None = None,
    ) -> None:
        self.oauth = oauth
        self.capabilities = capabilities
        self.twin = twin
        self.oracle = oracle

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "twin_query_decision",
                "description": "Consult recorded preferences or request owner review. Model estimates are experimental.",
                "inputSchema": {
                    "type": "object",
                    "required": ["subject_id", "domain", "context", "candidate_options"],
                    "properties": {
                        "subject_id": {"type": "string"},
                        "caller_agent": {"type": "string"},
                        "domain": {"type": "string"},
                        "context": {"type": "string"},
                        "risk_tier": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "monetary_value": {"type": "number"},
                        "metadata": {"type": "object"},
                        "candidate_options": {"type": "array"},
                    },
                },
            },
            {
                "name": "panel_read_result",
                "description": "Read aggregated quality-reviewed answers from the live human panel.",
                "inputSchema": {"type": "object", "required": ["topic", "question_id"]},
            },
            {
                "name": "twin_trust_map",
                "description": "Read aggregate human audit results for an authorized domain.",
                "inputSchema": {"type": "object", "required": ["subject_id", "domain"],
                                "properties": {"subject_id": {"type": "string"}, "domain": {"type": "string"}}},
            },
            {
                "name": "twin_query_fast",
                "description": "Consult recorded low-risk preferences without invoking an inference model.",
                "inputSchema": {
                    "type": "object",
                    "required": ["subject_id", "domain", "context", "candidate_options"],
                },
            },
            {
                "name": "twin_check_boundary",
                "description": "Run deterministic boundary and consent policy checks prior to taking action.",
                "inputSchema": {
                    "type": "object",
                    "required": ["subject_id", "domain", "context", "candidate_options"],
                },
            },
            {
                "name": "twin_record_decision",
                "description": "Record a verified human decision and rationale into the local Behavioral Vault.",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "twin_resolve_escalation",
                "description": "Resolve a pending human-in-the-loop escalation ticket and write back to the vault.",
                "inputSchema": {
                    "type": "object",
                    "required": ["ticket_id", "human_chosen_action", "human_rationale"],
                },
            },
            {
                "name": "cohort_simulate",
                "description": "Generate experimental cohort forecasts for comparison with empirical population baselines.",
                "inputSchema": {
                    "type": "object",
                    "required": ["cohort_id", "stimulus", "options"],
                },
            },
            {
                "name": "vault_audit_log",
                "description": "Retrieve recent tamper-logged audit records of external agent queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "default": 20}},
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "twin_query_decision":
            query = QueryContext(
                subject_id=arguments["subject_id"],
                caller_agent=arguments.get("caller_agent", "mcp_client"),
                domain=arguments["domain"],
                context=arguments["context"],
                candidate_options=[CandidateOption.model_validate(o) for o in arguments["candidate_options"]],
                risk_tier=RiskTier(arguments.get("risk_tier", "low")),
                monetary_value=arguments.get("monetary_value"),
                metadata=arguments.get("metadata", {}),
            )
            res = self.twin.query_decision(query)
            payload = res.model_dump()
            if res.action == GateAction.ESCALATE:
                return {
                    "status": "input_required",
                    "inputRequired": True,
                    "ticketId": res.escalation_ticket.ticket_id if res.escalation_ticket else None,
                    "content": [{"type": "text", "text": json.dumps(payload)}],
                    "structuredContent": payload,
                }
            return {
                "status": "ok",
                "inputRequired": False,
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
            }

        if name == "panel_read_result":
            from anthromesh.judgment.panel import HumanPanel

            report = HumanPanel(self.twin.vault).report(
                arguments["question_id"], expected_topic=arguments["topic"],
            )
            return {"content": [{"type": "text", "text": json.dumps(report)}], "structuredContent": report}

        if name == "twin_trust_map":
            self.twin.vault.assert_active_subject(arguments["subject_id"])
            domain = arguments["domain"]
            if not isinstance(domain, str) or not domain:
                raise ValueError("Specify a domain.")
            data = {"scope": "aggregate_randomized_audits", "cells": PredictionMonitor(
                self.twin.vault
            ).trust_map(domain)}
            return {"content": [{"type": "text", "text": json.dumps(data)}], "structuredContent": data}

        if name in {"twin_check_boundary", "twin_query_fast"}:
            query = QueryContext(
                subject_id=arguments["subject_id"],
                caller_agent=arguments.get("caller_agent", "mcp_client"),
                domain=arguments["domain"],
                context=arguments["context"],
                candidate_options=[CandidateOption.model_validate(o) for o in arguments["candidate_options"]],
                risk_tier=RiskTier(arguments.get("risk_tier", "low")),
                monetary_value=arguments.get("monetary_value"),
                metadata=arguments.get("metadata", {}),
            )
            if name == "twin_query_fast":
                result = self.twin.gate.evaluate_fast(query)
                verdict = result.model_dump() if result else {"action": "needs_deliberation"}
            else:
                verdict = self.twin.check_boundary(query).model_dump()
            return {"content": [{"type": "text", "text": json.dumps(verdict)}], "structuredContent": verdict}

        if name == "twin_record_decision":
            decision = DecisionLog.model_validate(arguments)
            saved = self.twin.record_decision(decision).model_dump()
            return {"content": [{"type": "text", "text": json.dumps(saved)}], "structuredContent": saved}

        if name == "twin_resolve_escalation":
            ticket, new_dec = self.twin.resolve_escalation(
                ticket_id=arguments["ticket_id"],
                human_chosen_action=arguments["human_chosen_action"],
                human_rationale=arguments["human_rationale"],
            )
            out = {"ticket": ticket.model_dump(), "recorded_decision": new_dec.model_dump()}
            return {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}

        if name == "cohort_simulate":
            cquery = CohortQuery(
                cohort_id=arguments["cohort_id"],
                caller_agent=arguments.get("caller_agent", "mcp_client"),
                stimulus=arguments["stimulus"],
                options=[CandidateOption.model_validate(o) for o in arguments["options"]],
                sample_size=int(arguments.get("sample_size", 60)),
                preserve_long_tail=bool(arguments.get("preserve_long_tail", True)),
                use_twin_priors=arguments.get("use_twin_priors", True),
                random_seed=arguments.get("random_seed", 42),
            )
            sim = self.oracle.simulate(cquery).model_dump()
            return {"content": [{"type": "text", "text": json.dumps(sim)}], "structuredContent": sim}

        if name == "vault_audit_log":
            logs = self.twin.vault.list_audit_logs(limit=int(arguments.get("limit", 20)))
            return {"content": [{"type": "text", "text": json.dumps(logs)}], "structuredContent": {"logs": logs}}

        raise ValueError(f"Unknown MCP tool: {name}")

    @staticmethod
    @contextmanager
    def _trace_tool(name: str, metadata: dict[str, Any]):
        try:
            from opentelemetry import propagate, trace
        except ImportError:
            yield
            return
        carrier = {
            key: metadata[key]
            for key in ("traceparent", "tracestate", "baggage")
            if isinstance(metadata.get(key), str) and len(metadata[key]) <= 8192
        }
        parent = propagate.extract(carrier)
        with trace.get_tracer("anthromesh.mcp").start_as_current_span(
            "MCP tools/call", context=parent, kind=trace.SpanKind.SERVER,
            attributes={"mcp.tool.name": name},
        ):
            yield

    def _modern_twin_call(self, arguments: dict[str, Any], state: str | None) -> dict[str, Any]:
        digest = hashlib.sha256(
            json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        now = int(time.time())
        if state is None:
            result = self.call_tool("twin_query_decision", arguments)
            ticket_id = result.pop("ticketId", None)
            result.pop("status", None)
            result.pop("inputRequired", None)
            if not ticket_id:
                return result
            state = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
            self.twin.vault.save_mcp_continuation(token_hash, digest, ticket_id, now)
        else:
            if not isinstance(state, str) or len(state) != 43:
                raise ValueError("Invalid continuation.")
            token_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
            ticket_id = self.twin.vault.get_mcp_continuation(token_hash, digest, now)
            ticket = self.twin.vault.get_escalation(ticket_id) if ticket_id else None
            if ticket is None or ticket.status == EscalationStatus.EXPIRED:
                raise ValueError("Unknown or expired continuation.")
            if ticket.status == EscalationStatus.RESOLVED:
                result = {
                    "status": "resolved",
                    "answer_source": "human_owner",
                    "ticketId": ticket.ticket_id,
                    "chosen_action": ticket.human_chosen_action,
                    "rationale": ticket.human_rationale,
                }
                return {"content": [{"type": "text", "text": json.dumps(result)}],
                        "structuredContent": result}
        return {"resultType": "input_required", "requestState": state,
                "_meta": {"org.anthromesh/escalationTicket": ticket_id}}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def handle_jsonrpc(
        self, request: Any, *, transport_version: str | None = None,
        allow_owner_tools: bool = True, principal: Principal | None = None
    ) -> dict[str, Any] | None:
        if not isinstance(request, dict):
            return self._error(None, -32600, "Invalid request")
        req_id = request.get("id")
        method = request.get("method")
        if request.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return self._error(req_id, -32600, "Invalid request")
        if "id" not in request:
            return None
        params = request.get("params", {})
        if not isinstance(params, dict):
            return self._error(req_id, -32602, "Invalid params")
        meta = params.get("_meta", {})
        if not isinstance(meta, dict):
            return self._error(req_id, -32602, "Invalid metadata")
        body_version = meta.get(PROTOCOL_META_KEY)
        if body_version is not None and transport_version not in (None, body_version):
            return self._error(req_id, -32020, "Protocol versions disagree")
        version = transport_version or body_version
        if version not in (None, MCP_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION):
            error = self._error(req_id, -32022, "Unsupported protocol version")
            error["error"]["data"] = {
                "requested": version, "supported": [MCP_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION],
            }
            return error
        modern = version == MCP_PROTOCOL_VERSION or method == "server/discover"
        if modern and (body_version != MCP_PROTOCOL_VERSION
                       or not isinstance(meta.get(CAPABILITIES_META_KEY), dict)):
            return self._error(req_id, -32602, "Required client metadata is missing")
        capabilities = {"tools": {"listChanged": False}}
        try:
            if method == "server/discover":
                result = {"supportedVersions": [MCP_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION],
                          "capabilities": capabilities, **CACHE_HINT}
            elif method == "initialize" and not modern:
                result = {"protocolVersion": LEGACY_PROTOCOL_VERSION,
                          "serverInfo": SERVER_INFO, "capabilities": capabilities}
            elif method == "tools/list":
                tools = sorted(self.list_tools(), key=lambda tool: tool["name"])
                if not allow_owner_tools:
                    tools = [tool for tool in tools if tool["name"] not in OWNER_TOOLS]
                result = {"tools": tools, **(CACHE_HINT if modern else {})}
            elif method == "tools/call":
                arguments = params.get("arguments", {})
                if not isinstance(arguments, dict):
                    return self._error(req_id, -32602, "Invalid tool arguments")
                name = params["name"]
                if not isinstance(name, str):
                    return self._error(req_id, -32602, "Invalid tool name")
                if principal is not None:
                    if not principal.permits(name, arguments):
                        return self._error(req_id, -32009, "Insufficient scope")
                    arguments = {**arguments, "caller_agent": principal.client_id}
                if not allow_owner_tools and name in OWNER_TOOLS:
                    return self._error(req_id, -32602, "Tool unavailable on this transport")
                with self._trace_tool(name, meta if modern else {}):
                    if modern and name == "twin_query_decision":
                        result = self._modern_twin_call(arguments, params.get("requestState"))
                    else:
                        result = self.call_tool(name, arguments)
            elif method == "ping" and not modern:
                result = {}
            else:
                return self._error(req_id, -32601, "Method not found")
        except (KeyError, TypeError, ValueError):
            return self._error(req_id, -32602, "Invalid tool arguments")
        except Exception:
            return self._error(req_id, -32603, "Internal error")
        if modern:
            result.setdefault("resultType", "complete")
            result.setdefault("_meta", {}).update({
                PROTOCOL_META_KEY: MCP_PROTOCOL_VERSION,
                "io.modelcontextprotocol/serverInfo": SERVER_INFO,
            })
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def serve_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                resp = self.handle_jsonrpc(json.loads(line))
            except ValueError:
                resp = self._error(None, -32700, "Invalid JSON")
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

    def create_http_server(
        self, host: str = "127.0.0.1", port: int = 8081, *, allow_network_bind: bool = False
    ) -> ThreadingHTTPServer:
        if host not in {"127.0.0.1", "localhost"} and not allow_network_bind:
            raise ValueError("MCP HTTP binds to loopback by default. Use an isolated network for other binds.")
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                oauth = server_ref.oauth
                if oauth is None or self.path != "/.well-known/oauth-protected-resource/mcp":
                    self.send_error(405)
                    return
                body = json.dumps(oauth.metadata).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def send_rpc(self, status: int, response: dict[str, Any], *, forbidden: bool = False) -> None:
                body = json.dumps(response).encode("utf-8")
                self.send_response(status)
                if forbidden:
                    self.send_header("WWW-Authenticate", 'Bearer error="insufficient_scope"')
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "private, no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if self.path not in ("/mcp", "/"):
                    self.send_response(404)
                    self.end_headers()
                    return
                oauth = server_ref.oauth
                bound_port = self.server.server_port
                allowed_hosts = {f"localhost:{bound_port}", f"127.0.0.1:{bound_port}"}
                if oauth:
                    allowed_hosts.add(oauth.host)
                if self.headers.get("Host") not in allowed_hosts:
                    self.send_error(403)
                    return
                if self.headers.get("Origin") not in (None, f"http://localhost:{bound_port}",
                                                     f"http://127.0.0.1:{bound_port}"):
                    self.send_error(403)
                    return
                principal = None
                tokens = server_ref.capabilities
                if oauth or tokens:
                    headers = self.headers.get_all("Authorization") or []
                    if len(headers) == 1:
                        principal = oauth.authenticate(headers[0]) if oauth else None
                        if principal is None and tokens:
                            principal = tokens.authenticate(headers[0])
                    if principal is None:
                        self.send_response(401)
                        challenge = f'Bearer resource_metadata="{oauth.metadata_url}"' if oauth else "Bearer"
                        self.send_header("WWW-Authenticate", challenge)
                        self.end_headers()
                        return
                if self.headers.get_content_type() != "application/json":
                    self.send_error(415)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4 * 1024 * 1024:
                        self.send_error(413)
                        return
                    request = json.loads(self.rfile.read(length))
                except (UnicodeError, ValueError):
                    self.send_rpc(400, server_ref._error(None, -32700, "Invalid JSON"))
                    return

                version = self.headers.get("Mcp-Protocol-Version")
                method = request.get("method") if isinstance(request, dict) else None
                params = request.get("params") if isinstance(request, dict) else None
                declared = self.headers.get_all("Mcp-Method") or []
                name = self.headers.get_all("Mcp-Name") or []
                name_fields = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri",
                               "tasks/get": "taskId", "tasks/update": "taskId", "tasks/cancel": "taskId"}
                name_field = name_fields.get(method) if isinstance(method, str) else None
                expected_name = params.get(name_field) if name_field and isinstance(params, dict) else None
                meta = params.get("_meta") if isinstance(params, dict) else None
                body_version = meta.get(PROTOCOL_META_KEY) if isinstance(meta, dict) else None
                modern = MCP_PROTOCOL_VERSION in (version, body_version) or method == "server/discover"
                if (len(self.headers.get_all("Mcp-Protocol-Version") or []) > 1
                        or ((modern or declared) and declared != [method])
                        or ((modern or name) and name != ([expected_name] if expected_name else []))
                        or (modern and body_version is not None and version != body_version)):
                    identifier = request.get("id") if isinstance(request, dict) else None
                    self.send_rpc(400, server_ref._error(identifier, -32020, "MCP headers disagree with request"))
                    return
                resp = server_ref.handle_jsonrpc(
                    request, transport_version=version, allow_owner_tools=False, principal=principal,
                )
                if resp is None:
                    self.send_response(202)
                    self.end_headers()
                    return
                code = resp.get("error", {}).get("code")
                forbidden = principal is not None and code == -32009
                statuses = {-32700: 400, -32600: 400, -32601: 404, -32602: 400,
                            -32603: 500, -32020: 400, -32021: 400, -32022: 400}
                status = 403 if forbidden else statuses.get(code, 200)
                self.send_rpc(status, resp, forbidden=forbidden)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return ThreadingHTTPServer((host, port), _Handler)

    def serve_http(
        self, host: str = "127.0.0.1", port: int = 8081, *, allow_network_bind: bool = False
    ) -> None:
        with self.create_http_server(host, port, allow_network_bind=allow_network_bind) as httpd:
            print(f"AnthroMesh MCP HTTP server listening on http://{host}:{httpd.server_port}/mcp")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
