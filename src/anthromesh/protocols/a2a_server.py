from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.models import (
    CandidateOption,
    CohortQuery,
    DecisionLog,
    EscalationTicket,
    GateAction,
    QueryContext,
    RiskTier,
)
from anthromesh.models.decision import utc_now_iso
from anthromesh.twin.engine import PersonalDecisionTwin


def build_agent_card(base_url: str = "http://127.0.0.1:8082", *, modern: bool = False) -> dict[str, Any]:
    card = {
        "name": "AnthroMesh Human Behavioral Middleware",
        "description": (
            "Recorded human preferences and experimental cohort evaluation "
            "with provenance and owner escalation."
        ),
        "version": "0.1.0",
        "protocolVersion": "1.0",
        "url": base_url.rstrip("/"),
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "humanInTheLoopEscalation": True,
        },
        "skills": [
            {
                "id": "twin_decision",
                "name": "Personal Decision Twin Judgment",
                "description": "Consults recorded preferences and may request owner review. Model estimates are experimental.",
            },
            {
                "id": "cohort_simulation",
                "name": "Stratified Research Cohort Simulation",
                "description": "Generates experimental cohort forecasts for comparison with population baselines.",
            },
        ],
    }
    if modern:
        card.pop("protocolVersion")
        card.pop("url")
        card["supportedInterfaces"] = [{
            "url": f"{base_url.rstrip('/')}/a2a", "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }]
        card["defaultInputModes"] = ["application/json"]
        card["defaultOutputModes"] = ["application/json"]
        card["capabilities"] = {"streaming": True}
        for skill in card["skills"]:
            skill["tags"] = ["behavioral-simulation"]
    return card


_TERMINAL = {"completed", "failed", "canceled", "rejected"}


def _wire_task(task: dict[str, Any], *, include_artifacts: bool = True) -> dict[str, Any]:
    task_id = task["id"]
    context_id = task.get("context_id", task_id)
    source = task["status"]
    status = {
        "state": "TASK_STATE_" + source["state"].upper().replace("-", "_"),
        "timestamp": datetime.fromisoformat(source["timestamp"]).astimezone(timezone.utc)
        .isoformat().replace("+00:00", "Z"),
    }
    if source.get("message"):
        status["message"] = {
            "messageId": f"{task_id}-{source['state']}",
            "taskId": task_id,
            "contextId": context_id,
            "role": "ROLE_AGENT",
            "parts": [{"text": source["message"]}],
        }
    result = {"id": task_id, "contextId": context_id, "status": status}
    if include_artifacts:
        result["artifacts"] = [
            {"name": item["name"], "artifactId": f"{task_id}-{index}",
             "parts": [{"data": item["data"]}]}
            for index, item in enumerate(task["artifacts"])
        ]
    return result


class A2AService:
    """Stateful A2A v1.0 task coordinator supporting submitted -> working -> input-required / completed / failed."""

    def __init__(
        self,
        twin: PersonalDecisionTwin,
        oracle: ResearchCohortOracle,
        base_url: str = "http://127.0.0.1:8082",
    ) -> None:
        self.twin = twin
        self.oracle = oracle
        self.base_url = base_url
        self.vault = twin.vault

    def get_agent_card(self) -> dict[str, Any]:
        return build_agent_card(self.base_url)

    def send_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = payload.get("id") or f"task_{uuid.uuid4().hex[:12]}"
        skill = payload.get("skill", "twin_decision")
        data = payload.get("input") or {}
        now = utc_now_iso()

        task: dict[str, Any] = {
            "id": task_id,
            "skill": skill,
            "subject_id": data.get("subject_id") if skill == "twin_decision" else None,
            "context_id": payload.get("context_id") or task_id,
            "status": {"state": "working", "timestamp": now},
            "history": [{"state": "submitted", "timestamp": now}],
            "artifacts": [],
            "escalation_ticket_id": None,
        }

        if not self.vault.create_a2a_task(task):
            return self.get_task(task_id) or task

        try:
            if skill == "twin_decision":
                query = QueryContext(
                    subject_id=data["subject_id"],
                    caller_agent=data.get("caller_agent", "a2a_remote_agent"),
                    domain=data["domain"],
                    context=data["context"],
                    candidate_options=[CandidateOption.model_validate(o) for o in data["candidate_options"]],
                    risk_tier=RiskTier(data.get("risk_tier", "low")),
                    monetary_value=data.get("monetary_value"),
                    metadata=data.get("metadata", {}),
                )
                decision = self.twin.query_decision(query)
                if decision.action == GateAction.ESCALATE:
                    state = "input-required"
                    task["escalation_ticket_id"] = (
                        decision.escalation_ticket.ticket_id if decision.escalation_ticket else None
                    )
                elif decision.action == GateAction.DENIED:
                    state = "failed"
                else:
                    state = "completed"
                task["status"] = {"state": state, "timestamp": utc_now_iso(), "message": decision.rationale}
                task["history"].append({"state": state, "timestamp": utc_now_iso()})
                task["artifacts"] = [{"name": "gate_decision", "data": decision.model_dump(mode="json")}]

            elif skill == "cohort_simulation":
                cquery = CohortQuery(
                    cohort_id=data["cohort_id"],
                    caller_agent=data.get("caller_agent", "a2a_remote_agent"),
                    stimulus=data["stimulus"],
                    options=[CandidateOption.model_validate(o) for o in data["options"]],
                    sample_size=int(data.get("sample_size", 60)),
                    preserve_long_tail=bool(data.get("preserve_long_tail", True)),
                    use_twin_priors=data.get("use_twin_priors", True),
                    random_seed=data.get("random_seed", 42),
                )
                sim = self.oracle.simulate(cquery)
                task["status"] = {"state": "completed", "timestamp": utc_now_iso()}
                task["history"].append({"state": "completed", "timestamp": utc_now_iso()})
                task["artifacts"] = [{"name": "cohort_simulation", "data": sim.model_dump(mode="json")}]

            else:
                raise ValueError(f"Unsupported A2A skill '{skill}'.")

        except Exception as exc:
            task["status"] = {"state": "failed", "timestamp": utc_now_iso(), "message": str(exc)}
            task["history"].append({"state": "failed", "timestamp": utc_now_iso()})

        if not self.vault.save_a2a_task(task, expected_state="working"):
            if task["escalation_ticket_id"]:
                self.vault.expire_escalation(task["escalation_ticket_id"])
            return self.get_task(task_id) or task
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.vault.get_a2a_task(task_id)

    def resolve_task(
        self,
        task_id: str,
        human_chosen_action: str,
        human_rationale: str,
    ) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"A2A task '{task_id}' not found.")
        ticket_id = task.get("escalation_ticket_id")
        if task["status"]["state"] != "input-required" or not ticket_id:
            raise ValueError(f"A2A task '{task_id}' does not have a pending escalation ticket.")

        def complete(ticket: EscalationTicket, decision: DecisionLog) -> dict[str, Any]:
            timestamp = utc_now_iso()
            task["status"] = {
                "state": "completed", "timestamp": timestamp,
                "message": f"Resolved by human owner: {human_chosen_action}",
            }
            task["history"].append({"state": "completed", "timestamp": timestamp})
            task["artifacts"].append({
                "name": "human_resolution",
                "data": {"ticket": ticket.model_dump(), "recorded_decision": decision.model_dump()},
            })
            return task

        self.twin.resolve_escalation(
            ticket_id=ticket_id, human_chosen_action=human_chosen_action,
            human_rationale=human_rationale, complete_task=complete,
        )
        return task

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task is None:
            raise LookupError(task_id)
        state = task["status"]["state"]
        if state == "canceled":
            return task
        if state in _TERMINAL:
            raise ValueError("Terminal task cannot be canceled.")
        timestamp = utc_now_iso()
        task["status"] = {"state": "canceled", "timestamp": timestamp}
        task["history"].append({"state": "canceled", "timestamp": timestamp})
        if not self.vault.cancel_a2a_task(task, expected_state=state):
            raise ValueError("Task state changed during cancellation.")
        return task

    def send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        message = params["message"]
        if message.get("role") != "ROLE_USER" or not isinstance(message.get("messageId"), str):
            raise ValueError("A2A requires a user message with a message ID.")
        if message.get("taskId"):
            raise NotImplementedError("Resolve pending tasks through the local owner inbox.")
        parts = message["parts"]
        if len(parts) != 1 or not isinstance(parts[0].get("data"), dict):
            raise ValueError("This interface accepts a single structured data part.")
        data = parts[0]["data"]
        return self.send_task({"skill": data["skill"], "input": data["input"],
                               "context_id": message.get("contextId")})

    def handle_jsonrpc(self, request: dict[str, Any]) -> Any:
        req_id = request.get("id")
        envelope = {"jsonrpc": "2.0", "id": req_id}
        method = request.get("method")
        params = request.get("params", {})
        if (request.get("jsonrpc") != "2.0" or "id" not in request
                or not isinstance(method, str) or not isinstance(params, dict)):
            return {**envelope, "error": {"code": -32600, "message": "Invalid request"}}
        if params.get("tenant"):
            return {**envelope, "error": {"code": -32602, "message": "Unknown tenant"}}
        try:
            if method in {"CreateTaskPushNotificationConfig", "GetTaskPushNotificationConfig",
                          "ListTaskPushNotificationConfigs", "DeleteTaskPushNotificationConfig"}:
                return {**envelope, "error": {"code": -32003, "message": "Push notifications unsupported"}}
            if method == "GetExtendedAgentCard":
                raise NotImplementedError
            if method == "SendMessage":
                result = {"task": _wire_task(self.send_message(params))}
            elif method == "SendStreamingMessage":
                return self._stream_task(self.send_message(params), req_id)
            elif method == "ListTasks":
                limit = params.get("pageSize", 50)
                if type(limit) is not int or not 1 <= limit <= 100:
                    raise ValueError("Invalid page size.")
                state = params.get("status")
                if state is not None and not state.startswith("TASK_STATE_"):
                    raise ValueError("Invalid task state.")
                rows, total = self.vault.list_a2a_tasks(
                    after=params.get("pageToken") or "", limit=limit + 1,
                    context=params.get("contextId"),
                    status=state[11:].lower().replace("_", "-") if state else None,
                )
                result = {
                    "tasks": [_wire_task(task, include_artifacts=bool(params.get("includeArtifacts")))
                              for task in rows[:limit]],
                    "nextPageToken": rows[limit - 1]["id"] if len(rows) > limit else "",
                    "pageSize": limit, "totalSize": total,
                }
            elif method in {"GetTask", "CancelTask", "SubscribeToTask"}:
                task = self.get_task(params["id"])
                if task is None:
                    raise LookupError
                if method == "SubscribeToTask":
                    if task["status"]["state"] in _TERMINAL:
                        raise NotImplementedError
                    return self._stream_task(task, req_id)
                if method == "CancelTask":
                    try:
                        task = self.cancel_task(params["id"])
                    except ValueError:
                        return {**envelope, "error": {"code": -32002, "message": "Task not cancelable"}}
                result = _wire_task(task)
            else:
                return {**envelope, "error": {"code": -32601, "message": "Method not found"}}
            return {**envelope, "result": result}
        except KeyError:
            code, message = -32602, "Required parameter missing"
        except LookupError:
            code, message = -32001, "Task not found"
        except NotImplementedError:
            code, message = -32004, "Operation unsupported"
        except (AttributeError, TypeError, ValueError):
            code, message = -32602, "Invalid parameters"
        except Exception:
            code, message = -32603, "Internal error"
        return {**envelope, "error": {"code": code, "message": message}}

    def _stream_task(self, task: dict[str, Any], req_id: Any) -> Any:
        from starlette.responses import StreamingResponse

        def event(payload: dict[str, Any]) -> str:
            return "data: " + json.dumps({"jsonrpc": "2.0", "id": req_id, "result": payload}) + "\n\n"

        async def stream() -> Any:
            yield event({"task": _wire_task(task)})
            current = task
            while current["status"]["state"] not in _TERMINAL:
                await asyncio.sleep(0.5)
                updated = await asyncio.to_thread(self.get_task, task["id"])
                if updated is None:
                    return
                if updated != current:
                    formatted = _wire_task(updated)
                    for artifact in formatted["artifacts"][len(current["artifacts"]):]:
                        yield event({"artifactUpdate": {
                            "taskId": task["id"], "contextId": formatted["contextId"],
                            "artifact": artifact, "lastChunk": True,
                        }})
                    yield event({"statusUpdate": {
                        "taskId": task["id"], "contextId": formatted["contextId"],
                        "status": formatted["status"],
                    }})
                    current = updated

        return StreamingResponse(stream(), media_type="text/event-stream")

    def create_fastapi_app(self) -> Any:
        from fastapi import FastAPI, Header, HTTPException  # type: ignore

        app = FastAPI(title="AnthroMesh A2A Server", version="0.1.0")

        @app.get("/.well-known/agent-card.json")
        def modern_agent_card() -> dict[str, Any]:
            return build_agent_card(self.base_url, modern=True)

        @app.get("/.well-known/agent.json", include_in_schema=False)
        def agent_card() -> dict[str, Any]:
            return self.get_agent_card()

        @app.post("/a2a")
        def a2a_rpc(payload: dict[str, Any], a2a_version: str | None = Header(default=None)) -> Any:
            if a2a_version is not None and a2a_version != "1.0":
                return {"jsonrpc": "2.0", "id": payload.get("id"),
                        "error": {"code": -32009, "message": "A2A protocol version not supported"}}
            return self.handle_jsonrpc(payload)

        @app.post("/tasks/send")
        def tasks_send(payload: dict[str, Any]) -> dict[str, Any]:
            return self.send_task(payload)

        @app.get("/tasks/{task_id}")
        def tasks_get(task_id: str) -> dict[str, Any]:
            task = self.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return task

        @app.post("/tasks/{task_id}/resolve")
        def tasks_resolve(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            try:
                return self.resolve_task(
                    task_id=task_id,
                    human_chosen_action=payload["human_chosen_action"],
                    human_rationale=payload["human_rationale"],
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return app

    def serve_http(
        self, host: str = "127.0.0.1", port: int = 8082, *, allow_network_bind: bool = False
    ) -> None:
        if host not in {"127.0.0.1", "localhost"} and not allow_network_bind:
            raise ValueError("A2A HTTP binds to loopback by default. Use an isolated network for other binds.")
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install AnthroMesh with the protocols extra.") from exc
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        from urllib.parse import urlsplit

        app = self.create_fastapi_app()
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=[
                "127.0.0.1", "localhost", urlsplit(self.base_url).hostname,
            ],
        )
        uvicorn.run(app, host=host, port=port)
