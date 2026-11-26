from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from time import monotonic
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from anthromesh.models import CandidateOption, GateAction, GateDecision, QueryContext, RiskTier


class RosGateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    domain: str = Field(min_length=1, max_length=80)
    context: str = Field(min_length=1, max_length=4096)
    candidate_options: list[CandidateOption] = Field(min_length=2, max_length=16)
    risk_tier: RiskTier
    monetary_value: float | None = None


class RosPreferenceGate:
    def __init__(
        self, evaluate: Callable[[QueryContext], GateDecision | None],
        *, subject_id: str, domains: set[str], minimum_risk: RiskTier,
    ) -> None:
        if not subject_id or not domains:
            raise ValueError("Configure a subject and at least one permitted ROS domain.")
        self.evaluate = evaluate
        self.subject_id = subject_id
        self.domains = domains
        self.minimum_risk = minimum_risk

    def consult(self, request: RosGateRequest) -> dict:
        outcome = {"request_id": request.request_id, "disposition": "hold_for_deliberation"}
        if request.domain not in self.domains:
            return {**outcome, "disposition": "hold", "reason": "domain_out_of_scope"}
        risk = max(request.risk_tier, self.minimum_risk, key=lambda value: value.severity)
        query = QueryContext(
            subject_id=self.subject_id, caller_agent="ros2_preference_gate", domain=request.domain,
            context=request.context, candidate_options=request.candidate_options,
            risk_tier=risk, monetary_value=request.monetary_value,
        )
        decision = self.evaluate(query)
        if decision is None:
            return outcome
        if decision.action == GateAction.DENIED:
            return {**outcome, "disposition": "hold", "reason": "policy_denied"}
        if decision.action != GateAction.APPROVED or decision.provenance.inference_tier != "fast":
            return outcome
        if decision.chosen_action not in {option.option_id for option in request.candidate_options}:
            return {**outcome, "disposition": "hold", "reason": "invalid_preference"}
        return {**outcome, "disposition": "preference_available",
                "chosen_action": decision.chosen_action, "query_id": decision.query_id}


def create_ros_node(
    gate: RosPreferenceGate, *, request_topic: str, result_topic: str, deadline_ms: int = 50,
):
    from rclpy.node import Node
    from std_msgs.msg import String

    if not request_topic or request_topic == result_topic or not 5 <= deadline_ms <= 5000:
        raise ValueError("Configure distinct ROS topics and a deadline from 5 ms to 5 seconds.")

    class GateNode(Node):
        def __init__(self):
            super().__init__("anthromesh_preference_gate")
            self.publisher = self.create_publisher(String, result_topic, 16)
            self.subscription = self.create_subscription(String, request_topic, self.receive, 16)
            self.timer = self.create_timer(0.005, self.publish_ready)
            self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="am-ros")
            self.pending: dict[str, tuple[Future, float]] = {}
            self.late: set[Future] = set()

        def emit(self, result: dict) -> None:
            self.publisher.publish(String(data=json.dumps(result, separators=(",", ":"))))

        def receive(self, msg: String) -> None:
            arrived = monotonic()
            if len(msg.data) > 32768:
                self.emit({"request_id": None, "disposition": "hold", "reason": "invalid_request"})
                return
            try:
                request = RosGateRequest.model_validate_json(msg.data)
            except ValueError:
                self.emit({"request_id": None, "disposition": "hold", "reason": "invalid_request"})
                return
            if request.request_id in self.pending:
                return
            if len(self.pending) + len(self.late) >= 8:
                self.emit({"request_id": request.request_id, "disposition": "hold", "reason": "busy"})
                return
            completion = self.pool.submit(gate.consult, request)
            self.pending[request.request_id] = (completion, arrived + deadline_ms / 1000)

        def publish_ready(self) -> None:
            self.late = {future for future in self.late if not future.done()}
            for identifier, (future, deadline) in tuple(self.pending.items()):
                if monotonic() >= deadline:
                    if not future.cancel() and not future.done():
                        self.late.add(future)
                    result = {"request_id": identifier, "disposition": "hold", "reason": "deadline"}
                elif not future.done():
                    continue
                else:
                    try:
                        result = future.result()
                    except Exception:
                        result = {"request_id": identifier, "disposition": "hold", "reason": "unavailable"}
                del self.pending[identifier]
                self.emit(result)

        def close(self) -> None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.destroy_node()

    return GateNode()


def serve_ros(
    gate: RosPreferenceGate, *, request_topic: str, result_topic: str, deadline_ms: int,
) -> None:
    import rclpy

    rclpy.init(args=[])
    node = None
    try:
        node = create_ros_node(
            gate, request_topic=request_topic, result_topic=result_topic, deadline_ms=deadline_ms,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
        if rclpy.ok():
            rclpy.shutdown()
