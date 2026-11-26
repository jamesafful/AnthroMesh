"""Disposable ROS graph smoke test for the AnthroMesh preference gate."""
import json
import tempfile
from pathlib import Path
from time import monotonic

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.models import CandidateOption, DecisionLog, RiskTier
from anthromesh.ros_gate import RosPreferenceGate, create_ros_node
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        vault = BehavioralVault(str(Path(temp) / "ros-test.db"))
        options = [CandidateOption(option_id="quiet", label="Quiet"),
                   CandidateOption(option_id="normal", label="Normal")]
        for _ in range(2):
            vault.add_decision(DecisionLog(
                subject_id="owner", domain="interaction", context="Choose greeting volume",
                options_considered=options, chosen_action="quiet", rationale="Prefer quiet greetings.",
            ))
        twin = PersonalDecisionTwin(vault, PolicyEngine(), HeuristicAdapter(), fast_enabled=True)
        gate = RosPreferenceGate(
            twin.gate.evaluate_fast, subject_id="owner", domains={"interaction"}, minimum_risk=RiskTier.LOW,
        )
        rclpy.init()
        node = create_ros_node(gate, request_topic="/am_request", result_topic="/am_result", deadline_ms=250)
        test = rclpy.create_node("anthromesh_smoke_client")
        pub = test.create_publisher(String, "/am_request", 16)
        received = {}
        sub = test.create_subscription(
            String, "/am_result", lambda msg: remember(received, msg.data), 16,
        )
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(test)
        try:
            deadline = monotonic() + 5
            while (not pub.get_subscription_count() or not sub.get_publisher_count()) and monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
            assert pub.get_subscription_count() and sub.get_publisher_count()
            args = {"domain": "interaction", "context": "Choose greeting volume", "risk_tier": "low",
                    "candidate_options": [option.model_dump() for option in options]}
            pub.publish(String(data=json.dumps({**args, "request_id": "known"})))
            pub.publish(String(data=json.dumps({**args, "request_id": "unknown", "context": "New context"})))
            until = monotonic() + 5
            while len(received) < 2 and monotonic() < until:
                executor.spin_once(timeout_sec=0.02)
            assert received["known"]["disposition"] == "preference_available"
            assert received["known"]["chosen_action"] == "quiet"
            assert received["unknown"]["disposition"] == "hold_for_deliberation"
            assert vault.verify_audit_chain()
            print("ROS 2 DDS smoke passed: exact preference and abstention returned over distinct topics.")
        finally:
            executor.remove_node(node)
            executor.remove_node(test)
            node.close()
            test.destroy_node()
            rclpy.shutdown()


def remember(received: dict, payload: str) -> None:
    value = json.loads(payload)
    received[value["request_id"]] = value


if __name__ == "__main__":
    main()
