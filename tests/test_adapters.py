from __future__ import annotations

from unittest.mock import patch

from anthromesh.adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    HeuristicAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    VLLMAdapter,
    create_adapter,
    list_supported_adapters,
)
from anthromesh.config import AnthroMeshConfig
from anthromesh.models import CandidateOption, QueryContext


def test_adapter_factory_and_mocked_local_cloud_calls() -> None:
    assert set(list_supported_adapters()) == {
        "heuristic", "ollama", "vllm", "openai", "anthropic", "gemini"
    }
    assert isinstance(create_adapter(AnthroMeshConfig(adapter_name="heuristic")), HeuristicAdapter)

    q = QueryContext(
        subject_id="u1",
        domain="code_review",
        context="Test context",
        candidate_options=[
            CandidateOption(option_id="opt_a", label="Option A"),
            CandidateOption(option_id="opt_b", label="Option B"),
        ],
    )
    json_reply = '{"chosen_option": "opt_b", "rationale": "Grounded choice", "self_reported_confidence": 0.85, "option_scores": {"opt_a": 0.15, "opt_b": 0.85}}'

    with patch("anthromesh.adapters.local.post_json", return_value={"message": {"content": json_reply}}):
        pred = OllamaAdapter(model="llama3.2").predict_decision(q, [])
        assert pred.chosen_option == "opt_b"
        assert pred.self_reported_confidence == 0.85

    with patch("anthromesh.adapters.local.post_json", return_value={"choices": [{"message": {"content": json_reply}}]}):
        pred = VLLMAdapter(model="llama3").predict_decision(q, [])
        assert pred.chosen_option == "opt_b"

    with patch("anthromesh.adapters.cloud.post_json", return_value={"choices": [{"message": {"content": json_reply}}]}):
        pred = OpenAIAdapter(model="gpt-4o-mini", api_key="sk-test").predict_decision(q, [])
        assert pred.chosen_option == "opt_b"

    with patch("anthromesh.adapters.cloud.post_json", return_value={"content": [{"text": json_reply}]}):
        pred = AnthropicAdapter(model="claude-3-5-sonnet-latest", api_key="ant-test").predict_decision(q, [])
        assert pred.chosen_option == "opt_b"

    with patch(
        "anthromesh.adapters.cloud.post_json",
        return_value={"candidates": [{"content": {"parts": [{"text": json_reply}]}}]},
    ):
        pred = GeminiAdapter(model="gemini-2.5-flash", api_key="gem-test").predict_decision(q, [])
        assert pred.chosen_option == "opt_b"
