from __future__ import annotations

from anthromesh.models import CandidateOption, CohortArchetype, DecisionLog, QueryContext
from .base import BaseModelAdapter, StructuredPrediction, post_json


class OllamaAdapter(BaseModelAdapter):
    """Local inference adapter for Ollama (/api/chat)."""

    adapter_name = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.2},
        }
        data = post_json(f"{self.base_url}/api/chat", payload)
        return data["message"]["content"]

    def predict_decision(
        self,
        query: QueryContext,
        evidence: list[tuple[DecisionLog, float]],
    ) -> StructuredPrediction:
        system, user = self.build_decision_prompt(query, evidence)
        return self.parse_prediction(self._chat(system, user), query.candidate_options)

    def simulate_cohort_archetype(
        self,
        stimulus: str,
        options: list[CandidateOption],
        archetype: CohortArchetype,
    ) -> StructuredPrediction:
        system, user = self.build_cohort_prompt(stimulus, options, archetype)
        return self.parse_prediction(self._chat(system, user), options)


class VLLMAdapter(BaseModelAdapter):
    """Local/self-hosted inference adapter for vLLM OpenAI-compatible servers (/v1/chat/completions)."""

    adapter_name = "vllm"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = post_json(f"{self.base_url}/chat/completions", payload, headers=headers)
        return data["choices"][0]["message"]["content"]

    def predict_decision(
        self,
        query: QueryContext,
        evidence: list[tuple[DecisionLog, float]],
    ) -> StructuredPrediction:
        system, user = self.build_decision_prompt(query, evidence)
        return self.parse_prediction(self._chat(system, user), query.candidate_options)

    def simulate_cohort_archetype(
        self,
        stimulus: str,
        options: list[CandidateOption],
        archetype: CohortArchetype,
    ) -> StructuredPrediction:
        system, user = self.build_cohort_prompt(stimulus, options, archetype)
        return self.parse_prediction(self._chat(system, user), options)
