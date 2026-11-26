from __future__ import annotations

from anthromesh.models import CandidateOption, CohortArchetype, DecisionLog, QueryContext
from .base import BaseModelAdapter, StructuredPrediction, post_json


class OpenAIAdapter(BaseModelAdapter):
    """Cloud inference adapter for OpenAI Chat Completions API."""

    adapter_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        super().__init__(model=model)
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIAdapter.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
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


class AnthropicAdapter(BaseModelAdapter):
    """Cloud inference adapter for Anthropic Messages API."""

    adapter_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        super().__init__(model=model)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicAdapter.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "temperature": 0.2,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = post_json(f"{self.base_url}/messages", payload, headers=headers)
        return data["content"][0]["text"]

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


class GeminiAdapter(BaseModelAdapter):
    """Cloud inference adapter for Google Gemini generateContent API."""

    adapter_name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        super().__init__(model=model)
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiAdapter.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _chat(self, system: str, user: str) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        data = post_json(url, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"]

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
