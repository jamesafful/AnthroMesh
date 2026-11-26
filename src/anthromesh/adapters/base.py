from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from anthromesh.models import CandidateOption, CohortArchetype, DecisionLog, QueryContext


class StructuredPrediction(BaseModel):
    chosen_option: str
    rationale: str
    self_reported_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    option_scores: dict[str, float] = Field(default_factory=dict)
    raw_text: str = ""


_inference_deadline: ContextVar[float | None] = ContextVar("anthromesh_inference_deadline", default=None)


@contextmanager
def inference_budget(seconds: float | None):
    if seconds is None:
        yield
        return
    token = _inference_deadline.set(monotonic() + seconds)
    try:
        yield
        remaining_inference_time()
    finally:
        _inference_deadline.reset(token)


def remaining_inference_time() -> float | None:
    deadline = _inference_deadline.get()
    if deadline is None:
        return None
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Inference budget exhausted.")
    return remaining


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """HTTP JSON POST using httpx when installed, falling back to stdlib urllib."""
    remaining = remaining_inference_time()
    if remaining is not None:
        timeout = min(timeout, remaining)
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    try:
        import httpx  # type: ignore

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload, headers=req_headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError("Inference request exceeded its timeout.") from exc
    except ImportError:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("Inference request exceeded its timeout.") from exc
            raise


def extract_json_object(text: str) -> dict[str, Any]:
    """Extracts the first JSON object from model output, tolerating markdown code fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class BaseModelAdapter(ABC):
    """Pluggable inference adapter contract for local runners (Ollama/vLLM) and cloud APIs."""

    adapter_name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def predict_decision(
        self,
        query: QueryContext,
        evidence: list[tuple[DecisionLog, float]],
    ) -> StructuredPrediction:
        """Predicts which candidate option the human subject would choose given retrieved evidence."""

    @abstractmethod
    def simulate_cohort_archetype(
        self,
        stimulus: str,
        options: list[CandidateOption],
        archetype: CohortArchetype,
    ) -> StructuredPrediction:
        """Predicts how a specific cohort archetype responds to a research stimulus."""

    def predict_persona(self, query: QueryContext, persona: str) -> StructuredPrediction:
        """Predict a choice from a short consented persona and the case only."""
        system = (
            "Estimate which option the person described in the input would choose. "
            "Return a JSON object with chosen_option, rationale, self_reported_confidence, "
            "and option_scores keyed by every option ID."
        )
        user = json.dumps({
            "persona": persona, "domain": query.domain, "context": query.context,
            "candidate_options": [item.model_dump() for item in query.candidate_options],
        }, ensure_ascii=False)
        return self.parse_prediction(self._chat(system, user), query.candidate_options)

    def _chat(self, system: str, user: str) -> str:
        raise NotImplementedError("This adapter cannot run prompt-only inference.")

    @staticmethod
    def build_decision_prompt(
        query: QueryContext,
        evidence: list[tuple[DecisionLog, float]],
    ) -> tuple[str, str]:
        system = (
            "You are a provenance-grounded human behavioral prediction engine. "
            "Do NOT default to generic or average advice. Base your prediction strictly on the "
            "retrieved past human decisions, trade-offs, and rationales. "
            "Respond ONLY with a valid JSON object matching: "
            '{"chosen_option": "<option_id>", "rationale": "<concise rationale citing evidence>", '
            '"self_reported_confidence": <float 0..1>, "option_scores": {"<option_id>": <float 0..1>}}.'
        )
        ev_lines = []
        for idx, (dec, sim) in enumerate(evidence, 1):
            ev_lines.append(
                f"[{idx}] id={dec.decision_id} sim={sim:.2f} domain={dec.domain}\n"
                f"    context: {dec.context}\n"
                f"    chosen: {dec.chosen_action} | rejected: {', '.join(dec.rejected_options)}\n"
                f"    rationale: {dec.rationale}"
            )
        opt_lines = [f"- {o.option_id}: {o.label} ({o.description})" for o in query.candidate_options]
        user = (
            f"Subject: {query.subject_id}\nDomain: {query.domain}\n"
            f"Current Context: {query.context}\n\n"
            f"Candidate Options:\n" + "\n".join(opt_lines) + "\n\n"
            f"Retrieved Historical Evidence:\n"
            + ("\n".join(ev_lines) if ev_lines else "NONE (no prior evidence available)")
        )
        return system, user

    @staticmethod
    def build_cohort_prompt(
        stimulus: str,
        options: list[CandidateOption],
        archetype: CohortArchetype,
    ) -> tuple[str, str]:
        system = (
            "You are simulating a specific human cohort archetype for empirical research. "
            "Preserve authentic human variance and minority/long-tail viewpoints—do NOT collapse "
            "to polite consensus. Respond ONLY with valid JSON: "
            '{"chosen_option": "<option_id>", "rationale": "<in-character rationale>", '
            '"self_reported_confidence": <float 0..1>, "option_scores": {"<option_id>": <float 0..1>}}.'
        )
        opt_lines = [f"- {o.option_id}: {o.label} ({o.description})" for o in options]
        user = (
            f"Archetype: {archetype.label} (long_tail={archetype.is_long_tail})\n"
            f"Traits: {json.dumps(archetype.traits)}\n"
            f"Prior Tendencies: {json.dumps(archetype.prior_tendencies)}\n"
            f"Sample Past Rationales: {json.dumps(archetype.sample_rationales)}\n\n"
            f"Research Stimulus: {stimulus}\n\n"
            f"Options:\n" + "\n".join(opt_lines)
        )
        return system, user

    @staticmethod
    def parse_prediction(raw_text: str, options: list[CandidateOption]) -> StructuredPrediction:
        valid_ids = {o.option_id for o in options}
        parsed = extract_json_object(raw_text)
        chosen = str(parsed.get("chosen_option", "")).strip()
        if chosen not in valid_ids:
            chosen = options[0].option_id
        conf = float(parsed.get("self_reported_confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        scores = {
            k: max(0.0, min(1.0, float(v)))
            for k, v in (parsed.get("option_scores") or {}).items()
            if k in valid_ids
        }
        return StructuredPrediction(
            chosen_option=chosen,
            rationale=str(parsed.get("rationale", "")).strip() or "Predicted from behavioral context.",
            self_reported_confidence=conf,
            option_scores=scores,
            raw_text=raw_text,
        )
