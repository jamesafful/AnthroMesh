from __future__ import annotations

from anthromesh.config import AnthroMeshConfig
from .base import BaseModelAdapter
from .cloud import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from .heuristic import HeuristicAdapter
from .local import OllamaAdapter, VLLMAdapter


def list_supported_adapters() -> list[str]:
    return ["heuristic", "ollama", "vllm", "openai", "anthropic", "gemini"]


def create_adapter(config: AnthroMeshConfig | None = None) -> BaseModelAdapter:
    cfg = config or AnthroMeshConfig()
    name = cfg.adapter_name.lower().strip()

    if name == "heuristic":
        return HeuristicAdapter(model=cfg.model_name or "deterministic-v1")
    if name == "ollama":
        return OllamaAdapter(model=cfg.model_name, base_url=cfg.ollama_base_url)
    if name == "vllm":
        return VLLMAdapter(model=cfg.model_name, base_url=cfg.vllm_base_url, api_key=cfg.vllm_api_key)
    if name == "openai":
        return OpenAIAdapter(model=cfg.model_name, api_key=cfg.openai_api_key)
    if name == "anthropic":
        return AnthropicAdapter(model=cfg.model_name, api_key=cfg.anthropic_api_key)
    if name == "gemini":
        return GeminiAdapter(model=cfg.model_name, api_key=cfg.gemini_api_key)

    raise ValueError(f"Unsupported adapter '{name}'. Choose from: {', '.join(list_supported_adapters())}")
