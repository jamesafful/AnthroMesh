from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AnthroMeshConfig:
    """Central runtime configuration loaded from environment variables or explicit overrides."""

    db_path: str = field(default_factory=lambda: os.getenv("ANTHROMESH_DB_PATH", "./anthromesh_vault.db"))
    postgres_dsn: str | None = field(default_factory=lambda: os.getenv("ANTHROMESH_POSTGRES_DSN"), repr=False)
    postgres_schema: str = field(default_factory=lambda: os.getenv("ANTHROMESH_POSTGRES_SCHEMA", "anthromesh"))
    encryption_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROMESH_VAULT_KEY"), repr=False
    )
    vector_index: bool = field(default_factory=lambda: os.getenv("ANTHROMESH_VECTOR_INDEX") == "1")
    embedding_model: str | None = field(default_factory=lambda: os.getenv("ANTHROMESH_EMBEDDING_MODEL"))
    managed_calibration: bool = field(default_factory=lambda: os.getenv("ANTHROMESH_MANAGED_CALIBRATION") == "1")
    calibration_path: str | None = field(default_factory=lambda: os.getenv("ANTHROMESH_CALIBRATION_PATH"))
    policy_path: str | None = field(default_factory=lambda: os.getenv("ANTHROMESH_POLICY_PATH"))
    organization_policy_path: str | None = field(
        default_factory=lambda: os.getenv("ANTHROMESH_ORGANIZATION_POLICY_PATH"),
    )
    deliberation_seconds: float = field(
        default_factory=lambda: float(os.getenv("ANTHROMESH_DELIBERATION_SECONDS", "5"))
    )
    monitor_enabled: bool = field(default_factory=lambda: os.getenv("ANTHROMESH_JUDGMENT_MONITOR") == "1")
    memory_only: bool = field(default_factory=lambda: os.getenv("ANTHROMESH_MEMORY_ONLY") == "1")
    fast_enabled: bool = field(default_factory=lambda: os.getenv("ANTHROMESH_FAST_TIER") == "1")
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("ANTHROMESH_CONFIDENCE_THRESHOLD", "0.72"))
    )
    min_evidence_count: int = field(
        default_factory=lambda: int(os.getenv("ANTHROMESH_MIN_EVIDENCE_COUNT", "2"))
    )
    adapter_name: str = field(default_factory=lambda: os.getenv("ANTHROMESH_ADAPTER", "heuristic"))
    model_name: str = field(default_factory=lambda: os.getenv("ANTHROMESH_MODEL", "llama3.2"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    vllm_base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    vllm_api_key: str = field(default_factory=lambda: os.getenv("VLLM_API_KEY", "EMPTY"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    mcp_host: str = field(default_factory=lambda: os.getenv("ANTHROMESH_MCP_HOST", "127.0.0.1"))
    mcp_port: int = field(default_factory=lambda: int(os.getenv("ANTHROMESH_MCP_PORT", "8081")))
    a2a_host: str = field(default_factory=lambda: os.getenv("ANTHROMESH_A2A_HOST", "127.0.0.1"))
    a2a_port: int = field(default_factory=lambda: int(os.getenv("ANTHROMESH_A2A_PORT", "8082")))
