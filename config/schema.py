"""Pydantic schema for a per-project guardrail configuration.

Rows of this shape live in the Unity Catalog table created by
config/ddl/configs_table.sql (guardrails.governance.project_configs).
Onboarding a new project is an insert into that table, not a code change.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    project_id: str
    scope_definition: str  # free text description of allowed domain
    scope_confidence_band: tuple[float, float] = (
        0.35,
        0.65,
    )  # inside this band -> escalate to LLM judge
    scope_context_turns: int = 3  # recent turns considered by scope_guard for follow-up continuity
    pii_necessity_threshold: float = 0.5  # below this -> mask
    pii_custom_recognizers: list[dict] = Field(default_factory=list)
    harm_severity_block_threshold: int = 4  # e.g. block at severity >= 4 (Azure 0/2/4/6 scale)
    harm_categories_enabled: list[str] = Field(
        default_factory=lambda: ["hate", "violence", "self_harm", "sexual"]
    )
    injection_gate_enabled: bool = True
    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
