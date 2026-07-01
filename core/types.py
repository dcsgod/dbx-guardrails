"""Shared pydantic models for the guardrail pipeline.

All check modules (injection_gate, pii_guard, scope_guard, harm_guard) speak
these types; policy_engine consumes them and is the sole place block/mask/
allow decisions get made.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Turn = dict  # {"role": "user" | "assistant", "content": str}


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    confidence: float


class PIIVerdict(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    masked_text: str
    masked_entity_ids: list[str] = Field(default_factory=list)


class ScopeVerdict(BaseModel):
    in_scope: bool
    confidence: float
    reason: str | None = None
    escalated: bool = False


class InjectionVerdict(BaseModel):
    flagged: bool
    confidence: float
    reason: str | None = None


class HarmVerdict(BaseModel):
    """Raw scores only. Whether this blocks is decided exclusively by
    policy_engine.decide against project_config.harm_severity_block_threshold
    -- this type intentionally has no `blocked` field."""

    category_scores: dict[str, float] = Field(default_factory=dict)
    severity: int = 0
    escalated: bool = False


class CheckResult(BaseModel):
    injection: InjectionVerdict
    pii: PIIVerdict
    scope: ScopeVerdict
    harm: HarmVerdict
    latency_ms: dict[str, float] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    action: Literal["allow", "mask", "block"]
    masked_text: str | None = None
    reasons: list[str] = Field(default_factory=list)
