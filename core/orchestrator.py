"""Single orchestration function shared by both integration modes
(client/async_client.py in library mode, serving/pyfunc_wrapper.py in
service mode) -- this is the "zero dependency on how it's invoked" core
referenced by design principle 1. Neither wrapper re-implements this logic.
"""
from __future__ import annotations

import asyncio
import time

from config.schema import ProjectConfig
from core import harm_guard, injection_gate, pii_guard, policy_engine, scope_guard
from core.types import CheckResult, HarmVerdict, PIIVerdict, PolicyDecision, ScopeVerdict


async def run_checks(
    text: str,
    project_config: ProjectConfig,
    task_intent: str | None = None,
    history: list[dict] | None = None,
) -> tuple[CheckResult, PolicyDecision]:
    latency_ms: dict[str, float] = {}

    t0 = time.perf_counter()
    injection_verdict = await injection_gate.check_injection(text, project_config)
    latency_ms["injection"] = (time.perf_counter() - t0) * 1000

    if injection_verdict.flagged:
        # Short-circuit: policy_engine's precedence order makes pii/scope/
        # harm irrelevant once injection blocks, so skip computing them.
        pii_verdict = PIIVerdict(entities=[], masked_text=text, masked_entity_ids=[])
        scope_verdict = ScopeVerdict(in_scope=True, confidence=1.0, reason="skipped: injection blocked")
        harm_verdict = HarmVerdict(category_scores={}, severity=0)
    else:
        async def _run_pii() -> PIIVerdict:
            start = time.perf_counter()
            entities = pii_guard.detect_entities(text, project_config)
            verdict = await pii_guard.mask(text, entities, project_config, task_intent=task_intent)
            latency_ms["pii"] = (time.perf_counter() - start) * 1000
            return verdict

        async def _run_scope() -> ScopeVerdict:
            start = time.perf_counter()
            verdict = await scope_guard.check_scope(text, project_config, history=history)
            latency_ms["scope"] = (time.perf_counter() - start) * 1000
            return verdict

        async def _run_harm() -> HarmVerdict:
            start = time.perf_counter()
            verdict = await harm_guard.check_harm(text, project_config)
            latency_ms["harm"] = (time.perf_counter() - start) * 1000
            return verdict

        pii_verdict, scope_verdict, harm_verdict = await asyncio.gather(
            _run_pii(), _run_scope(), _run_harm()
        )

    decision = policy_engine.decide(injection_verdict, pii_verdict, scope_verdict, harm_verdict, project_config)
    result = CheckResult(
        injection=injection_verdict,
        pii=pii_verdict,
        scope=scope_verdict,
        harm=harm_verdict,
        latency_ms=latency_ms,
    )
    return result, decision
