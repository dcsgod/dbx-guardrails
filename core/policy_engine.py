"""Combines verdicts per project config into a single PolicyDecision.

Pure function, no I/O -- fully unit-testable without any model calls (feed it
verdicts directly). This is the ONLY place any threshold-vs-verdict block
decision is made; no check module decides `blocked` on its own.

Precedence: injection block > harm block > scope block > PII masking > allow.
"""
from __future__ import annotations

from config.schema import ProjectConfig
from core.types import HarmVerdict, InjectionVerdict, PIIVerdict, PolicyDecision, ScopeVerdict


def decide(
    injection: InjectionVerdict,
    pii: PIIVerdict,
    scope: ScopeVerdict,
    harm: HarmVerdict,
    project_config: ProjectConfig,
) -> PolicyDecision:
    if injection.flagged:
        return PolicyDecision(
            action="block",
            masked_text=None,
            reasons=[f"injection: {injection.reason or 'flagged by injection gate'}"],
        )

    if harm.severity >= project_config.harm_severity_block_threshold:
        top_category = max(harm.category_scores, key=harm.category_scores.get, default="unknown")
        return PolicyDecision(
            action="block",
            masked_text=None,
            reasons=[f"harm: severity={harm.severity} category={top_category}"],
        )

    if not scope.in_scope:
        return PolicyDecision(
            action="block",
            masked_text=None,
            reasons=[f"scope: {scope.reason or 'out of scope'}"],
        )

    if pii.masked_entity_ids:
        return PolicyDecision(
            action="mask",
            masked_text=pii.masked_text,
            reasons=[f"pii: masked {len(pii.masked_entity_ids)} entity(ies)"],
        )

    return PolicyDecision(action="allow", masked_text=None, reasons=[])
