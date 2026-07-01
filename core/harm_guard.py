"""Distilled multi-label harm classifier + Llama Guard 3 LLM-judge
escalation.

Returns raw category_scores/severity only -- this module never decides
`blocked`. policy_engine.decide is the sole place
harm_severity_block_threshold is applied (see core/policy_engine.py).
"""
from __future__ import annotations

from typing import Awaitable, Callable

from config.schema import ProjectConfig
from core._util import maybe_await
from core.types import HarmVerdict

FastClassifyFn = Callable[[str], HarmVerdict | None | Awaitable[HarmVerdict | None]]
LLMJudgeFn = Callable[[str], HarmVerdict | Awaitable[HarmVerdict]]

# Placeholder calibration table mapping raw category scores to the Azure
# 0/2/4/6 severity scale. Production wiring loads this from a Delta/UC asset
# (per project.md section 4.4) fit against a labeled validation set --
# stored as data, not hardcoded, once Phase 3 of the roadmap trains it.
_DEFAULT_CALIBRATION_TABLE = {
    "hate": [(0.25, 0), (0.5, 2), (0.7, 4), (0.85, 6)],
    "violence": [(0.25, 0), (0.5, 2), (0.7, 4), (0.85, 6)],
    "self_harm": [(0.2, 0), (0.4, 2), (0.6, 4), (0.8, 6)],
    "sexual": [(0.3, 0), (0.55, 2), (0.75, 4), (0.9, 6)],
}


def map_to_severity(
    category_scores: dict[str, float],
    calibration_table: dict[str, list[tuple[float, int]]] | None = None,
) -> int:
    """Maps raw scores to the 0/2/4/6 severity scale, calibrated per
    category against a labeled validation set. Returns the max severity
    across all scored categories (worst-case wins)."""
    table = calibration_table or _DEFAULT_CALIBRATION_TABLE
    severities: list[int] = []
    for category, score in category_scores.items():
        thresholds = table.get(category)
        if not thresholds:
            continue
        severity = 0
        for threshold, mapped in sorted(thresholds):
            if score >= threshold:
                severity = mapped
        severities.append(severity)
    return max(severities, default=0)


def _default_fast_classify(text: str) -> HarmVerdict | None:
    """Placeholder used until the distilled classifier (trained via
    training/harm_classifier/) is registered. Returns None (meaning
    "escalate") for anything not trivially benign, so the safe default is
    to defer to the LLM judge rather than guess."""
    if not text.strip():
        return HarmVerdict(category_scores={}, severity=0)
    return None


async def fast_classify(
    text: str, classifier_fn: FastClassifyFn | None = None
) -> HarmVerdict | None:
    """The distilled multi-label classifier; returns a verdict if
    confident, None if not."""
    fn = classifier_fn or _default_fast_classify
    return await maybe_await(fn(text))


async def llm_judge_classify(text: str, llm_fn: LLMJudgeFn | None = None) -> HarmVerdict:
    """Llama Guard 3 call for escalation."""
    if llm_fn is not None:
        return await maybe_await(llm_fn(text))

    from core._model_client import invoke_endpoint

    system_prompt = (
        "You are a content-safety classifier. Score the CURRENT message on "
        "hate, violence, self_harm, and sexual categories, each 0.0-1.0. "
        'Respond with structured JSON: {"category_scores": '
        '{"hate": float, "violence": float, "self_harm": float, "sexual": float}}.'
    )
    response = await invoke_endpoint(
        endpoint_name="harm-guard-llama-guard-3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    scores = response.get("category_scores", {})
    return HarmVerdict(
        category_scores=scores,
        severity=map_to_severity(scores),
        escalated=True,
    )


async def check_harm(
    text: str,
    project_config: ProjectConfig,
    classifier_fn: FastClassifyFn | None = None,
    llm_fn: LLMJudgeFn | None = None,
    calibration_table: dict[str, list[tuple[float, int]]] | None = None,
) -> HarmVerdict:
    fast_result = await fast_classify(text, classifier_fn=classifier_fn)
    verdict = fast_result if fast_result is not None else await llm_judge_classify(text, llm_fn=llm_fn)

    # Category filtering + calibration apply uniformly regardless of which
    # path produced the verdict, so project config is respected the same way
    # whether the fast classifier or the LLM judge answered.
    enabled = set(project_config.harm_categories_enabled)
    scores = (
        {k: v for k, v in verdict.category_scores.items() if k in enabled}
        if enabled
        else verdict.category_scores
    )
    return verdict.model_copy(
        update={"category_scores": scores, "severity": map_to_severity(scores, calibration_table)}
    )
