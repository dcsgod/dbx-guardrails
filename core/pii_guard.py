"""Entity detection + necessity classification + masking.

Fast path only -- no LLM call in this module's hot path (design principle
5/6). The necessity classifier is the distilled model trained via
training/pii_necessity/; there is no live LLM escalation here because PII
decisions must be low-latency and deterministic per project config.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from config.schema import ProjectConfig
from core._util import maybe_await
from core.types import Entity, PIIVerdict

NecessityFn = Callable[[Entity, str, str | None], float | Awaitable[float]]

_PLACEHOLDER_BY_TYPE = {
    "PHONE_NUMBER": "[PHONE]",
    "US_SSN": "[SSN]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
}


def _placeholder_for(entity_type: str) -> str:
    return _PLACEHOLDER_BY_TYPE.get(entity_type, f"[{entity_type}]")


def detect_entities(text: str, project_config: ProjectConfig) -> list[Entity]:
    """Presidio pass plus any custom recognizers registered in the project
    config. Presidio is imported lazily so this module can be unit-tested
    (and the rest of the pipeline exercised) without the dependency
    installed in every environment.
    """
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer

    analyzer = AnalyzerEngine()
    for recognizer_cfg in project_config.pii_custom_recognizers:
        analyzer.registry.add_recognizer(PatternRecognizer(**recognizer_cfg))

    results = analyzer.analyze(text=text, language="en")
    return [
        Entity(
            text=text[r.start : r.end],
            type=r.entity_type,
            start=r.start,
            end=r.end,
            confidence=r.score,
        )
        for r in results
    ]


def _default_necessity(entity: Entity, context: str, task_intent: str | None) -> float:
    """Placeholder used until the fine-tuned necessity classifier (trained
    via training/pii_necessity/) is registered and swapped in. Conservative
    default: assume PII is not necessary unless proven otherwise, so the
    fail-safe direction is to mask, not leak."""
    return 0.0


async def score_necessity(
    entity: Entity,
    context: str,
    task_intent: str | None = None,
    necessity_fn: NecessityFn | None = None,
) -> float:
    """Probability the entity is necessary to fulfill the request. Input is
    the entity plus a window of surrounding text plus (if available) the
    declared task/intent for the turn."""
    fn = necessity_fn or _default_necessity
    return await maybe_await(fn(entity, context, task_intent))


async def mask(
    text: str,
    entities: list[Entity],
    project_config: ProjectConfig,
    task_intent: str | None = None,
    necessity_fn: NecessityFn | None = None,
) -> PIIVerdict:
    """Masks any entity below the project's configured necessity threshold
    with a typed placeholder; leaves the rest untouched."""
    if not entities:
        return PIIVerdict(entities=[], masked_text=text, masked_entity_ids=[])

    to_mask: list[tuple[Entity, str]] = []
    for entity in entities:
        window_start = max(0, entity.start - 40)
        window_end = min(len(text), entity.end + 40)
        context = text[window_start:window_end]
        necessity = await score_necessity(entity, context, task_intent, necessity_fn=necessity_fn)
        if necessity < project_config.pii_necessity_threshold:
            to_mask.append((entity, f"{entity.type}:{entity.start}:{entity.end}"))

    masked_text = text
    masked_ids: list[str] = []
    for entity, entity_id in sorted(to_mask, key=lambda pair: pair[0].start, reverse=True):
        placeholder = _placeholder_for(entity.type)
        masked_text = masked_text[: entity.start] + placeholder + masked_text[entity.end :]
        masked_ids.append(entity_id)

    return PIIVerdict(entities=entities, masked_text=masked_text, masked_entity_ids=masked_ids)
