import pytest

from config.schema import ProjectConfig
from core.pii_guard import mask
from core.types import Entity


def _config(**overrides) -> ProjectConfig:
    return ProjectConfig(project_id="p1", scope_definition="test", pii_necessity_threshold=0.5, **overrides)


@pytest.mark.asyncio
async def test_mask_masks_low_necessity_entities():
    text = "My SSN is 123-45-6789, thanks."
    entity = Entity(text="123-45-6789", type="US_SSN", start=10, end=21, confidence=0.95)

    async def necessity_fn(entity, context, task_intent):
        return 0.1  # below threshold -> mask

    verdict = await mask(text, [entity], _config(), necessity_fn=necessity_fn)

    assert "[SSN]" in verdict.masked_text
    assert "123-45-6789" not in verdict.masked_text
    assert verdict.masked_entity_ids == ["US_SSN:10:21"]


@pytest.mark.asyncio
async def test_mask_leaves_necessary_entities_untouched():
    text = "Update my phone to 555-123-4567 please."
    entity = Entity(text="555-123-4567", type="PHONE_NUMBER", start=20, end=32, confidence=0.9)

    async def necessity_fn(entity, context, task_intent):
        return 0.9  # above threshold -> keep

    verdict = await mask(text, [entity], _config(), task_intent="update_contact_info", necessity_fn=necessity_fn)

    assert verdict.masked_text == text
    assert verdict.masked_entity_ids == []


@pytest.mark.asyncio
async def test_mask_no_entities_returns_original_text():
    text = "No PII here."
    verdict = await mask(text, [], _config())
    assert verdict.masked_text == text
    assert verdict.entities == []
