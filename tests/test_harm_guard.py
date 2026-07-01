import pytest

from config.schema import ProjectConfig
from core.harm_guard import check_harm, map_to_severity
from core.types import HarmVerdict


def _config(**overrides) -> ProjectConfig:
    return ProjectConfig(project_id="p1", scope_definition="test", **overrides)


def test_map_to_severity_picks_max_across_categories():
    scores = {"hate": 0.2, "violence": 0.95}
    severity = map_to_severity(scores)
    assert severity == 6  # violence 0.95 crosses the 1.0-bucket threshold -> 6


def test_map_to_severity_unknown_category_ignored():
    assert map_to_severity({"unknown_category": 0.99}) == 0


def test_map_to_severity_empty_scores():
    assert map_to_severity({}) == 0


@pytest.mark.asyncio
async def test_check_harm_uses_fast_path_when_confident():
    async def fake_classifier(text):
        return HarmVerdict(category_scores={"hate": 0.9}, severity=6)

    verdict = await check_harm("some text", _config(), classifier_fn=fake_classifier)
    assert verdict.severity == 6
    assert verdict.escalated is False


@pytest.mark.asyncio
async def test_check_harm_escalates_when_fast_path_unconfident():
    async def fake_classifier(text):
        return None  # unconfident -> escalate

    async def fake_llm(text):
        return HarmVerdict(category_scores={"hate": 0.9, "violence": 0.1}, severity=6, escalated=True)

    verdict = await check_harm("some text", _config(), classifier_fn=fake_classifier, llm_fn=fake_llm)
    assert verdict.escalated is True
    assert verdict.severity == 6


@pytest.mark.asyncio
async def test_check_harm_filters_to_enabled_categories_only():
    async def fake_classifier(text):
        return None

    async def fake_llm(text):
        return HarmVerdict(category_scores={"hate": 0.9, "sexual": 0.9}, severity=6, escalated=True)

    config = _config(harm_categories_enabled=["hate"])
    verdict = await check_harm("some text", config, classifier_fn=fake_classifier, llm_fn=fake_llm)
    assert "sexual" not in verdict.category_scores
    assert "hate" in verdict.category_scores
