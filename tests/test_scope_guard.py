import pytest

from config.schema import ProjectConfig
from core.scope_guard import check_scope, embed_and_search, fast_verdict
from core.types import ScopeVerdict


def _config(**overrides) -> ProjectConfig:
    return ProjectConfig(
        project_id="p1",
        scope_definition="Retail analytics: sales, inventory, competitive intel.",
        scope_confidence_band=(0.35, 0.65),
        **overrides,
    )


@pytest.mark.asyncio
async def test_embed_and_search_includes_history_in_search_text():
    captured = {}

    async def fake_search_fn(search_text: str, project_id: str):
        captured["search_text"] = search_text
        return [("example", 0.9)]

    history = [
        {"role": "user", "content": "Can you pull the competitive news brief?"},
        {"role": "assistant", "content": "Sure, here it is..."},
    ]
    await embed_and_search("Yes.", "p1", history=history, search_fn=fake_search_fn)

    assert "competitive news brief" in captured["search_text"]
    assert "Yes." in captured["search_text"]


@pytest.mark.asyncio
async def test_embed_and_search_without_history_is_just_the_query():
    captured = {}

    async def fake_search_fn(search_text: str, project_id: str):
        captured["search_text"] = search_text
        return [("example", 0.9)]

    await embed_and_search("Yes.", "p1", history=None, search_fn=fake_search_fn)
    assert captured["search_text"] == "Yes."


def test_fast_verdict_above_band_is_in_scope():
    config = _config()
    verdict = fast_verdict([("example", 0.9)], config)
    assert verdict is not None
    assert verdict.in_scope is True


def test_fast_verdict_below_band_is_out_of_scope():
    config = _config()
    verdict = fast_verdict([("example", 0.1)], config)
    assert verdict is not None
    assert verdict.in_scope is False


def test_fast_verdict_inside_band_escalates():
    config = _config()
    verdict = fast_verdict([("example", 0.5)], config)
    assert verdict is None


def test_fast_verdict_no_scores_escalates():
    config = _config()
    assert fast_verdict([], config) is None


@pytest.mark.asyncio
async def test_check_scope_regression_terse_followup_after_in_scope_opener():
    """This is the exact false-positive class the accelerator fixes: a
    terse follow-up would look out-of-scope in isolation, but with history
    threaded through, the fast path (or escalation) should call it in-scope.
    """
    config = _config()
    history = [
        {"role": "user", "content": "How did our electronics category do against competitors this year?"},
        {"role": "assistant", "content": "Share held steady, softness in gaming consoles."},
    ]

    async def search_fn(search_text: str, project_id: str):
        # Simulate the vector index correctly matching once history is
        # folded into the search text (would score low on "Yes." alone).
        if "electronics" in search_text or "competitors" in search_text:
            return [("competitive intel example", 0.85)]
        return [("unrelated example", 0.05)]

    verdict = await check_scope(
        "How many Nintendo Switch 2 consoles did we sell in the last year?",
        config,
        history=history,
        search_fn=search_fn,
    )
    assert verdict.in_scope is True


@pytest.mark.asyncio
async def test_check_scope_escalates_to_llm_judge_when_ambiguous():
    config = _config()

    async def search_fn(search_text: str, project_id: str):
        return [("borderline example", 0.5)]  # inside confidence band -> escalate

    async def llm_fn(query: str, scope_definition: str, history_block: str):
        return ScopeVerdict(in_scope=True, confidence=0.7, reason="judged in-scope", escalated=True)

    verdict = await check_scope("some ambiguous query", config, search_fn=search_fn, llm_fn=llm_fn)
    assert verdict.escalated is True
    assert verdict.in_scope is True
