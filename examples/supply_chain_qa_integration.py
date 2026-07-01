"""Reference integration for a supply-chain Q&A app (async OpenAI SDK, Genie
Spaces orchestration) -- the first integration target for this accelerator,
but nothing here is supply-chain-specific; swap `PROJECT_ID` and the Genie
call for any other async fan-out flow.

This file is documentation-by-example, not a dependency of the library.
Run with mocked models (no live Databricks needed):
    python -m examples.supply_chain_qa_integration
"""
from __future__ import annotations

import asyncio

from client.async_client import GuardrailClient
from config.loader import ConfigLoader
from config.schema import ProjectConfig
from core import orchestrator
from core.types import HarmVerdict, InjectionVerdict, PIIVerdict, ScopeVerdict

PROJECT_ID = "supply_chain_qa"


def _install_mock_checks() -> None:
    """Swaps the four check modules' entrypoints for lightweight mocks so
    this example runs standalone, without a live Databricks workspace
    (no vector search index, no served injection/harm models). A real
    integration does NOT do this -- it relies on config.loader.get_config
    resolving real UC config and core/ calling real served models."""

    async def fake_check_injection(text, config, classifier_fn=None):
        return InjectionVerdict(flagged=False, confidence=0.05)

    def fake_detect_entities(text, config):
        return []

    async def fake_pii_mask(text, entities, config, task_intent=None, necessity_fn=None):
        return PIIVerdict(entities=[], masked_text=text, masked_entity_ids=[])

    async def fake_check_scope(query, config, history=None, search_fn=None, llm_fn=None):
        # In-scope if the query or any recent history turn mentions a
        # supply-chain keyword -- stands in for the real vector-search fast
        # path, and still demonstrates history fixing the "and last year?"
        # false-positive case.
        keywords = ("availability", "inventory", "supplier", "shipment", "osa", "last year")
        text_blob = query.lower() + " " + " ".join(t.get("content", "").lower() for t in (history or []))
        return ScopeVerdict(in_scope=any(k in text_blob for k in keywords), confidence=0.8)

    async def fake_check_harm(text, config, classifier_fn=None, llm_fn=None, calibration_table=None):
        return HarmVerdict(category_scores={}, severity=0)

    orchestrator.injection_gate.check_injection = fake_check_injection
    orchestrator.pii_guard.detect_entities = fake_detect_entities
    orchestrator.pii_guard.mask = fake_pii_mask
    orchestrator.scope_guard.check_scope = fake_check_scope
    orchestrator.harm_guard.check_harm = fake_check_harm


def _mock_config_loader() -> ConfigLoader:
    """In a real deployment, ConfigLoader() with no args reads
    guardrails.governance.project_configs via the SQL warehouse. This mock
    lets the example run without a live Databricks workspace."""
    row = ProjectConfig(
        project_id=PROJECT_ID,
        scope_definition="Supply-chain operations: inventory levels, supplier lead times, shipment status, OSA.",
    ).model_dump(mode="json")
    row["scope_confidence_band_low"], row["scope_confidence_band_high"] = row.pop("scope_confidence_band")
    return ConfigLoader(fetch_fn=lambda project_id, table: row)


async def genie_orchestrate(question: str) -> str:
    """Placeholder for the app's real Genie Spaces call."""
    return f"[genie answer for: {question}]"


async def handle_user_turn(client: GuardrailClient, question: str, history: list[dict]) -> str:
    # Pre-check: block/mask before spending any Genie/LLM budget on the
    # request. `history` is whatever the app already tracks for this
    # conversation -- the client does no history management of its own.
    pre_decision = await client.check(question, task_intent="supply_chain_question", history=history)
    if pre_decision.action == "block":
        return f"[blocked pre-check: {pre_decision.reasons}]"
    effective_question = pre_decision.masked_text if pre_decision.action == "mask" else question

    answer = await genie_orchestrate(effective_question)

    # Post-check: the assembled response can itself leak PII pulled from
    # underlying data, or (rarely) drift out of policy -- check it too
    # before returning to the user.
    post_decision = await client.check(answer, task_intent="supply_chain_question", history=history)
    if post_decision.action == "block":
        return "[response blocked post-check]"
    return post_decision.masked_text if post_decision.action == "mask" else answer


async def main() -> None:
    _install_mock_checks()
    client = GuardrailClient(project_id=PROJECT_ID, mode="library", config_loader=_mock_config_loader())
    history: list[dict] = []

    for question in [
        "What's our current on-shelf availability for produce?",
        "and last year?",  # terse follow-up -- should stay in-scope via history
    ]:
        answer = await handle_user_turn(client, question, history)
        print(f"Q: {question}\nA: {answer}\n")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    asyncio.run(main())
