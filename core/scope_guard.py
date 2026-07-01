"""Vector-search fast path + LLM-judge escalation for scope enforcement.

History-aware by design: a terse follow-up ("Yes.", "and last year?") reads
as out-of-scope in isolation but is a clear continuation of an in-scope
thread once the recent turns are considered. `history` is optional everywhere
in this module -- callers that don't have it simply omit it, and behavior
degrades to an isolated check rather than requiring history.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from config.schema import ProjectConfig
from core._util import maybe_await, render_history
from core.types import ScopeVerdict

SearchFn = Callable[[str, str], list[tuple[str, float]] | Awaitable[list[tuple[str, float]]]]
LLMJudgeFn = Callable[[str, str, str], ScopeVerdict | Awaitable[ScopeVerdict]]


async def embed_and_search(
    query: str,
    project_id: str,
    history: list[dict] | None = None,
    context_turns: int = 3,
    search_fn: SearchFn | None = None,
) -> list[tuple[str, float]]:
    """Queries the project's Databricks Vector Search index of in-scope
    example queries. When `history` is non-empty, the text embedded for
    search is the current message prefixed with a short rendering of the
    last `context_turns` turns -- not the raw query alone. This is the fix
    for the false-positive class where a terse follow-up reads as
    out-of-scope in isolation but is a continuation of an in-scope thread.
    """
    history_block = render_history(history, context_turns)
    search_text = f"{history_block}\nuser: {query}" if history_block else query

    if search_fn is not None:
        return await maybe_await(search_fn(search_text, project_id))

    from databricks.vector_search.client import VectorSearchClient

    client = VectorSearchClient()
    index = client.get_index(index_name=f"guardrails.scope_index.{project_id}")
    results = index.similarity_search(query_text=search_text, num_results=5, columns=["example", "score"])
    rows = results.get("result", {}).get("data_array", []) if results else []
    return [(row[0], float(row[1])) for row in rows]


def fast_verdict(
    scores: list[tuple[str, float]], project_config: ProjectConfig
) -> ScopeVerdict | None:
    """Returns a verdict if the top similarity score is clearly above or
    below the project's confidence band; returns None (meaning "escalate")
    if it falls inside the band."""
    if not scores:
        return None

    top_example, top_score = max(scores, key=lambda pair: pair[1])
    low, high = project_config.scope_confidence_band

    if top_score >= high:
        return ScopeVerdict(in_scope=True, confidence=top_score, reason=f"matched: {top_example}")
    if top_score <= low:
        return ScopeVerdict(in_scope=False, confidence=1 - top_score, reason="no close in-scope match")
    return None


async def llm_judge_verdict(
    query: str,
    scope_definition: str,
    history: list[dict] | None = None,
    context_turns: int = 3,
    llm_fn: LLMJudgeFn | None = None,
) -> ScopeVerdict:
    """Called only on escalation; passes the project's explicit scope
    definition (client-authored text), the query, and the same recent-turn
    history to an LLM with a structured-output prompt, explicitly instructed
    to use history only to judge whether the *current* message continues an
    in-scope thread, not to re-judge prior turns."""
    history_block = render_history(history, context_turns)

    if llm_fn is not None:
        return await maybe_await(llm_fn(query, scope_definition, history_block))

    from core._model_client import invoke_endpoint

    system_prompt = (
        "You judge whether a user's CURRENT message is in-scope for this "
        f"assistant. In-scope definition:\n{scope_definition}\n\n"
        "If recent conversation turns are provided below, use them ONLY to "
        "judge whether the current message continues an in-scope thread "
        "(e.g. a short follow-up like 'Yes.' referring back to an in-scope "
        "question is in-scope) -- do not re-judge the prior turns "
        "themselves. Respond with structured JSON: "
        '{"in_scope": bool, "confidence": float, "reason": str}.'
    )
    user_content = f"[Recent conversation]\n{history_block}\n\n[Current message]\n{query}" if history_block else query

    response = await invoke_endpoint(
        endpoint_name="scope-guard-llm-judge",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return ScopeVerdict(
        in_scope=bool(response.get("in_scope", True)),
        confidence=float(response.get("confidence", 0.5)),
        reason=response.get("reason"),
        escalated=True,
    )


async def check_scope(
    query: str,
    project_config: ProjectConfig,
    history: list[dict] | None = None,
    search_fn: SearchFn | None = None,
    llm_fn: LLMJudgeFn | None = None,
) -> ScopeVerdict:
    """Orchestrates fast path then escalation, threading `history` through
    both."""
    scores = await embed_and_search(
        query,
        project_config.project_id,
        history=history,
        context_turns=project_config.scope_context_turns,
        search_fn=search_fn,
    )
    verdict = fast_verdict(scores, project_config)
    if verdict is not None:
        return verdict

    return await llm_judge_verdict(
        query,
        project_config.scope_definition,
        history=history,
        context_turns=project_config.scope_context_turns,
        llm_fn=llm_fn,
    )
