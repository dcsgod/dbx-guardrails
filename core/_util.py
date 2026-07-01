"""Small internal helpers shared across core/ check modules."""
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def maybe_await(value: T | Awaitable[T]) -> T:
    """Awaits `value` if it's awaitable, otherwise returns it as-is.

    Lets every check module accept either a sync or async pluggable
    classifier/LLM callable without forcing callers to wrap sync functions.
    """
    if inspect.isawaitable(value):
        return await value  # type: ignore[return-value]
    return value  # type: ignore[return-value]


def render_history(history: list[dict] | None, max_turns: int, max_chars: int = 800) -> str:
    """Renders the last `max_turns` of `history` as a compact text block for
    inclusion in a prompt or embedding input. Returns "" if no history.

    Deliberately does no token-budget trimming or summarization (unlike a
    full conversation-history fetch) -- this is meant to be cheap and just
    disambiguate short follow-ups, not carry long-range context.
    """
    if not history:
        return ""
    turns = history[-max_turns:] if max_turns > 0 else []
    lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns]
    block = "\n".join(lines)
    return block[:max_chars]
