"""Thin async client for calling Databricks Model Serving endpoints from
core/ check modules that need an LLM-judge escalation (scope_guard,
harm_guard). Lazily reads host/token from the Databricks SDK config so this
module has no hard dependency on a live workspace at import time -- tests
inject a `llm_fn`/`search_fn` directly instead of exercising this path.
"""
from __future__ import annotations

from typing import Any

import httpx


async def invoke_endpoint(
    endpoint_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    from databricks.sdk.config import Config

    cfg = Config()  # reads DATABRICKS_HOST/DATABRICKS_TOKEN or profile
    url = f"{cfg.host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"}
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    import json

    return json.loads(content)
