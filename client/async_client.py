"""Library-mode entrypoint: GuardrailClient. Both `library` and `service`
mode return the identical PolicyDecision type -- callers should never need
to know which mode is active.
"""
from __future__ import annotations

import json
from typing import Literal

import httpx

from config.loader import ConfigLoader
from core.orchestrator import run_checks
from core.types import PolicyDecision


class GuardrailClient:
    def __init__(
        self,
        project_id: str,
        mode: Literal["library", "service"] = "library",
        endpoint_url: str | None = None,
        config_loader: ConfigLoader | None = None,
        http_timeout_seconds: float = 10.0,
    ) -> None:
        if mode == "service" and not endpoint_url:
            raise ValueError("endpoint_url is required when mode='service'")

        self.project_id = project_id
        self.mode = mode
        self.endpoint_url = endpoint_url
        self._config_loader = config_loader or ConfigLoader()
        self._http_timeout_seconds = http_timeout_seconds

    async def check(
        self,
        text: str,
        task_intent: str | None = None,
        history: list[dict] | None = None,
    ) -> PolicyDecision:
        """`history` is optional and caller-supplied (e.g. the last few
        turns from whatever conversation store the integrating app already
        has) -- this client does no history management of its own, it just
        threads what it's given down to scope_guard/harm_guard."""
        if self.mode == "library":
            return await self._check_library(text, task_intent, history)
        return await self._check_service(text, task_intent, history)

    async def _check_library(
        self, text: str, task_intent: str | None, history: list[dict] | None
    ) -> PolicyDecision:
        project_config = self._config_loader.get_config(self.project_id)
        _check_result, decision = await run_checks(
            text, project_config, task_intent=task_intent, history=history
        )
        return decision

    async def _check_service(
        self, text: str, task_intent: str | None, history: list[dict] | None
    ) -> PolicyDecision:
        payload = {
            "project_id": self.project_id,
            "text": text,
            "task_intent": task_intent,
            "history": json.dumps(history) if history else None,
        }
        async with httpx.AsyncClient(timeout=self._http_timeout_seconds) as client:
            response = await client.post(self.endpoint_url, json={"dataframe_records": [payload]})
            response.raise_for_status()
            body = response.json()

        prediction = body["predictions"][0]
        return PolicyDecision.model_validate_json(prediction["policy_decision"])
