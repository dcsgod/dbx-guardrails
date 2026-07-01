"""Reads ProjectConfig rows from the Unity Catalog config table, with a short
in-process TTL cache. Used by both client/ (library mode) and serving/
(service mode) so config resolution behaves identically in either mode.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from config.schema import ProjectConfig

DEFAULT_TABLE = "guardrails.governance.project_configs"
DEFAULT_TTL_SECONDS = 60

FetchFn = Callable[[str, str], dict[str, Any] | None]


def _fetch_via_sql_warehouse(project_id: str, table: str, warehouse_id: str | None = None) -> dict[str, Any] | None:
    """Default fetch implementation: query UC via the Databricks SQL
    Statement Execution API. Returns the latest-version row for project_id,
    or None if no config exists.

    Requires `databricks-sdk` to be configured (env vars / profile) in the
    calling environment. Imported lazily so this module has no hard
    dependency on a live Databricks session at import time -- tests inject
    a fake `fetch_fn` instead of exercising this path.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    wh = warehouse_id or w.config.warehouse_id
    query = (
        f"SELECT * FROM {table} WHERE project_id = :project_id "
        f"ORDER BY version DESC LIMIT 1"
    )
    result = w.statement_execution.execute_statement(
        warehouse_id=wh,
        statement=query,
        parameters=[{"name": "project_id", "value": project_id}],
    )
    rows = result.result.data_array if result.result else None
    if not rows:
        return None
    columns = [c.name for c in result.manifest.schema.columns]
    return dict(zip(columns, rows[0]))


def _row_to_config(row: dict[str, Any]) -> ProjectConfig:
    """Translates the flat DDL row shape (configs_table.sql) into the nested
    ProjectConfig pydantic shape."""
    row = dict(row)
    low = row.pop("scope_confidence_band_low", 0.35)
    high = row.pop("scope_confidence_band_high", 0.65)
    row["scope_confidence_band"] = (float(low), float(high))

    recognizers = row.get("pii_custom_recognizers")
    if isinstance(recognizers, str):
        row["pii_custom_recognizers"] = json.loads(recognizers) if recognizers else []

    return ProjectConfig.model_validate(row)


class ConfigLoader:
    """Per-project_id cache with a short TTL. One instance is shared across
    a process (client mode) or a serving replica (service mode)."""

    def __init__(
        self,
        table: str = DEFAULT_TABLE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        fetch_fn: FetchFn | None = None,
    ) -> None:
        self.table = table
        self.ttl_seconds = ttl_seconds
        self._fetch_fn = fetch_fn or (lambda pid, tbl: _fetch_via_sql_warehouse(pid, tbl))
        self._cache: dict[str, tuple[float, ProjectConfig]] = {}

    def get_config(self, project_id: str) -> ProjectConfig:
        cached = self._cache.get(project_id)
        now = time.monotonic()
        if cached is not None and (now - cached[0]) < self.ttl_seconds:
            return cached[1]

        row = self._fetch_fn(project_id, self.table)
        if row is None:
            raise KeyError(
                f"No guardrail config found for project_id={project_id!r} in {self.table}"
            )
        config = _row_to_config(row)
        self._cache[project_id] = (now, config)
        return config

    def invalidate(self, project_id: str | None = None) -> None:
        if project_id is None:
            self._cache.clear()
        else:
            self._cache.pop(project_id, None)


_default_loader: ConfigLoader | None = None


def get_config(project_id: str) -> ProjectConfig:
    """Module-level convenience using a lazily-created default loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader.get_config(project_id)
