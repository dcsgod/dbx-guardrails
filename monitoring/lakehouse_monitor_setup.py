"""Attaches Lakehouse Monitoring to the audit_log table: tracks drift on
category-score distributions, escalation rates per stage, block rates per
project, and latency percentiles.
"""
from __future__ import annotations

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)

AUDIT_TABLE = "guardrails.governance.audit_log"
BASELINE_TABLE = "guardrails.governance.audit_log_baseline"


def setup_monitor(
    table_name: str = AUDIT_TABLE,
    output_schema: str = "guardrails.governance",
    granularities: list[str] | None = None,
) -> None:
    w = WorkspaceClient()
    w.quality_monitors.create(
        table_name=table_name,
        assets_dir=f"/Shared/guardrails_monitoring/{table_name.replace('.', '_')}",
        output_schema_name=output_schema,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            granularities=granularities or ["1 day"],
            model_id_col="model_version_harm",
            prediction_col="policy_decision",
            timestamp_col="timestamp",
        ),
        slicing_exprs=["project_id"],
    )


if __name__ == "__main__":
    setup_monitor()
