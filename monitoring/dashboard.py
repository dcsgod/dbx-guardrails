"""Databricks SQL dashboard definition covering: requests per project,
block/mask/allow breakdown, escalation rate trend, latency percentiles, and
a false-positive-rate proxy (allow-after-human-override rate, if exposed).
"""
from __future__ import annotations

AUDIT_TABLE = "guardrails.governance.audit_log"

DASHBOARD_QUERIES = {
    "requests_per_project": f"""
        SELECT project_id, date_trunc('day', timestamp) AS day, count(*) AS requests
        FROM {AUDIT_TABLE}
        GROUP BY 1, 2
        ORDER BY 2 DESC
    """,
    "decision_breakdown": f"""
        SELECT
            project_id,
            get_json_object(policy_decision, '$.action') AS action,
            count(*) AS n
        FROM {AUDIT_TABLE}
        GROUP BY 1, 2
    """,
    "escalation_rate_trend": f"""
        SELECT
            date_trunc('day', timestamp) AS day,
            avg(int(escalated_scope)) AS scope_escalation_rate,
            avg(int(escalated_harm)) AS harm_escalation_rate
        FROM {AUDIT_TABLE}
        GROUP BY 1
        ORDER BY 1
    """,
    "latency_percentiles": f"""
        SELECT
            project_id,
            percentile_approx(latency_ms_injection, 0.5) AS p50_injection,
            percentile_approx(latency_ms_injection, 0.95) AS p95_injection,
            percentile_approx(latency_ms_scope, 0.5) AS p50_scope,
            percentile_approx(latency_ms_scope, 0.95) AS p95_scope,
            percentile_approx(latency_ms_harm, 0.5) AS p50_harm,
            percentile_approx(latency_ms_harm, 0.95) AS p95_harm
        FROM {AUDIT_TABLE}
        GROUP BY 1
    """,
    # Proxy for false-positive rate: a scope/harm block that was later
    # overridden (allowed) by a human reviewer, if the client's admin UI
    # exposes an override action back into the audit log. Returns 0 rows
    # until that override write-path exists -- left here as the target
    # query rather than removed, per project.md's "not silently dropped"
    # convention for known-incomplete coverage.
    "false_positive_proxy": f"""
        SELECT project_id, count(*) AS overridden_blocks
        FROM {AUDIT_TABLE}
        WHERE get_json_object(policy_decision, '$.action') = 'block'
          AND get_json_object(policy_decision, '$.overridden') = 'true'
        GROUP BY 1
    """,
}


def render_lakeview_definition() -> dict:
    """Returns a minimal Lakeview dashboard definition (datasets + one
    counter/line widget per query) suitable for `databricks lakeview
    create`. Kept intentionally minimal -- extend widget layout in the
    Databricks UI once created."""
    return {
        "datasets": [{"name": name, "query": query.strip()} for name, query in DASHBOARD_QUERIES.items()],
        "pages": [
            {
                "name": "guardrail_overview",
                "layout": [{"widget": {"name": name}, "position": {"x": 0, "y": i * 4, "width": 6, "height": 4}} for i, name in enumerate(DASHBOARD_QUERIES)],
            }
        ],
    }
