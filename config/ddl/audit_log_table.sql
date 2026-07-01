-- Unity Catalog audit log for every guardrail check. Default logging is
-- privacy-preserving (hash + entity types only); a project may opt into raw
-- text logging via a future config flag, never on by default.

CREATE TABLE IF NOT EXISTS guardrails.governance.audit_log (
    request_id            STRING NOT NULL,
    project_id            STRING NOT NULL,
    conversation_id        STRING
        COMMENT 'Caller-supplied, nullable -- lets a block/mask be traced back to its thread',
    timestamp              TIMESTAMP NOT NULL,
    input_text_hash         STRING NOT NULL
        COMMENT 'SHA-256 of input text; raw text only logged if project opts in (not modeled here yet)',
    history_turns_used      INT NOT NULL DEFAULT 0
        COMMENT 'How many turns of caller-supplied history were actually used for this request',
    injection_verdict        STRING
        COMMENT 'JSON-encoded InjectionVerdict',
    pii_verdict             STRING
        COMMENT 'JSON-encoded PIIVerdict (entity types only, no raw entity text)',
    scope_verdict           STRING
        COMMENT 'JSON-encoded ScopeVerdict',
    harm_verdict            STRING
        COMMENT 'JSON-encoded HarmVerdict',
    policy_decision         STRING NOT NULL
        COMMENT 'JSON-encoded PolicyDecision',
    latency_ms_injection      DOUBLE,
    latency_ms_pii           DOUBLE,
    latency_ms_scope         DOUBLE,
    latency_ms_harm          DOUBLE,
    model_version_injection   STRING,
    model_version_pii        STRING,
    model_version_scope      STRING,
    model_version_harm       STRING,
    escalated_scope          BOOLEAN NOT NULL DEFAULT FALSE,
    escalated_harm           BOOLEAN NOT NULL DEFAULT FALSE
)
USING DELTA
PARTITIONED BY (project_id)
COMMENT 'Per-request audit trail for the dbx-guardrails policy engine.'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
