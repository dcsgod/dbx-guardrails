-- Unity Catalog config table for dbx-guardrails.
-- Append-only, versioned per project_id: the loader always takes the row
-- with MAX(version) for a given project_id. Onboarding a new project is an
-- INSERT here, never a code change or redeploy.

CREATE CATALOG IF NOT EXISTS guardrails
    COMMENT 'dbx-guardrails accelerator: config + audit tables';

CREATE SCHEMA IF NOT EXISTS guardrails.governance
    COMMENT 'Per-project guardrail configuration and audit log';

CREATE TABLE IF NOT EXISTS guardrails.governance.project_configs (
    project_id                  STRING NOT NULL
        COMMENT 'Stable identifier for the integrating application/team',
    scope_definition             STRING NOT NULL
        COMMENT 'Free-text description of the in-scope domain, client-authored',
    scope_confidence_band_low    DOUBLE NOT NULL DEFAULT 0.35
        COMMENT 'Similarity score below this -> fast-path out-of-scope',
    scope_confidence_band_high   DOUBLE NOT NULL DEFAULT 0.65
        COMMENT 'Similarity score above this -> fast-path in-scope; between low/high -> escalate',
    scope_context_turns          INT NOT NULL DEFAULT 3
        COMMENT 'Recent turns of history considered by scope_guard for follow-up continuity',
    pii_necessity_threshold      DOUBLE NOT NULL DEFAULT 0.5
        COMMENT 'Necessity score below this -> mask the entity',
    pii_custom_recognizers       STRING
        COMMENT 'JSON-encoded list of custom Presidio recognizer configs',
    harm_severity_block_threshold INT NOT NULL DEFAULT 4
        COMMENT 'Block when severity >= this value, on the Azure 0/2/4/6 scale',
    harm_categories_enabled      ARRAY<STRING>
        COMMENT 'Subset of hate/violence/self_harm/sexual actively enforced for this project',
    injection_gate_enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    version                      INT NOT NULL
        COMMENT 'Monotonically increasing per project_id; loader takes MAX(version)',
    updated_at                   TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Append-only per-project guardrail configuration, versioned.'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
