# dbx-guardrails

A Databricks-native guardrail system providing **selective PII masking**,
**scope enforcement** (blocks off-topic / out-of-context questions, with
conversation-history awareness so terse follow-ups aren't false-positived),
and **harm detection** (offensive/unsafe content), unified behind one policy
engine. Reusable across any Databricks project via a pip-installable
library, and centrally servable via Databricks Model Serving for
non-Python callers.

General-purpose accelerator, not a single client's tool. See `project.md`
(or the design doc this was built from) for the full architecture rationale.

## Install

```bash
pip install -e ".[dev]"          # core + test deps
pip install -e ".[training]"     # + torch/transformers for training/
pip install -e ".[eval]"         # + pandas/sklearn for eval/
```

## Quickstart

### Library mode (in-process, async)

```python
from client.async_client import GuardrailClient

client = GuardrailClient(project_id="supply_chain_qa", mode="library")
decision = await client.check(
    "and last year?",
    history=[
        {"role": "user", "content": "What's our OSA for produce this week?"},
        {"role": "assistant", "content": "94%, down 2pts from last week."},
    ],
)
# decision.action in {"allow", "mask", "block"}
```

`history` is optional everywhere -- omit it and scope/harm checks fall back
to judging the message in isolation. Pass it (the last few turns from
whatever conversation store your app already has) to fix the false-positive
class where a short follow-up reads as out-of-scope on its own.

### Service mode (Databricks Model Serving, any language via REST)

```python
client = GuardrailClient(
    project_id="supply_chain_qa",
    mode="service",
    endpoint_url="https://<workspace>/serving-endpoints/dbx-guardrails/invocations",
)
decision = await client.check("some user message")
```

Both modes return the identical `PolicyDecision` type -- callers never need
to know which mode is active. See `examples/supply_chain_qa_integration.py`
for a full pre-check/post-check integration.

## Onboarding a new project

Insert a row into `guardrails.governance.project_configs` (schema in
`config/ddl/configs_table.sql`, pydantic model in `config/schema.py`).
That's it -- no code change, no redeploy. The config loader
(`config/loader.py`) picks it up within `scope_context_turns`'s TTL window
(default 60s).

Before onboarding, also run:
```bash
python -m training.scope_index.build_vector_index --project-id <your_project_id> --examples-csv <your_examples.csv>
```
to build that project's in-scope example index (seeded with generic
examples if you omit `--examples-csv`, for a quick smoke test).

## Running the benchmark suite

```bash
python -m eval.run_benchmark
```

Runs the full ladder (regex baseline -> zero-shot Llama Guard ->
distilled/fast-path alone -> full hybrid) against the seeded benign-follow-up
regression set plus a small harm/injection sample, and logs one MLflow run
covering every rung. Any model promotion to champion must show this run and
must not regress false-positive rate vs. the current champion
(`eval.run_benchmark.promote_ok`).

## Repo layout

- `core/` -- all checking logic (`injection_gate`, `pii_guard`, `scope_guard`,
  `harm_guard`, `policy_engine`), zero dependency on how it's invoked.
- `config/` -- `ProjectConfig` schema, UC-backed loader with TTL cache, DDL.
- `client/` -- `GuardrailClient`, the library-mode async entrypoint.
- `serving/` -- the same `core/` wrapped as an MLflow pyfunc for Model Serving.
- `training/` -- bootstrap-label + distill scripts for the three trained
  classifiers (PII necessity, harm, injection) plus the scope vector index
  builder. Seeded with synthetic examples where real client data isn't
  available yet -- swap in real data without restructuring.
- `eval/` -- dataset loaders (ToxicChat, OpenAI moderation set, HarmBench,
  the benign-follow-up regression set) and the benchmark-ladder runner.
- `monitoring/` -- Lakehouse Monitoring setup + SQL dashboard definition.
- `tests/` -- unit tests, all model/network calls mocked.
- `examples/` -- reference integration (supply-chain Q&A, async fan-out).

## Status

Core pipeline, config, client/serving wrappers, training scaffolding, eval
harness, monitoring setup, and tests are in place and use placeholder
heuristic/default implementations for the three trained classifiers until
`training/*/train_classifier.py` are run against real (or seeded synthetic)
labeled data and the resulting models are registered/served. Swapping a
placeholder for a trained model is a config change in `serving/`, not a
`core/` rewrite -- every check function accepts its model call as an
injectable parameter for exactly this reason.
