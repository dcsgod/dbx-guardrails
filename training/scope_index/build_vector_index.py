"""Builds/refreshes the Databricks Vector Search index of in-scope example
queries that core.scope_guard.embed_and_search queries against, one index
per project_id (guardrails.scope_index.<project_id>).

No real per-project example queries exist yet -- `seed_examples()` provides
a small generic retail-analytics set (mirroring the supply-chain Q&A
integration target in project.md) so the index-build + scope-check pipeline
can be exercised end to end. Swap in a project's real example queries
(curated from their docs/FAQs/actual traffic) via `--examples-csv`.
"""
from __future__ import annotations

import argparse

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient


def seed_examples(project_id: str) -> pd.DataFrame:
    rows = [
        {"id": "1", "example": "What were our on-shelf availability rates last week?"},
        {"id": "2", "example": "Show me the sales trend for the beverage category."},
        {"id": "3", "example": "What's our current supplier lead time for produce?"},
        {"id": "4", "example": "Summarize the latest competitive pricing news."},
        {"id": "5", "example": "How many units did we sell of this SKU last month?"},
    ]
    return pd.DataFrame(rows).assign(project_id=project_id)


def build_index(
    project_id: str,
    examples_df: pd.DataFrame,
    source_table: str | None = None,
    endpoint_name: str = "guardrails-scope-index-endpoint",
) -> None:
    table_name = source_table or f"guardrails.scope_index.{project_id}_examples"

    w = WorkspaceClient()
    examples_df.to_parquet(f"/tmp/{project_id}_scope_examples.parquet")
    # In a real workspace, write examples_df to `table_name` via Spark/UC
    # before creating the index; omitted here since this script runs
    # outside a cluster in this scaffold.

    client = VectorSearchClient()
    index_name = f"guardrails.scope_index.{project_id}"
    existing = [idx.name for idx in client.list_indexes(endpoint_name).vector_indexes or []]
    if index_name in existing:
        client.get_index(index_name).sync()
        return

    client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name=table_name,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="example",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--examples-csv", default=None)
    args = parser.parse_args()

    examples_df = pd.read_csv(args.examples_csv) if args.examples_csv else seed_examples(args.project_id)
    build_index(args.project_id, examples_df)


if __name__ == "__main__":
    main()
