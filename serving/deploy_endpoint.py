"""Creates/updates a Databricks Model Serving endpoint from the registered
guardrail pyfunc model (see pyfunc_wrapper.log_model). Supports a
scale-to-zero config for dev/staging.
"""
from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)


def deploy(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> None:
    w = WorkspaceClient()
    served_entity = ServedEntityInput(
        entity_name=model_name,
        entity_version=model_version,
        workload_size=workload_size,
        scale_to_zero_enabled=scale_to_zero,
    )
    config = EndpointCoreConfigInput(served_entities=[served_entity])

    existing = None
    for ep in w.serving_endpoints.list():
        if ep.name == endpoint_name:
            existing = ep
            break

    if existing is None:
        w.serving_endpoints.create(name=endpoint_name, config=config)
    else:
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=[served_entity])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the dbx-guardrails Model Serving endpoint")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-name", required=True, help="UC-registered model name, e.g. guardrails.models.policy_engine")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--workload-size", default="Small")
    parser.add_argument("--no-scale-to-zero", action="store_true")
    args = parser.parse_args()

    deploy(
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=args.model_version,
        workload_size=args.workload_size,
        scale_to_zero=not args.no_scale_to_zero,
    )


if __name__ == "__main__":
    main()
