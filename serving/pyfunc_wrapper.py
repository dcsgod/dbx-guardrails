"""MLflow pyfunc.PythonModel exposing core.orchestrator.run_checks via
Databricks Model Serving, for non-Python callers (service mode).

Input schema (pandas DataFrame with one row per request, or a dict for
single-row invocation): {project_id: str, text: str, task_intent: str|None,
history: list[dict]|None}. Output: {check_result: <CheckResult JSON>,
policy_decision: <PolicyDecision JSON>} per row.

Delegates to the exact same core/ orchestration as client/async_client.py's
library mode -- this wrapper adds no business logic of its own.
"""
from __future__ import annotations

import asyncio
import json

import mlflow
import pandas as pd

from config.loader import get_config
from core.orchestrator import run_checks


class GuardrailPyfuncModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            rows = [model_input]
        elif isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        else:
            raise TypeError(f"Unsupported model_input type: {type(model_input)}")

        outputs = [self._predict_one(row) for row in rows]
        return pd.DataFrame(outputs)

    @staticmethod
    def _predict_one(row: dict) -> dict:
        project_id = row["project_id"]
        text = row["text"]
        task_intent = row.get("task_intent")
        history = row.get("history")
        if isinstance(history, str) and history:
            history = json.loads(history)

        project_config = get_config(project_id)
        check_result, policy_decision = asyncio.run(
            run_checks(text, project_config, task_intent=task_intent, history=history)
        )
        return {
            "check_result": check_result.model_dump_json(),
            "policy_decision": policy_decision.model_dump_json(),
        }


def log_model(artifact_path: str = "guardrail_model") -> str:
    """Registers the pyfunc model with MLflow. Note: this logs the
    orchestration wrapper itself. The three trained stage models (pii
    necessity, harm classifier, injection classifier) are logged and
    versioned independently via training/*/train_classifier.py so they can
    be championed/challenged on their own -- this wrapper loads whichever
    versions are current champions at serve time (see config/loader.py for
    the analogous pattern applied to project config).
    """
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=GuardrailPyfuncModel(),
            pip_requirements=["pydantic>=2.6", "presidio-analyzer", "presidio-anonymizer", "httpx"],
        )
    return model_info.model_uri
