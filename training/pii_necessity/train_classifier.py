"""Distills bootstrap labels (training/pii_necessity/bootstrap_labels.py)
into the small production necessity classifier that core/pii_guard.py's
score_necessity plugs into at serve time.

Uses a lightweight DeBERTa-v3-base sequence classification head over
"entity + context window + task_intent" text, matching the interface
core.pii_guard.score_necessity expects (entity, context, task_intent) ->
probability necessary.
"""
from __future__ import annotations

import mlflow
import pandas as pd


def build_training_text(row: pd.Series) -> str:
    return f"[TASK] {row['task_intent']} [ENTITY] {row['entity_id']} [CONTEXT] {row['text']}"


def train(labeled_csv_path: str, base_model: str = "microsoft/deberta-v3-base", epochs: int = 3) -> str:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    df = pd.read_csv(labeled_csv_path)
    df["text"] = df.apply(build_training_text, axis=1)
    df["label"] = df["necessary"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256), batched=True)

    with mlflow.start_run(run_name="pii_necessity_classifier"):
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_examples", len(df))

        args = TrainingArguments(
            output_dir="./_pii_necessity_ckpt",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            logging_steps=10,
            report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="pii_necessity_classifier",
            registered_model_name="guardrails.models.pii_necessity_classifier",
        )
    return model_info.model_uri


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-csv", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args.labeled_csv, epochs=args.epochs)
