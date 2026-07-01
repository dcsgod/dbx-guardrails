"""Distills Llama-Guard-labeled + human-QA'd examples
(training/harm_classifier/bootstrap_labels.py) into the small production
multi-label harm classifier that core.harm_guard.fast_classify plugs into at
serve time.

Multi-label regression head over ModernBERT/DeBERTa-v3, one output per
category in training.harm_classifier.bootstrap_labels.CATEGORIES, matching
the HarmVerdict.category_scores shape.
"""
from __future__ import annotations

import mlflow
import pandas as pd

from training.harm_classifier.bootstrap_labels import CATEGORIES


def train(labeled_csv_path: str, base_model: str = "answerdotai/ModernBERT-base", epochs: int = 3) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    df = pd.read_csv(labeled_csv_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(CATEGORIES), problem_type="regression"
    )

    dataset = Dataset.from_pandas(df[["text", *CATEGORIES]])
    dataset = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256), batched=True
    )
    dataset = dataset.map(lambda ex: {"labels": [ex[c] for c in CATEGORIES]})

    with mlflow.start_run(run_name="harm_classifier"):
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_examples", len(df))
        mlflow.log_param("categories", ",".join(CATEGORIES))

        args = TrainingArguments(
            output_dir="./_harm_classifier_ckpt",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            logging_steps=10,
            report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="harm_classifier",
            registered_model_name="guardrails.models.harm_classifier",
        )
    return model_info.model_uri


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-csv", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args.labeled_csv, epochs=args.epochs)
