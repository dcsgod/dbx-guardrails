"""Fine-tunes the sub-10ms injection-gate classifier (core.injection_gate's
production model) on the adversarial + benign set from generate_adversarial.py.

Binary sequence classification over DeBERTa-v3-base, matching the
core.injection_gate.fast_classify(text) -> InjectionVerdict interface.
"""
from __future__ import annotations

import mlflow
import pandas as pd


def train(dataset_csv_path: str, base_model: str = "microsoft/deberta-v3-base", epochs: int = 4) -> str:
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    df = pd.read_csv(dataset_csv_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128), batched=True
    )

    with mlflow.start_run(run_name="injection_gate_classifier"):
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_examples", len(df))
        mlflow.log_param("n_positive", int(df["label"].sum()))

        args = TrainingArguments(
            output_dir="./_injection_gate_ckpt",
            num_train_epochs=epochs,
            per_device_train_batch_size=32,
            logging_steps=10,
            report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="injection_gate_classifier",
            registered_model_name="guardrails.models.injection_gate_classifier",
        )
    return model_info.model_uri


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    train(args.dataset_csv, epochs=args.epochs)
