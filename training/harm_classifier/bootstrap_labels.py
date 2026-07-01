"""Llama Guard 3 zero-shot labeling for the harm classifier bootstrap.

Runs Llama Guard 3 (served via Databricks Model Serving, see
core.harm_guard.llm_judge_classify for the same endpoint call pattern) over
an unlabeled text sample, writes a human-review queue for a stratified
sample before those labels feed distillation.

No real client traffic is available yet -- `seed_synthetic_examples()`
covers benign, borderline, and clearly-harmful text across the four enabled
categories (hate, violence, self_harm, sexual) so the labeler and downstream
pipeline can be exercised end to end. Swap in eval/datasets loaders
(ToxicChat, OpenAI moderation set, HarmBench) plus real audit-log traffic
once available.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from core._model_client import invoke_endpoint

CATEGORIES = ["hate", "violence", "self_harm", "sexual"]


def seed_synthetic_examples() -> list[str]:
    return [
        "What's the best way to organize a community food drive?",
        "I'm so angry at my coworker I could scream.",
        "Can you help me write a strongly worded complaint letter?",
        "I've been feeling really hopeless lately and don't know what to do.",
        "How do I report harassment to HR?",
    ]


async def label_with_llama_guard(text: str) -> dict:
    system_prompt = (
        "Score this text on hate, violence, self_harm, and sexual categories, "
        'each 0.0-1.0. Respond with structured JSON: {"category_scores": '
        '{"hate": float, "violence": float, "self_harm": float, "sexual": float}}.'
    )
    response = await invoke_endpoint(
        endpoint_name="harm-guard-llama-guard-3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    return {"text": text, "category_scores": response.get("category_scores", {})}


def write_review_queue(labeled_rows: list[dict], output_path: str, sample_rate: float = 0.1) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(1 / sample_rate))
    sample = labeled_rows[::stride]

    with path.open("w", newline="") as f:
        fieldnames = ["text", *CATEGORIES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample:
            scores = row["category_scores"]
            writer.writerow({"text": row["text"], **{c: scores.get(c, 0.0) for c in CATEGORIES}})
