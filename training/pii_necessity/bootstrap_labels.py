"""Bootstraps necessity labels for entities detected by presidio, using an
LLM judge, then writes a human-review queue for a sample.

No real client traffic is available yet, so `seed_synthetic_examples()`
generates labeled examples covering the common necessity patterns (an SSN in
a support ticket is rarely necessary to answer a question; a phone number the
user is explicitly asking to have updated *is* necessary). Swap in real
traffic by replacing `load_unlabeled_examples()` once the audit log has
volume -- nothing downstream needs to change.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from core._model_client import invoke_endpoint
from core.pii_guard import detect_entities
from config.schema import ProjectConfig

REVIEW_QUEUE_SAMPLE_RATE = 0.1


def seed_synthetic_examples() -> list[dict]:
    return [
        {"text": "Can you update my phone number to 555-123-4567?", "task_intent": "update_contact_info"},
        {"text": "My SSN is 123-45-6789, can you look up my order?", "task_intent": "order_lookup"},
        {"text": "Email me the invoice at jane.doe@example.com", "task_intent": "send_invoice"},
        {"text": "I think John Smith from accounting sent this by mistake.", "task_intent": "general_question"},
        {"text": "For verification, my SSN ends in 6789.", "task_intent": "identity_verification"},
    ]


async def label_necessity(text: str, task_intent: str, project_config: ProjectConfig) -> list[dict]:
    entities = detect_entities(text, project_config)
    if not entities:
        return []

    system_prompt = (
        "For each entity below, judge whether it is NECESSARY for the "
        "assistant to see it in order to fulfill the stated task_intent. "
        'Respond with structured JSON: {"labels": [{"entity_id": str, '
        '"necessary": bool, "confidence": float}]}.'
    )
    entity_descriptions = [
        {"entity_id": f"{e.type}:{e.start}:{e.end}", "type": e.type, "text": e.text} for e in entities
    ]
    user_content = json.dumps({"text": text, "task_intent": task_intent, "entities": entity_descriptions})

    response = await invoke_endpoint(
        endpoint_name="pii-necessity-bootstrap-labeler",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.get("labels", [])


def write_review_queue(labeled_rows: list[dict], output_path: str) -> None:
    """Writes a sample of bootstrap labels to CSV for human QA before they
    feed the distillation training run. Swap for a Delta table write once
    this runs against real UC infra."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "task_intent", "entity_id", "necessary", "confidence"])
        writer.writeheader()
        for row in labeled_rows:
            writer.writerow(row)
