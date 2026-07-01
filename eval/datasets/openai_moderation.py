"""Loader for OpenAI's moderation eval set (mmathys/openai-moderation-api-evaluation
on HF Hub). Used for harm-category F1 in the benchmark ladder."""
from __future__ import annotations

import pandas as pd

CATEGORIES = ["hate", "violence", "self_harm", "sexual"]


def load_openai_moderation(limit: int | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns: text, hate, violence, self_harm,
    sexual (each 0/1 ground-truth labels)."""
    from datasets import load_dataset

    ds = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
    df = pd.DataFrame(ds)
    df = df.rename(columns={"prompt": "text", "S": "sexual", "H": "hate", "V": "violence", "SH": "self_harm"})
    keep = ["text", *CATEGORIES]
    df = df[[c for c in keep if c in df.columns]]
    if limit:
        df = df.head(limit)
    return df
