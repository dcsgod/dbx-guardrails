"""Loader for HarmBench (walledai/HarmBench on HF Hub) -- primarily used to
eval the injection-gate / jailbreak-resistance rung of the benchmark ladder.
"""
from __future__ import annotations

import pandas as pd


def load_harmbench(subset: str = "standard", limit: int | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns: text, category. All rows are
    adversarial by construction (label=1 for injection/harm benchmarking)."""
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", subset, split="train")
    df = pd.DataFrame(ds)
    df = df.rename(columns={"prompt": "text"})
    df["label"] = 1
    if limit:
        df = df.head(limit)
    return df[["text", "category", "label"]] if "category" in df.columns else df[["text", "label"]]
