"""Loader for the ToxicChat eval set (lmsys/toxic-chat on HF Hub). Used as
one of the harm/injection benchmark-ladder rungs' evaluation data.
"""
from __future__ import annotations

import pandas as pd


def load_toxicchat(split: str = "test", limit: int | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns: text, toxic (0/1), jailbreaking (0/1).
    Downloads lazily via `datasets` -- not required at import time so this
    module can be imported in environments without network access.
    """
    from datasets import load_dataset

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
    df = pd.DataFrame(ds)
    df = df.rename(columns={"user_input": "text"})[["text", "toxicity", "jailbreaking"]]
    df = df.rename(columns={"toxicity": "toxic"})
    if limit:
        df = df.head(limit)
    return df
