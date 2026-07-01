"""Synthetic multi-turn benign-follow-up set: the regression dataset for the
exact false-positive class this accelerator is designed to fix (a terse
follow-up that reads as out-of-scope in isolation but is a clear
continuation of an in-scope thread). Every row is ground-truth in_scope=True.

This is the dataset the scope-guard benchmark rung (eval/run_benchmark.py)
runs against to compute false-positive rate -- the single metric that
matters most for not annoying real users. Seeded here since no real
multi-turn audit-log traffic exists yet; expand with real incidents logged
to the audit table over time (project.md section 7).
"""
from __future__ import annotations

import pandas as pd


def load_benign_followups() -> pd.DataFrame:
    rows = [
        {
            "history": [
                {"role": "user", "content": "Can you pull up the competitive news brief for this week?"},
                {"role": "assistant", "content": "Here's the brief: three competitors adjusted pricing on electronics..."},
            ],
            "text": "Yes.",
            "in_scope": True,
        },
        {
            "history": [
                {"role": "user", "content": "How did our electronics category do against key competitors this year?"},
                {"role": "assistant", "content": "Overall share held steady, with softness in gaming consoles."},
            ],
            "text": "How many Nintendo Switch 2 consoles did we sell in the last year?",
            "in_scope": True,
        },
        {
            "history": [
                {"role": "user", "content": "What's our current on-shelf availability for produce?"},
                {"role": "assistant", "content": "OSA for produce is at 94% this week, down 2pts from last."},
            ],
            "text": "and last year?",
            "in_scope": True,
        },
        {
            "history": [
                {"role": "user", "content": "Summarize NPS trends for the last quarter."},
                {"role": "assistant", "content": "NPS rose 3 points, driven by improved checkout wait times."},
            ],
            "text": "competitive news brief?",
            "in_scope": True,
        },
        {
            "history": [],
            "text": "Yes.",
            "in_scope": None,  # ambiguous with zero history -- not scored for false-positive rate
        },
    ]
    return pd.DataFrame(rows)
