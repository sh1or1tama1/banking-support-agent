"""Loading the two datasets into tidy pandas DataFrames.

Bitext  : columns -> row_id, flags, instruction, category, intent, response
BANKING77: columns -> b77_id, text, b77_intent, b77_split
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config as C

BITEXT_COLS = ["flags", "instruction", "category", "intent", "response"]


def load_bitext(local_path: str | Path | None = None) -> pd.DataFrame:
    """Load Bitext Retail Banking (25,545 rows on the official card).

    Order of preference: explicit `local_path` -> config.BITEXT_LOCAL -> cached raw parquet ->
    Hugging Face Hub (needs internet; works in Colab).
    """
    cache = C.RAW / "bitext_raw.parquet"
    path = Path(local_path or C.BITEXT_LOCAL) if (local_path or C.BITEXT_LOCAL) else None
    if path is not None:
        path = path if path.is_absolute() else C.ROOT / path
        df =pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    elif cache.exists():
        df = pd.read_parquet(cache)
    else:
        from datasets import load_dataset  # imported lazily: not needed offline

        df = load_dataset(C.BITEXT_HF_ID)["train"].to_pandas()

    # The card calls the first field "flags"; some mirrors call it "tags".
    df = df.rename(columns={"tags": "flags"})
    missing = [c for c in BITEXT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Bitext file is missing columns {missing}; found {list(df.columns)}")
    df = df[BITEXT_COLS].copy()
    for c in BITEXT_COLS:
        df[c] = df[c].astype("string")
    if "row_id" not in df.columns:
        df.insert(0, "row_id", [f"bx_{i:05d}" for i in range(len(df))])
    if not cache.exists() and path is None:
        df.to_parquet(cache, index=False)
    return df


def load_banking77() -> pd.DataFrame:
    """Load BANKING77 (PolyAI) from the official GitHub CSVs (10,003 train / 3,080 test)."""
    cache = C.RAW / "banking77_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for split, url in C.BANKING77_URLS.items():
        part = pd.read_csv(url)
        part["b77_split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True).rename(columns={"category": "b77_intent"})
    df.insert(0, "b77_id", [f"b77_{i:05d}" for i in range(len(df))])
    df.to_parquet(cache, index=False)
    return df
