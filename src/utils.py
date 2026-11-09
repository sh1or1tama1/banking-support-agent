"""Small shared helpers: text normalisation, Bitext flag parsing, seeding, plotting style."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config as C

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

# Bitext "flags" letters -> meaning (from the dataset card)
FLAG_MEANING = {
    "M": "morphological variation",
    "L": "semantic/lexical variation",
    "B": "basic syntax",
    "I": "interrogative",
    "C": "coordinated (multi-part)",
    "N": "negation",
    "P": "polite",
    "Q": "colloquial",
    "W": "offensive",
    "K": "keyword mode",
    "E": "abbreviations",
    "Z": "errors & typos",
}


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize(text: str) -> str:
    """Normalisation used ONLY for duplicate detection / grouping (never fed to models)."""
    text = PLACEHOLDER_RE.sub(" ph ", str(text).lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_flags(flags: str) -> set[str]:
    return {ch for ch in str(flags or "").upper() if ch in FLAG_MEANING}


def placeholders(text: str) -> list[str]:
    return [m.strip() for m in PLACEHOLDER_RE.findall(str(text))]


def stable_hash(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def sample_per_group(df, col: str, n: int, seed: int = C.SEED):
    """Up to n random rows per group (pandas-version-safe; keeps all columns)."""
    return df.sample(frac=1.0, random_state=seed).groupby(col, sort=True).head(n).sort_index()


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def style_axes(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    """Recessive grid/axes, text in ink colours (not series colours)."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C.GRID)
    ax.tick_params(colors=C.INK_2, labelsize=9)
    ax.grid(axis="y", color=C.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=C.INK, fontsize=12, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=C.INK_2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=C.INK_2)
    return ax


def savefig(fig, name: str):
    path = C.FIGS / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return path
