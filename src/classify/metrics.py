"""Shared Task-1 metrics so every classifier (TF-IDF, BERT, LLM few-shot) is scored identically."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .. import config as C
from ..utils import parse_flags, savefig


def summary(y_true, y_pred, model: str, eval_set: str, ms_per_query: float | None = None) -> dict:
    return {"model": model, "eval_set": eval_set, "n": len(y_true),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            # macro over the gold labels present in this eval set (BANKING77 covers only a subset of intents)
            "macro_f1": round(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0), 4),
            "ms_per_query": None if ms_per_query is None else round(ms_per_query, 3)}


def per_class(y_true, y_pred) -> pd.DataFrame:
    r = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(r).T.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore").sort_values("f1-score")


def accuracy_by_flag(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Robustness view: accuracy on rows carrying each Bitext flag (typos Z, colloquial Q, ...)."""
    rows = []
    fl = df["flags"].map(parse_flags)
    for f in sorted({x for s in fl for x in s}):
        m = fl.map(lambda s: f in s)
        rows.append({"flag": f, "n": int(m.sum()), "accuracy": round(float((df.loc[m, "intent"] == df.loc[m, pred_col]).mean()), 4)})
    return pd.DataFrame(rows)


def top_confusions(y_true, y_pred, k: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"true": y_true, "pred": y_pred})
    d = d[d["true"] != d["pred"]]
    return d.value_counts().head(k).reset_index(name="count")


def plot_confusion(y_true, y_pred, name: str, title: str):
    labs = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labs, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)), labs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labs)), labs, fontsize=7)
    ax.set_xlabel("predicted", color=C.INK_2)
    ax.set_ylabel("true", color=C.INK_2)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=C.INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="row-normalised share")
    savefig(fig, name)
    plt.close(fig)
    return cm
