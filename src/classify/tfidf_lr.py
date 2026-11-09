"""Task 1 BASELINE: TF-IDF (word 1-2 grams + char 3-5 grams) + Logistic Regression.

Hyper-parameter C is chosen on the VALIDATION split only; the test split is touched once.
The fitted model also serves as the Agentic RAG's intent tool (predict_with_conf).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .. import config as C


def make_pipeline(c: float = 10.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("tfidf", feats), ("lr", LogisticRegression(C=c, max_iter=3000))])


def tune_and_fit(train: pd.DataFrame, val: pd.DataFrame, grid=(0.1, 1, 10, 100)) -> tuple[Pipeline, pd.DataFrame]:
    rows = []
    for c in grid:
        m = make_pipeline(c).fit(train["instruction"], train["intent"])
        p = m.predict(val["instruction"])
        rows.append({"C": c, "val_accuracy": accuracy_score(val["intent"], p),
                     "val_macro_f1": f1_score(val["intent"], p, average="macro")})
    res = pd.DataFrame(rows)
    best_c = res.sort_values(["val_macro_f1", "C"], ascending=[False, True]).iloc[0]["C"]
    model = make_pipeline(best_c).fit(train["instruction"], train["intent"])
    return model, res


def predict_timed(model: Pipeline, texts: list[str]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    pred = model.predict(texts)
    return pred, (time.perf_counter() - t0) / max(1, len(texts)) * 1000  # ms per query


def as_tool(model: Pipeline):
    """Wrap for AgenticRAG: query -> (intent, confidence)."""
    def tool(query: str) -> tuple[str, float]:
        proba = model.predict_proba([query])[0]
        i = int(np.argmax(proba))
        return model.classes_[i], float(proba[i])
    return tool
