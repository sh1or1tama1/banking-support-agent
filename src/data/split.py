"""Leakage-aware train / validation / test split for Bitext.

Why: Bitext is synthetic and template-generated, so many instructions are near-copies
("activate my card" / "activate my card pls"). A random split puts near-copies on both sides
and inflates test scores. We therefore

  1. normalise text and group exact duplicates,
  2. inside each intent, greedily cluster near-duplicates (char n-gram TF-IDF cosine >= threshold;
     leader clustering, so clusters cannot chain into one giant blob),
  3. merge groups that share identical normalised text across intents (label conflicts),
  4. split whole groups with StratifiedGroupKFold so every intent appears in every split,
  5. audit leakage: for each test row, the max similarity to any training row,
     compared against a naive random split.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .. import config as C
from ..utils import normalize, save_json, stable_hash


def _char_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)


def _leader_cluster(texts: list[str], threshold: float) -> np.ndarray:
    """Greedy leader clustering: each text joins the first earlier leader with cos >= threshold."""
    if len(texts) == 1:
        return np.zeros(1, dtype=int)
    X = _char_vectorizer().fit_transform(texts)
    sims = (X @ X.T).toarray()
    labels = -np.ones(len(texts), dtype=int)
    leaders: list[int] = []
    for i in range(len(texts)):
        if leaders:
            s = sims[i, leaders]
            j = int(np.argmax(s))
            if s[j] >= threshold:
                labels[i] = labels[leaders[j]]
                continue
        labels[i] = len(leaders)
        leaders.append(i)
    return labels


def assign_groups(df: pd.DataFrame, threshold: float = C.NEAR_DUP_THRESHOLD) -> pd.DataFrame:
    """Add columns: norm_text, group_id, group_size, label_conflict."""
    df = df.copy()
    df["norm_text"] = df["instruction"].map(normalize)

    group_of_norm: dict[tuple[str, str], str] = {}
    for intent, sub in df.groupby("intent", sort=True):
        uniq = sorted(sub["norm_text"].unique())
        labels = _leader_cluster(uniq, threshold)
        for t, lab in zip(uniq, labels):
            group_of_norm[(intent, t)] = f"{intent}::{lab:04d}"
    df["group_id"] = [group_of_norm[(i, t)] for i, t in zip(df["intent"], df["norm_text"])]

    # Same normalised text under different intents = label conflict -> force one group.
    intents_per_text = df.groupby("norm_text")["intent"].nunique()
    conflict_texts = set(intents_per_text[intents_per_text > 1].index)
    df["label_conflict"] = df["norm_text"].isin(conflict_texts)
    if conflict_texts:
        parent: dict[str, str] = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        for _, sub in df[df["label_conflict"]].groupby("norm_text"):
            gs = sorted(sub["group_id"].unique())
            for g in gs[1:]:
                parent[find(g)] = find(gs[0])
        df["group_id"] = df["group_id"].map(find)

    df["group_size"] = df.groupby("group_id")["row_id"].transform("count")
    return df


def make_split(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Assign df['split'] in {train, val, test} by whole groups, stratified by intent."""
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=C.N_SPLIT_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["intent"], groups=df["group_id"])):
        fold[idx] = k
    n_train = round(C.N_SPLIT_FOLDS * C.SPLIT_FRACTIONS["train"])
    n_val = round(C.N_SPLIT_FOLDS * C.SPLIT_FRACTIONS["val"])
    df["split"] = np.where(fold < n_train, "train", np.where(fold < n_train + n_val, "val", "test"))
    return df


def leakage_audit(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """For each test row: max char-TF-IDF cosine to any train row.
    Returns a long DataFrame (strategy, max_sim) comparing our grouped split with a random split."""
    out = []
    random_split = df.copy()
    tr, rest = train_test_split(df.index, test_size=0.30, random_state=seed, stratify=df["intent"])
    random_split["split"] = "train"
    random_split.loc[rest[: len(rest) // 2], "split"] = "test"
    for name, d in (("grouped split (ours)", df), ("random split", random_split)):
        train_txt = d.loc[d["split"] == "train", "norm_text"]
        test_txt = d.loc[d["split"] == "test", "norm_text"]
        vec = _char_vectorizer().fit(train_txt)
        A, B = vec.transform(test_txt), vec.transform(train_txt)
        max_sim = []
        for start in range(0, A.shape[0], 1000):  # chunk to keep memory small
            max_sim.append((A[start:start + 1000] @ B.T).max(axis=1).toarray().ravel())
        out.append(pd.DataFrame({"strategy": name, "max_sim_to_train": np.concatenate(max_sim)}))
    return pd.concat(out, ignore_index=True)


def write_manifest(df: pd.DataFrame, audit: pd.DataFrame | None = None) -> dict:
    counts = df.groupby("split").size().to_dict()
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": C.SEED,
        "near_dup_threshold": C.NEAR_DUP_THRESHOLD,
        "n_rows": int(len(df)),
        "split_counts": {k: int(v) for k, v in counts.items()},
        "split_fractions": {k: round(v / len(df), 4) for k, v in counts.items()},
        "n_groups": int(df["group_id"].nunique()),
        "n_label_conflict_rows": int(df["label_conflict"].sum()),
        "intents_missing_from_split": {
            s: sorted(set(df["intent"]) - set(df.loc[df["split"] == s, "intent"])) for s in ("train", "val", "test")
        },
        "test_ids_hash": stable_hash(sorted(df.loc[df["split"] == "test", "row_id"])),
    }
    if audit is not None:
        for name, sub in audit.groupby("strategy"):
            manifest[f"leakage::{name}"] = {
                "median_max_sim": round(float(sub["max_sim_to_train"].median()), 4),
                "share_test_rows_with_near_copy_in_train(>=0.9)": round(float((sub["max_sim_to_train"] >= 0.9).mean()), 4),
            }
    save_json(manifest, C.SPLIT_MANIFEST)
    return manifest
