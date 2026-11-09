"""Knowledge base + retrievers.

Knowledge base (KB): built ONLY from the Bitext TRAIN split (never val/test -> no answer leakage).
One KB document = one (customer question, reference answer) pair, de-duplicated by normalised
question. We index the question text (question-to-question matching, standard for FAQ-style KBs)
and return the paired answer as context.

Retrievers (same interface: .search(query, k, intent=None) -> list[dict]):
  DenseRetriever : sentence embeddings + cosine (numpy; FAISS optional)  <- Naive RAG baseline
  BM25Retriever  : lexical BM25 (rank_bm25)                               <- for hybrid search
  hybrid_search  : Reciprocal Rank Fusion of several retrievers           <- Agentic RAG may use
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..embeddings import Embedder
from ..utils import normalize, stable_hash


def build_kb(splits: pd.DataFrame) -> pd.DataFrame:
    train = splits[splits["split"] == "train"].copy()
    train["norm_q"] = train["instruction"].map(normalize)
    kb = train.drop_duplicates("norm_q").rename(columns={"instruction": "question", "response": "answer"})
    kb = kb[["row_id", "intent", "category", "question", "answer"]].rename(columns={"row_id": "doc_id"})
    kb = kb.reset_index(drop=True)
    kb.attrs["version"] = "kb_" + stable_hash(kb["doc_id"])
    return kb


def _hits(kb: pd.DataFrame, idx, scores) -> list[dict]:
    return [
        {"doc_id": kb.at[i, "doc_id"], "intent": kb.at[i, "intent"], "question": kb.at[i, "question"],
         "answer": kb.at[i, "answer"], "score": float(s)}
        for i, s in zip(idx, scores)
    ]


class DenseRetriever:
    name = "dense"

    def __init__(self, kb: pd.DataFrame, embedder: Embedder | None = None):
        self.kb = kb
        self.embedder = embedder or Embedder()
        self.embedder.fit(kb["question"].tolist())  # no-op for SBERT; fits the offline fallback on KB only
        self.emb = self.embedder.encode(kb["question"].tolist())

    def search(self, query: str, k: int = C.RAG_TOP_K, intent: str | None = None) -> list[dict]:
        q = self.embedder.encode([query])[0]
        scores = self.emb @ q
        if intent is not None:  # metadata filter (used by the agentic system, not by the baseline)
            scores = np.where(self.kb["intent"].to_numpy() == intent, scores, -np.inf)
        idx = np.argsort(-scores)[:k]
        idx = [i for i in idx if np.isfinite(scores[i])]
        return _hits(self.kb, idx, scores[idx])


class BM25Retriever:
    name = "bm25"

    def __init__(self, kb: pd.DataFrame):
        from rank_bm25 import BM25Okapi

        self.kb = kb
        self.bm25 = BM25Okapi([normalize(q).split() for q in kb["question"]])

    def search(self, query: str, k: int = C.RAG_TOP_K, intent: str | None = None) -> list[dict]:
        scores = np.asarray(self.bm25.get_scores(normalize(query).split()), dtype=float)
        if intent is not None:
            scores = np.where(self.kb["intent"].to_numpy() == intent, scores, -np.inf)
        idx = np.argsort(-scores)[:k]
        idx = [i for i in idx if np.isfinite(scores[i])]
        return _hits(self.kb, idx, scores[idx])


def hybrid_search(retrievers, query: str, k: int = C.RAG_TOP_K, intent: str | None = None,
                  depth: int = 20, rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (rrf_k + rank_r(d))."""
    fused: dict[str, dict] = {}
    for r in retrievers:
        for rank, h in enumerate(r.search(query, depth, intent), start=1):
            d = fused.setdefault(h["doc_id"], {**h, "score": 0.0})
            d["score"] += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda h: -h["score"])[:k]


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (doc {h['doc_id']}, intent: {h['intent']})\nQ: {h['question']}\nA: {h['answer']}"
        for i, h in enumerate(hits, start=1)
    )


def retrieval_proxy_metrics(retriever, queries: pd.DataFrame, k: int = 5) -> dict:
    """Weak relevance proxy: is a document of the gold intent in the top-k? (Recall@1/@k, MRR@k)."""
    r1, rk, rr = [], [], []
    for q, gold in zip(queries["query"], queries["gold_intent"]):
        intents = [h["intent"] for h in retriever.search(q, k)]
        r1.append(intents[:1] == [gold])
        rk.append(gold in intents)
        rr.append(1 / (intents.index(gold) + 1) if gold in intents else 0.0)
    return {"recall@1": float(np.mean(r1)), f"recall@{k}": float(np.mean(rk)), f"mrr@{k}": float(np.mean(rr)), "n": len(r1)}
