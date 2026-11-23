"""Choose the Agentic RAG intent tool and its confidence threshold WITHOUT the eval set or an LLM.

For each tool (TF-IDF, and BERT if outputs/bert_intent exists) and threshold, measure whether the top retrieved
documents have the gold intent when retrieval is filtered by the tool's intent (hybrid dense+BM25, k=3).
Data: 800 VAL queries (synthetic) and 800 BANKING77 *train* queries (real, mapped intents).

    python scripts/choose_intent_tool.py      -> outputs/tables/intent_tool_threshold_sweep.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data.eval_sets import banking77_mapped
from src.data.load import load_banking77
from src.rag.retriever import BM25Retriever, DenseRetriever, build_kb, hybrid_search

THRESHOLDS = [None, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def main():
    splits = pd.read_parquet(C.SPLITS_FILE)
    val = splits[splits["split"] == "val"].sample(800, random_state=C.SEED)
    b77 = banking77_mapped(load_banking77(), split="train").sample(800, random_state=C.SEED)
    sets = {"val (synthetic)": (val["instruction"].tolist(), val["intent"].tolist()),
            "b77 train (real)": (b77["text"].tolist(), b77["gold_intent"].tolist())}
    tools = {}
    from src.classify.tfidf_lr import as_tool
    tools["tfidf"] = as_tool(joblib.load(C.OUTPUTS / "tfidf_lr.joblib"))
    if (C.OUTPUTS / "bert_intent" / "config.json").exists():
        from src.classify.bert import BertIntent
        tools["bert"] = BertIntent().as_tool()

    kb = build_kb(splits)
    dense, bm25 = DenseRetriever(kb), BM25Retriever(kb)
    unfiltered = {name: [[h["intent"] for h in hybrid_search([dense, bm25], q, 3)] for q in qs] for name, (qs, _) in sets.items()}
    rows = []
    for tname, tool in tools.items():
        for name, (qs, gold) in sets.items():
            preds = [tool(q) for q in qs]
            acc = np.mean([p == g for (p, _), g in zip(preds, gold)])
            for th in THRESHOLDS:
                r1 = r3 = used = 0
                for q, g, (p, c), base in zip(qs, gold, preds, unfiltered[name]):
                    if th is not None and c >= th:
                        used += 1
                        ints = [h["intent"] for h in hybrid_search([dense, bm25], q, 3, p)]
                    else:
                        ints = base
                    r1 += ints[:1] == [g]; r3 += g in ints
                rows.append({"tool": tname, "set": name, "tool_accuracy": round(acc, 3),
                             "threshold": "no filter" if th is None else th, "filter_used": round(used / len(qs), 3),
                             "recall@1": round(r1 / len(qs), 3), "recall@3": round(r3 / len(qs), 3)})
            print(f"done {tname} / {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "intent_tool_threshold_sweep.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
