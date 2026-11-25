"""Prompt-tuning loop for the Agentic RAG on the DEV set (never on eval_sets.parquet).

    python scripts/dev_agentic.py --prompts v1 --tool tfidf          # run + summary + failure traces
    python scripts/dev_agentic.py --prompts v2 --tool bert --naive   # also run Naive RAG on the dev set

Runs are written to outputs/runs/dev/<tag>.jsonl (separate from the official runs).
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd

from src import config as C
from src.eval.harness import behaviour_correct, escalation_confusion, run_system, summarize
from src.llm.client import LLMClient
from src.rag.agentic_rag import AgenticConfig, AgenticRAG
from src.rag.retriever import BM25Retriever, DenseRetriever, build_kb
from src.rag.systems import NaiveRAG


def load_tool(name: str):
    if name == "tfidf":
        from src.classify.tfidf_lr import as_tool
        return as_tool(joblib.load(C.OUTPUTS / "tfidf_lr.joblib"))
    if name == "bert":
        from src.classify.bert import BertIntent
        return BertIntent().as_tool()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="v1")
    ap.add_argument("--tool", default="tfidf", choices=["tfidf", "bert"])
    ap.add_argument("--threshold", type=float, default=AgenticConfig().intent_conf_threshold)
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--show", type=int, default=30, help="failure traces to print")
    a = ap.parse_args()

    (C.RUNS / "dev").mkdir(exist_ok=True)
    dev = pd.read_parquet(C.PROCESSED / "dev_queries.parquet")
    kb = build_kb(pd.read_parquet(C.SPLITS_FILE))
    dense, bm25 = DenseRetriever(kb), BM25Retriever(kb)
    llm = LLMClient()
    systems = {}
    if a.naive:
        systems["dev/naive_rag"] = NaiveRAG(llm, dense)
    cfg = AgenticConfig(intent_conf_threshold=a.threshold)
    tag = f"dev/agentic_{a.prompts}_{a.tool}_t{a.threshold:g}"
    systems[tag] = AgenticRAG(llm, dense, bm25, classifier=load_tool(a.tool), cfg=cfg, prompt_version=a.prompts, name=tag)

    pd.set_option("display.width", 220)
    for tag, s in systems.items():
        run = run_system(s, dev, tag=tag)
        d = run.merge(dev[["eval_id", "expected_action", "gold_intent"]], on="eval_id")
        d["ok"] = [behaviour_correct(x, e) for x, e in zip(d["action"], d["expected_action"])]
        print(f"\n===== {tag}: behaviour accuracy {d['ok'].mean():.3f} | cost ${run['cost_usd'].sum():.3f} | "
              f"mean LLM calls {run['n_llm_calls'].mean():.1f} | median {run['wall_time_s'].median():.1f}s")
        print(summarize(run, dev)[["n", "behaviour_accuracy", "answer_rate", "escalate_rate", "clarify_rate"]].to_string())
        print(escalation_confusion(run, dev).to_string(index=False))
        if "agentic" in tag:
            for _, r in d[~d["ok"]].head(a.show).iterrows():
                print(f"\n--- [{r.eval_set}] expected={r.expected_action} got={r.action} gold={r.gold_intent}\n    Q: {r.query}")
                for st in r["trace"]:
                    print(f"    - {st['step']}: {str(st['output'])[:260]}")
                print(f"    A: {str(r.answer)[:200]}")


if __name__ == "__main__":
    main()
