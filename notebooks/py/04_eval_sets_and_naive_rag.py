# %% [markdown]
# # 04 · Frozen evaluation set + Naive RAG baseline (+ Direct LLM reference)
# **Owner:** A builds the eval set (Phase 1); B owns the RAG systems from Phase 2.
#
# 1. Build `data/processed/eval_sets.parquet` (FROZEN; every Task-2 system is scored on it)
# 2. Build the knowledge base from the TRAIN split only
# 3. Retrieval sanity check (intent-proxy Recall@1/@5, MRR@5)
# 4. Run Direct LLM (reference) and Naive RAG (baseline) → `outputs/runs/*.jsonl`
#
# LLM setup (Colab → 🔑 Secrets): `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, then set
# `LLM_PROVIDER` and `LLM_MODEL` below. Start with `LIMIT = 20` to check cost before the full run.

# %%
import sys, os, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:  # Colab: clone the team repo once per session, install requirements, load API keys from Secrets
    REPO_URL = "https://github.com/<your-team>/banking-support-agent.git"   # <- set once
    if not os.path.exists("/content/banking-support-agent"):
        subprocess.run(["git", "clone", REPO_URL, "/content/banking-support-agent"], check=True)
    os.chdir("/content/banking-support-agent")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    from google.colab import userdata
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        try: os.environ[k] = userdata.get(k)
        except Exception: pass
# os.environ["LLM_PROVIDER"] = "openai"      # or "anthropic"
# os.environ["LLM_MODEL"] = "<cheapest current small model>"
# os.environ["PRICE_IN_PER_M"], os.environ["PRICE_OUT_PER_M"] = "<usd>", "<usd>"   # from the pricing page
sys.path.insert(0, os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebooks" else "."))

import pandas as pd
from IPython.display import display
from src import config as C
from src.data.load import load_banking77
from src.data.eval_sets import build_eval_sets, human_eval_subset
from src.rag.retriever import build_kb, DenseRetriever, BM25Retriever, retrieval_proxy_metrics
from src.rag.systems import DirectLLM, NaiveRAG
from src.llm.client import LLMClient
from src.eval.harness import run_system, summarize, escalation_confusion

LIMIT = None   # e.g. 20 for a first, cheap test run

# %% [markdown]
# ## 1. Frozen evaluation set

# %%
splits = pd.read_parquet(C.SPLITS_FILE)
if C.EVAL_SETS_FILE.exists():
    ev = pd.read_parquet(C.EVAL_SETS_FILE)          # already frozen -> reuse
else:
    ev = build_eval_sets(splits, load_banking77())
    ev.to_parquet(C.EVAL_SETS_FILE, index=False)
    human_eval_subset(ev).to_csv(C.PROCESSED / "human_eval_subset.csv", index=False)
display(ev.groupby(["eval_set", "expected_action"]).size().rename("queries").reset_index())
display(ev.groupby("eval_set").head(2)[["eval_set", "query", "gold_intent", "expected_action"]])

# %% [markdown]
# ## 2. Knowledge base (train split only) and retrievers

# %%
kb = build_kb(splits)
print("KB documents:", len(kb), "| version:", kb.attrs.get("version"))
dense = DenseRetriever(kb)
bm25 = BM25Retriever(kb)
print("embedding backend:", dense.embedder.backend)

# %%
answerable = ev[ev["expected_action"] == "answer"]
rows = []
for name, r in (("dense", dense), ("bm25", bm25)):
    for s, sub in answerable.groupby("eval_set"):
        rows.append({"retriever": name, "eval_set": s, **retrieval_proxy_metrics(r, sub, k=5)})
retr = pd.DataFrame(rows).round(3); display(retr)
retr.to_csv(C.TABLES / "retrieval_proxy_metrics.csv", index=False)
# Read: does the top document have the right intent? Low Recall@1 on b77_real_inscope / bitext_hard
# is exactly where an agentic system (query rewrite, hybrid search, grading) should help.

# %% [markdown]
# ## 3. Run Direct LLM (reference) and Naive RAG (baseline)

# %%
llm = LLMClient()
print("provider:", llm.provider, "| model:", llm.model)
systems = [DirectLLM(llm), NaiveRAG(llm, dense, k=C.RAG_TOP_K)]
runs = {s.name: run_system(s, ev, limit=LIMIT) for s in systems}

# %%
for name, run in runs.items():
    print(f"\n=== {name} ===")
    display(summarize(run, ev))
    display(escalation_confusion(run, ev))

# %%
# Look at a few outputs side by side
cmp = ev[["eval_id", "eval_set", "query"]].merge(
    runs["direct_llm"][["eval_id", "action", "answer"]].rename(columns={"action": "direct_action", "answer": "direct_answer"}), on="eval_id"
).merge(runs["naive_rag"][["eval_id", "action", "answer", "contexts"]], on="eval_id")
display(cmp.groupby("eval_set").head(1))
