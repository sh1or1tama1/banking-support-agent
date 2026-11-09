# %% [markdown]
# # 05 · Task 2 challenger — Agentic RAG (LangGraph) + ablations
# **Owner:** B (Task-2 lead) · **Phase 2**
#
# The v0 graph in `src/rag/agentic_rag.py` already runs end to end. B's job:
# 1. Read and understand every node (you will defend it in Q&A) — draw it on paper once.
# 2. Run on 20 queries, read the `trace` of failures, improve prompts (`prompts/agentic_*_v2.md` + CHANGELOG).
# 3. Freeze the prompts, run the full eval set, then the ablations below.
#
# | Run tag | What changes | Question it answers |
# |---|---|---|
# | `agentic_full` | everything on | headline challenger |
# | `agentic_no_validation` | `use_validation=False` | does validate→revise→escalate help? |
# | `agentic_no_grading` | `use_grading=False` | does grading + query rewrite help? |
# | `agentic_no_hybrid` | `use_hybrid=False` | does BM25 fusion help (typos, real wording)? |
# | `agentic_no_classifier` | `use_classifier_tool=False` | does the Task-1 tool help vs LLM-predicted intent? |

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
# os.environ["LLM_PROVIDER"] = "openai"; os.environ["LLM_MODEL"] = "<same model as notebook 04>"
sys.path.insert(0, os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebooks" else "."))

import joblib, json, pandas as pd
from IPython.display import display
from src import config as C
from src.rag.retriever import build_kb, DenseRetriever, BM25Retriever
from src.rag.agentic_rag import AgenticRAG, AgenticConfig
from src.classify.tfidf_lr import as_tool
from src.llm.client import LLMClient
from src.eval.harness import run_system, summarize, escalation_confusion, load_run

LIMIT = 20   # development: 20 queries. Set to None for the frozen full run.

splits = pd.read_parquet(C.SPLITS_FILE)
ev = pd.read_parquet(C.EVAL_SETS_FILE)
kb = build_kb(splits)
dense, bm25 = DenseRetriever(kb), BM25Retriever(kb)
# Intent tool = best Task-1 classifier. Start with TF-IDF; swap in BERT once C has saved it.
classifier = as_tool(joblib.load(C.OUTPUTS / "tfidf_lr.joblib"))
llm = LLMClient()

# %% [markdown]
# ## 1. One query, full trace (use this to debug and to make the demo slide)

# %%
agent = AgenticRAG(llm, dense, bm25, classifier=classifier)
out = agent.answer("i lost my card yesterday and i also want to know how to send money to my brother")
print(out["action"], "|", out["answer"][:300])
for step in out["trace"]:
    print("-", step["step"], "→", str(step["output"])[:150])
print({k: out[k] for k in ("n_llm_calls", "input_tokens", "output_tokens", "cost_usd", "wall_time_s")})

# %% [markdown]
# ## 2. Full system + ablations

# %%
variants = {
    "agentic_full": AgenticConfig(),
    "agentic_no_validation": AgenticConfig(use_validation=False),
    "agentic_no_grading": AgenticConfig(use_grading=False),
    "agentic_no_hybrid": AgenticConfig(use_hybrid=False),
    "agentic_no_classifier": AgenticConfig(use_classifier_tool=False),
}
runs = {}
for tag, cfg in variants.items():
    runs[tag] = run_system(AgenticRAG(llm, dense, bm25, classifier=classifier, cfg=cfg, name=tag), ev, tag=tag, limit=LIMIT)

# %%
rows = []
for tag, run in runs.items():
    s = summarize(run, ev).reset_index()
    s.insert(0, "system", tag)
    rows.append(s)
abl = pd.concat(rows)
display(abl.pivot_table(index="eval_set", columns="system", values="behaviour_accuracy").round(3))
display(abl.groupby("system")[["mean_llm_calls", "mean_tokens", "total_cost_usd", "median_wall_s"]].mean().round(3))
abl.to_csv(C.TABLES / "task2_agentic_ablations.csv", index=False)

# %% [markdown]
# ## 3. Failure analysis — read 10 traces where the agent got the behaviour wrong

# %%
full = runs["agentic_full"].merge(ev[["eval_id", "expected_action"]], on="eval_id")
bad = full[full["action"] != full["expected_action"]]
for _, r in bad.head(10).iterrows():
    print(f"\n[{r['eval_set']}] expected={r['expected_action']} got={r['action']}\nQ: {r['query']}\nA: {str(r['answer'])[:200]}")
    for st in r["trace"]:
        print("   -", st["step"], "→", str(st["output"])[:120])
