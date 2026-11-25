# %% [markdown]
# # 05 · Task 2 challenger — Agentic RAG (LangGraph) + ablations
# **Owner:** B (Task-2 lead) · **Phase 2**
#
# Prompts are **frozen at v2** (2026-11-23). They were tuned on the 48-query dev set with
# `scripts/dev_agentic.py`, never on the eval set; the record is in `prompts/CHANGELOG.md`.
# The intent tool and its threshold were chosen on dev data with `scripts/choose_intent_tool.py`.
# This notebook runs the frozen system on the full eval set, then the ablations below.
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
    REPO_URL = "https://github.com/sh1or1tama1/banking-support-agent.git"
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

LIMIT = int(os.getenv("LIMIT")) if os.getenv("LIMIT") else None   # None = full eval set
PROMPTS = "v2"          # frozen, see prompts/CHANGELOG.md
TOOL, THRESHOLD = "bert", 0.99   # chosen on dev data: outputs/tables/intent_tool_threshold_sweep.csv

splits = pd.read_parquet(C.SPLITS_FILE)
ev = pd.read_parquet(C.EVAL_SETS_FILE)
kb = build_kb(splits)
dense, bm25 = DenseRetriever(kb), BM25Retriever(kb)
# Intent tool = Task-1 classifier called by the agent (see TOOL above)
if TOOL == "bert":
    from src.classify.bert import BertIntent
    classifier = BertIntent().as_tool()
else:
    classifier = as_tool(joblib.load(C.OUTPUTS / "tfidf_lr.joblib"))
base = dict(intent_conf_threshold=THRESHOLD)
llm = LLMClient()

# %% [markdown]
# ## 1. One query, full trace (use this to debug and to make the demo slide)

# %%
agent = AgenticRAG(llm, dense, bm25, classifier=classifier, cfg=AgenticConfig(**base), prompt_version=PROMPTS)
out = agent.answer("i lost my card yesterday and i also want to know how to send money to my brother")
print(out["action"], "|", out["answer"][:300])
for step in out["trace"]:
    print("-", step["step"], "→", str(step["output"])[:150])
print({k: out[k] for k in ("n_llm_calls", "input_tokens", "output_tokens", "cost_usd", "wall_time_s")})

# %% [markdown]
# ## 2. Full system + ablations

# %%
variants = {
    "agentic_full": AgenticConfig(**base),
    "agentic_no_validation": AgenticConfig(**base, use_validation=False),
    "agentic_no_grading": AgenticConfig(**base, use_grading=False),
    "agentic_no_hybrid": AgenticConfig(**base, use_hybrid=False),
    "agentic_no_classifier": AgenticConfig(**base, use_classifier_tool=False),
}
runs = {}
for tag, cfg in variants.items():
    runs[tag] = run_system(AgenticRAG(llm, dense, bm25, classifier=classifier, cfg=cfg, name=tag, prompt_version=PROMPTS),
                           ev, tag=tag, limit=LIMIT)

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
