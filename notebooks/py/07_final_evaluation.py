# %% [markdown]
# # 07 · Final Task-2 evaluation: Direct LLM vs Naive RAG vs Agentic RAG
# **Owner:** A (lead) with B · **Phase 3** · Inputs: frozen runs in `outputs/runs/*.jsonl`
#
# 1. Automatic metrics (behaviour accuracy, escalation P/R/F1, invented-specifics, cost, latency)
# 2. Blinded human rubric on the 100-query subset (2 reviewers) + agreement (kappa)
# 3. LLM-as-a-judge on the same subset → judge-vs-human agreement → judge on the full set
# 4. Headline table + figures for the report and slides

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
sys.path.insert(0, os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebooks" else "."))

import pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from src import config as C
from src.eval.harness import load_run, summarize, escalation_confusion, auto_checks
from src.eval.judge import make_human_sheet, agreement, judge_run, judge_vs_human, score_table
from src.utils import style_axes, savefig

ev = pd.read_parquet(C.EVAL_SETS_FILE)
SYSTEMS = ["direct_llm", "naive_rag", "agentic_full"]          # tags in outputs/runs/
runs = {s: load_run(s) for s in SYSTEMS}
for s, r in runs.items():
    print(s, len(r), "rows")  # every system must cover ALL eval_ids before final numbers are reported

# %% [markdown]
# ## 1. Automatic metrics

# %%
rows = []
for s, r in runs.items():
    t = summarize(r, ev).reset_index(); t.insert(0, "system", s); rows.append(t)
auto = pd.concat(rows); auto.to_csv(C.TABLES / "task2_auto_metrics.csv", index=False)
display(auto.pivot_table(index="eval_set", columns="system", values="behaviour_accuracy").round(3))
esc = pd.concat([escalation_confusion(r, ev).assign(system=s) for s, r in runs.items()])
display(esc); esc.to_csv(C.TABLES / "task2_escalation.csv", index=False)
eff = auto.groupby("system").apply(lambda g: pd.Series({
    "mean_llm_calls": (g["mean_llm_calls"] * g["n"]).sum() / g["n"].sum(),
    "mean_tokens": (g["mean_tokens"] * g["n"]).sum() / g["n"].sum(),
    "total_cost_usd": g["total_cost_usd"].sum(),
    "median_wall_s(avg over sets)": g["median_wall_s"].mean(),
    "invented_specific_rate": (g["invented_specific_rate"] * g["n"]).sum() / g["n"].sum()}), include_groups=False)
display(eff.round(4)); eff.to_csv(C.TABLES / "task2_efficiency.csv")

# %%
pv = auto.pivot_table(index="eval_set", columns="system", values="behaviour_accuracy").reindex(columns=SYSTEMS)
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.26
for i, s in enumerate(SYSTEMS):
    ax.bar([x + (i - 1) * w for x in range(len(pv))], pv[s], width=w - 0.02, color=C.PALETTE[i], label=s)
ax.set_xticks(range(len(pv)), pv.index, rotation=20, fontsize=8)
style_axes(ax, "Correct behaviour (answer / clarify / escalate) by evaluation set", None, "share correct")
ax.set_ylim(0, 1.15); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.legend(frameon=False, ncol=3, loc="upper left")
savefig(fig, "task2_behaviour_by_set"); plt.show()

# %% [markdown]
# ## 2. Blinded human rubric (100 queries × 3 systems = 300 items per reviewer)
# Run the first cell once, share `human_sheet.csv` with both reviewers (NOT `human_key.csv`).
# Each reviewer fills a copy (`reviewer_1.csv`, `reviewer_2.csv`) using the rubric in `prompts/judge_v1.md`.

# %%
subset_ids = pd.read_csv(C.PROCESSED / "human_eval_subset.csv")["eval_id"].tolist()
sheet_path, key_path = C.PROCESSED / "human_sheet.csv", C.PROCESSED / "human_key.csv"
if not sheet_path.exists():
    sheet, key = make_human_sheet(runs, ev, subset_ids)
    sheet.to_csv(sheet_path, index=False); key.to_csv(key_path, index=False)
    print("Share", sheet_path, "with reviewers")

# %%
r1, r2 = C.PROCESSED / "reviewer_1.csv", C.PROCESSED / "reviewer_2.csv"
if r1.exists() and r2.exists():
    a, b = pd.read_csv(r1), pd.read_csv(r2)
    display(agreement(a, b))   # report these kappas; then adjudicate disagreements -> human_final.csv

# %% [markdown]
# ## 3. LLM-as-a-judge (validated against humans, then applied to all queries)

# %%
from src.llm.client import LLMClient
judge_llm = LLMClient()   # ideally set a different/stronger model for the judge via env vars
judged = pd.concat([judge_run(r, ev, judge_llm).assign(system=s) for s, r in runs.items()])
judged.to_csv(C.TABLES / "task2_judge_scores_raw.csv", index=False)
display(score_table(judged, ev))
final_h = C.PROCESSED / "human_final.csv"
if final_h.exists():
    display(judge_vs_human(judged, pd.read_csv(final_h), pd.read_csv(key_path)))

# %% [markdown]
# ## 4. What goes in the report (Model Training & Evaluation, 40 pts)
# - Headline table: system × {behaviour accuracy, escalation F1, judge relevance/coverage/grounded, unsafe rate, cost, latency}
# - Per eval set: where does Agentic RAG beat Naive RAG (b77_real_inscope? bitext_hard? unsupported?) and where not
# - Ablations (notebook 05): which component causes the gain
# - 10-20 qualitative examples with traces; honest negatives
