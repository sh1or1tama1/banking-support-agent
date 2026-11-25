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
    REPO_URL = "https://github.com/sh1or1tama1/banking-support-agent.git"
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
from src.eval.judge import make_human_sheet, agreement, judge_run, judge_vs_human, score_table, disagreements, merge_final, load_ratings
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
# Each reviewer scores the 300 replies in the Rating Desk page (or in a copy of the CSV) and saves
# `reviewer_1.csv` (Ryusei) / `reviewer_2.csv` (Szuyu) into data/processed/. Rubric: `prompts/judge_v1.md`.

# %%
subset_ids = pd.read_csv(C.PROCESSED / "human_eval_subset.csv")["eval_id"].tolist()
sheet_path, key_path = C.PROCESSED / "human_sheet.csv", C.PROCESSED / "human_key.csv"
if not sheet_path.exists():
    sheet, key = make_human_sheet(runs, ev, subset_ids)
    sheet.to_csv(sheet_path, index=False); key.to_csv(key_path, index=False)
    print("Share", sheet_path, "with reviewers")

# %%
r1, r2 = C.PROCESSED / "reviewer_1.csv", C.PROCESSED / "reviewer_2.csv"   # Ryusei, Szuyu
final_h = C.PROCESSED / "human_final.csv"
if r1.exists() and r2.exists():
    a, b = load_ratings(r1), load_ratings(r2)   # works for full sheets and for Rating Desk CSVs
    kappa = agreement(a, b); display(kappa); kappa.to_csv(C.TABLES / "task2_human_kappa.csv", index=False)
    dis = disagreements(a, b)                     # one row per (item, rubric field) where the raters differ
    dis.to_csv(C.PROCESSED / "human_disagreements.csv", index=False)
    print(len(dis), "disagreements -> both raters fill the `agreed` column together in human_disagreements.csv")
    resolved = C.PROCESSED / "human_disagreements_resolved.csv"
    if resolved.exists():
        merge_final(a, pd.read_csv(resolved)).to_csv(final_h, index=False)
        print("wrote", final_h)

# %% [markdown]
# ## 3. LLM-as-a-judge: Claude Sonnet 5.5 (a different provider and a stronger model than the generator)
# The generator is OpenAI `gpt-5.4-nano`, so judging with Anthropic avoids self-preference bias.
# Needs `ANTHROPIC_API_KEY` (Colab Secrets or env). Expected cost: about $1 for the 300-item subset and about
# $4 more for all 436 × 3 answers. Every call is cached, so a re-run is free.
# Step 3a judges the human subset only and checks agreement with humans; only if agreement is acceptable
# (kappa ≥ 0.4 on most items) run 3b on the full set and report judge scores for every query.

# %%
from src.llm.client import LLMClient
judge_llm = LLMClient(provider=os.getenv("JUDGE_PROVIDER", "anthropic"),
                      model=os.getenv("JUDGE_MODEL", "claude-sonnet-5-5"),
                      price_in_per_m=float(os.getenv("JUDGE_PRICE_IN_PER_M", 2.0)),
                      price_out_per_m=float(os.getenv("JUDGE_PRICE_OUT_PER_M", 10.0)),
                      max_tokens=300)
print("judge:", judge_llm.provider, judge_llm.model)
# Without a key the judge cells are skipped, so sections 1-2 can run first (e.g. before the API key exists).
HAVE_JUDGE = judge_llm.provider == "mock" or bool(os.getenv("ANTHROPIC_API_KEY" if judge_llm.provider == "anthropic" else "OPENAI_API_KEY"))
print("judge cells will run" if HAVE_JUDGE else "No judge API key found: skipping section 3 (set ANTHROPIC_API_KEY, then re-run)")

# %% 3a. judge the human-rating subset (300 answers)
subset_runs = {s: r[r["eval_id"].isin(subset_ids)] for s, r in runs.items()}
judged_sub = None if not HAVE_JUDGE else pd.concat([judge_run(r, ev, judge_llm).assign(system=s) for s, r in subset_runs.items()])
if judged_sub is not None:
    judged_sub.to_csv(C.TABLES / "task2_judge_scores_subset.csv", index=False)
if judged_sub is not None and final_h.exists() and key_path.exists():   # key_path = human_key.csv, kept by Ziming until rating is finished
    jh = judge_vs_human(judged_sub, pd.read_csv(final_h), pd.read_csv(key_path))
    display(jh); jh.to_csv(C.TABLES / "task2_judge_vs_human.csv", index=False)

# %% 3b. judge every answer (436 × 3) once 3a looks acceptable
RUN_FULL_JUDGE = HAVE_JUDGE and os.getenv("RUN_FULL_JUDGE", "0") == "1"
if RUN_FULL_JUDGE:
    judged = pd.concat([judge_run(r, ev, judge_llm).assign(system=s) for s, r in runs.items()])
    judged.to_csv(C.TABLES / "task2_judge_scores_raw.csv", index=False)
    st = score_table(judged, ev); display(st); st.to_csv(C.TABLES / "task2_judge_scores_by_set.csv")
    overall = judged.groupby("system")[["relevance", "step_coverage", "clarity", "grounded", "unsafe", "behaviour_ok"]] \
                    .apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean()).round(3)
    display(overall); overall.to_csv(C.TABLES / "task2_judge_scores_overall.csv")

# %% [markdown]
# ## 4. What goes in the report (Model Training & Evaluation, 40 pts)
# - Headline table: system × {behaviour accuracy, escalation F1, judge relevance/coverage/grounded, unsafe rate, cost, latency}
# - Per eval set: where does Agentic RAG beat Naive RAG (b77_real_inscope? bitext_hard? unsupported?) and where not
# - Ablations (notebook 05): which component causes the gain
# - 10-20 qualitative examples with traces; honest negatives
