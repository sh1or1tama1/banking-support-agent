# %% [markdown]
# # 01 · Data loading & leakage-aware split
# **Owner:** A (project lead) · **Phase 1** · Output: `data/processed/bitext_splits.parquet` (FROZEN after team sign-off)
#
# Steps: load Bitext + BANKING77 → schema audit → near-duplicate grouping → stratified *group* split
# (70/15/15) → leakage audit vs. a random split → manifest.
#
# ⚠️ Run this notebook **once**, commit the parquet + `split_manifest.json`, and never regenerate.
# Every model in the project is trained/evaluated on exactly this split.

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

import pandas as pd
from IPython.display import display
from src import config as C
from src.utils import set_seed
from src.data.load import load_bitext, load_banking77
from src.data.split import assign_groups, make_split, leakage_audit, write_manifest
set_seed()

# %% [markdown]
# ## 1. Load

# %%
bitext = load_bitext()          # Hugging Face in Colab; or set BITEXT_LOCAL to a downloaded parquet/csv
b77 = load_banking77()          # GitHub CSVs (official PolyAI release)
print("Bitext:", bitext.shape, "| BANKING77:", b77.shape)
display(bitext.head(3))
display(b77.groupby("b77_split").size().rename("rows"))

# %% [markdown]
# ## 2. Near-duplicate groups
# Inside each intent, queries whose character n-gram TF-IDF cosine ≥ `NEAR_DUP_THRESHOLD` join the same
# group (greedy leader clustering, so groups cannot chain together). Groups are never split across
# train/val/test.

# %%
grouped = assign_groups(bitext)
gs = grouped.drop_duplicates("group_id")["group_size"]
print(f"threshold={C.NEAR_DUP_THRESHOLD} | rows={len(grouped)} | groups={grouped['group_id'].nunique()} "
      f"| mean group size={gs.mean():.2f} | max={gs.max()} | label-conflict rows={grouped['label_conflict'].sum()}")
display(gs.describe().to_frame("group_size"))
# Sanity check: no group should hold a large share of its intent (that would make stratification impossible)
biggest = (grouped.groupby("intent")["group_size"].max() / grouped.groupby("intent").size()).sort_values(ascending=False)
display(biggest.head(5).to_frame("largest group / intent rows"))

# %% [markdown]
# ## 3. Split + leakage audit

# %%
splits = make_split(grouped)
display(pd.crosstab(splits["split"], splits["category"], margins=True))
audit = leakage_audit(splits)
display(audit.groupby("strategy")["max_sim_to_train"].describe(percentiles=[.5, .9]))

# %%
import matplotlib.pyplot as plt
from src.utils import style_axes, savefig
fig, ax = plt.subplots(figsize=(8, 3.6))
for (name, sub), color in zip(audit.groupby("strategy"), [C.PALETTE[0], C.PALETTE[1]]):
    ax.hist(sub["max_sim_to_train"], bins=40, alpha=0.65, color=color, label=name, density=True)
style_axes(ax, "Leakage audit: each test query's max similarity to any training query", "max cosine (char n-grams)", "density")
ax.legend(frameon=False)
savefig(fig, "split_leakage_audit"); plt.show()

# %% [markdown]
# ## 4. Save & freeze

# %%
cols = ["row_id", "flags", "instruction", "category", "intent", "response", "group_id", "group_size", "label_conflict", "split"]
splits[cols].to_parquet(C.SPLITS_FILE, index=False)
manifest = write_manifest(splits, audit)
manifest

# %% [markdown]
# **Report text to adapt (EDA §Data):** “Because Bitext is template-generated, we grouped near-duplicate
# queries (char 3–5-gram TF-IDF cosine ≥ 0.90) and split whole groups with stratified group K-fold
# (70/15/15, seed 42). Compared with a random split, the share of test queries with a near-copy in
# training fell from X% to Y% (Figure *split_leakage_audit*).” → fill X/Y from the manifest.
