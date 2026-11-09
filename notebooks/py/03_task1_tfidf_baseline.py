# %% [markdown]
# # 03 · Task 1 baseline — TF-IDF + Logistic Regression
# **Owner:** C (Task-1 lead) · **Phase 1** (runs in seconds on CPU)
#
# Protocol: fit on `train`, choose C on `val`, evaluate ONCE on `test` (Bitext) and on the real-customer
# external test (BANKING77, mapped intents). The saved model is also the Agentic RAG's intent tool.

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

import joblib, pandas as pd
from IPython.display import display
from src import config as C
from src.utils import set_seed
from src.data.load import load_banking77
from src.data.eval_sets import banking77_mapped
from src.classify import tfidf_lr, metrics as M
set_seed()

df = pd.read_parquet(C.SPLITS_FILE)
train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
b77_ext = banking77_mapped(load_banking77())                # high+medium confidence mappings
b77_strict = banking77_mapped(load_banking77(), strict=True)  # high confidence only
print(len(train), len(val), len(test), "| external:", len(b77_ext), "(strict", len(b77_strict), ")")

# %%
model, grid = tfidf_lr.tune_and_fit(train, val)
display(grid)
joblib.dump(model, C.OUTPUTS / "tfidf_lr.joblib")

# %%
results = []
test = test.copy()
test["pred"], ms = tfidf_lr.predict_timed(model, test["instruction"].tolist())
results.append(M.summary(test["intent"], test["pred"], "tfidf_lr", "bitext_test", ms))
for name, ext in (("banking77_mapped", b77_ext), ("banking77_mapped_strict", b77_strict)):
    p, ms = tfidf_lr.predict_timed(model, ext["text"].tolist())
    results.append(M.summary(ext["gold_intent"], p, "tfidf_lr", name, ms))
res = pd.DataFrame(results); display(res)
res.to_csv(C.TABLES / "task1_tfidf_lr_results.csv", index=False)

# %%
display(M.per_class(test["intent"], test["pred"]).head(8))
display(M.accuracy_by_flag(test))
display(M.top_confusions(test["intent"], test["pred"]))
M.plot_confusion(test["intent"], test["pred"], "task1_tfidf_confusion_bitext", "TF-IDF + LR · Bitext test")

# %% [markdown]
# ## Error analysis (fill in 10–20 examples for the report)
# Look at wrong predictions on BANKING77: are they mapping problems (our label is debatable) or model problems?

# %%
b77_ext = b77_ext.copy(); b77_ext["pred"] = model.predict(b77_ext["text"])
errs = b77_ext[b77_ext["pred"] != b77_ext["gold_intent"]][["b77_intent", "text", "gold_intent", "pred", "confidence"]]
display(errs.sample(min(20, len(errs)), random_state=C.SEED))
