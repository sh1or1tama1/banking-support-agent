# %% [markdown]
# # 06 · Task 1 challengers — fine-tuned BERT and LLM few-shot
# **Owner:** C (Task-1 lead) · **Phase 2** · Colab → Runtime → Change runtime type → **T4 GPU**
#
# Same splits, same metrics module as notebook 03 → the three models are directly comparable.
# LLM few-shot is scored on a stratified subset (20 per intent) of the test split to control API cost;
# report that subset size, and score the other two models on the SAME subset too (done below).

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
# os.environ["LLM_PROVIDER"] = "openai"; os.environ["LLM_MODEL"] = "<same model as Task 2>"
sys.path.insert(0, os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebooks" else "."))

import joblib, pandas as pd
from IPython.display import display
from src import config as C
from src.utils import set_seed, sample_per_group
from src.data.load import load_banking77
from src.data.eval_sets import banking77_mapped
from src.classify import metrics as M
set_seed()

df = pd.read_parquet(C.SPLITS_FILE)
train, val, test = (df[df["split"] == s].copy() for s in ("train", "val", "test"))
b77 = banking77_mapped(load_banking77())
test_small = sample_per_group(test, "intent", 20)            # shared subset for all 3 models
print(len(train), len(val), len(test), "| subset", len(test_small), "| BANKING77 mapped", len(b77))

# %% [markdown]
# ## 1. Fine-tune BERT (GPU)

# %%
from src.classify.bert import train_bert, BertIntent
model, tok, labels, history = train_bert(train, val)
display(history[["epoch", "eval_accuracy", "eval_macro_f1", "eval_loss"]])
bert = BertIntent()

# %% [markdown]
# ## 2. LLM few-shot (2 examples per intent from TRAIN)

# %%
from src.llm.client import LLMClient
from src.classify.llm_fewshot import LLMFewShotClassifier
fewshot = LLMFewShotClassifier(LLMClient(), train, k_per_intent=2)
print(fewshot.system[:1200])

# %% [markdown]
# ## 3. Compare all three on identical data

# %%
tfidf = joblib.load(C.OUTPUTS / "tfidf_lr.joblib")
from src.classify.tfidf_lr import predict_timed
rows, preds = [], {}
for set_name, d, text_col, gold_col in (("bitext_test_full", test, "instruction", "intent"),
                                        ("bitext_test_subset", test_small, "instruction", "intent"),
                                        ("banking77_mapped", b77, "text", "gold_intent")):
    texts = d[text_col].tolist()
    p, ms = predict_timed(tfidf, texts); rows.append(M.summary(d[gold_col], p, "tfidf_lr", set_name, ms)); preds[("tfidf_lr", set_name)] = p
    p, ms = bert.predict(texts);         rows.append(M.summary(d[gold_col], p, "bert_finetuned", set_name, ms)); preds[("bert_finetuned", set_name)] = p
    if set_name != "bitext_test_full":   # LLM only on the subset + external set (cost)
        r = fewshot.predict(texts)
        rows.append({**M.summary(d[gold_col], r["pred"], "llm_fewshot", set_name, 1000 * r["latency_s"].mean()),
                     "invalid_label_rate": round((r["pred"] == "INVALID").mean(), 4), "cost_usd": round(r["cost_usd"].sum(), 4)})
        preds[("llm_fewshot", set_name)] = r["pred"].to_numpy()
res = pd.DataFrame(rows); display(res)
res.to_csv(C.TABLES / "task1_all_models.csv", index=False)

# %%
# Robustness by Bitext flag (typos Z, colloquial Q, ...) for BERT vs TF-IDF on the full test split
test["pred"] = preds[("bert_finetuned", "bitext_test_full")]
display(M.accuracy_by_flag(test).rename(columns={"accuracy": "bert_acc"}))
M.plot_confusion(b77["gold_intent"], preds[("bert_finetuned", "banking77_mapped")], "task1_bert_confusion_banking77", "BERT · BANKING77 (real customers)")

# %% [markdown]
# ## 4. Interpretation checklist for the report
# - Bitext test: are all models near the ceiling? (expected - synthetic, templated data)
# - BANKING77: how much does each model drop on real wording? Which handles it best, and why?
# - Cost / latency per query: TF-IDF (ms, free) vs BERT (GPU ms) vs LLM (≈ seconds, $).
# - 10-20 concrete errors: mapping ambiguity vs genuine model mistakes.
# - Pick the intent tool for the Agentic RAG (best on BANKING77 with acceptable latency) and save its choice.
