# %% [markdown]
# # 02 · Exploratory Data Analysis (rubric: 15 pts)
# **Owner:** A (project lead) · **Phase 1** · Figures → `outputs/figures/`, tables → `outputs/tables/`
#
# Rubric checklist (tick each in the report):
# - [ ] sources, record counts, missing values, schema  (§1)
# - [ ] token distribution, vocabulary size, sequence length  (§3)
# - [ ] charts: word cloud, frequency plots, embedding map, similarity heatmap  (§3–5)
# - [ ] label / topic distribution  (§2)
# - [ ] linguistic features: n-grams, POS tags, entities  (§4)
# - [ ] generated (synthetic) data: quality, diversity, bias, anomalies, edge cases  (§6)

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
from src.data.load import load_banking77
from src.eda import eda
from src.embeddings import Embedder

df = pd.read_parquet(C.SPLITS_FILE)          # frozen split from notebook 01
b77 = load_banking77()
pd.set_option("display.max_colwidth", 120)

def save(t, name):
    t.to_csv(C.TABLES / f"{name}.csv", index=False); return t

# %% [markdown]
# ## 1. Sources, schema, missing values, quality
# | Dataset | Source | Role | Rows |
# |---|---|---|---|
# | Bitext Retail Banking (synthetic, CDLA-Sharing-1.0) | huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset | train/val/test, RAG knowledge base | see below |
# | BANKING77 (real customer queries, CC-BY-4.0) | github.com/PolyAI-LDN/task-specific-datasets | external test (real wording), unsupported-intent queries | 13,083 |

# %%
display(save(eda.schema_table(df[["flags", "instruction", "category", "intent", "response"]]).reset_index(names="column"), "eda_schema"))
display(save(eda.quality_report(df), "eda_quality"))
display(df.groupby("split").size().rename("rows").to_frame())

# %% [markdown]
# ## 2. Label distribution (intents, categories, language-variation flags)

# %%
counts = eda.plot_label_distribution(df)
display(counts.groupby("category")["n"].agg(["count", "sum"]).rename(columns={"count": "intents", "sum": "rows"}))
display(save(eda.plot_flags(df), "eda_flags"))

# %% [markdown]
# ## 3. Lengths, vocabulary, n-grams, word cloud

# %%
try:  # subword lengths with the same tokenizer BERT will use
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.BERT_MODEL)
except Exception as e:
    print("tokenizer unavailable -> word counts only:", type(e).__name__); tok = None
lengths = eda.length_table(df, tok)
display(save(lengths.describe(percentiles=[.5, .9, .99]).T.round(1).reset_index(names="measure"), "eda_lengths"))
eda.plot_lengths(lengths)
eda.plot_response_length_by_intent(df)
if tok is not None:
    print(f"queries longer than 64 BERT tokens: {(lengths['instruction_subwords'] > 64).mean():.2%} -> max_length=64 is safe")

# %%
vocab = pd.DataFrame([eda.vocab_stats(df["instruction"], "Bitext queries"), eda.vocab_stats(df["response"], "Bitext responses")])
display(save(vocab, "eda_vocab"))
uni, bi = eda.plot_top_ngrams(df["instruction"])
eda.plot_wordcloud(df["instruction"])
display(save(eda.distinctive_terms_per_intent(df), "eda_terms_per_intent"))

# %% [markdown]
# ## 4. Linguistic features: POS tags & named entities (spaCy, 3,000-row sample)

# %%
pos_df, ent_df = eda.pos_entity_profile(df["instruction"])
eda.plot_pos(pos_df)
display(save(pos_df, "eda_pos"), save(ent_df, "eda_entities_queries"))
_, ent_resp = eda.pos_entity_profile(df["response"], sample=1500)
display(save(ent_resp, "eda_entities_responses"))
# Interpretation hint: few real entities in queries (synthetic data has no names/amounts) vs. many
# placeholders in responses -> motivates the "no invented specifics" rule in our prompts.

# %% [markdown]
# ## 5. Embedding map & intent similarity (sentence-transformers; TF-IDF fallback offline)

# %%
from src.utils import sample_per_group
sample = sample_per_group(df, "intent", 150)
emb_model = Embedder().fit(df.loc[df["split"] == "train", "instruction"].tolist())
print("embedding backend:", emb_model.backend)
emb = emb_model.encode(sample["instruction"].tolist())
eda.plot_embedding_map(emb, sample["category"].reset_index(drop=True))
sim_pairs = eda.plot_intent_similarity(emb, sample["intent"].reset_index(drop=True))
display(save(sim_pairs.head(10), "eda_most_similar_intent_pairs"))
# These pairs (e.g. cancel_loan vs cancel_mortgage) are where we expect classifier confusion -> check in Task 1.

# %% [markdown]
# ## 6. Synthetic-data checks: diversity, placeholders, anomalies, synthetic vs real

# %%
display(save(eda.template_diversity(df), "eda_template_diversity"))
display(save(eda.placeholder_table(df), "eda_placeholders"))
for name, t in eda.anomalies(df).items():
    print(f"\n### {name} ({len(t)})"); display(t)

# %%
display(save(eda.synthetic_vs_real(df, b77), "eda_synthetic_vs_real"))
# Real customers write differently (length, vocabulary) -> justification for the BANKING77 external test.

# %% [markdown]
# ## 7. Findings to carry into the report (fill in after running)
# 1. Class balance: … → we report Macro-F1 but accuracy is also meaningful.
# 2. Templating: …% of responses share the top opening → reference-similarity metrics are inflated; use rubric + LLM judge.
# 3. Placeholders: … → policy rule 2 and the `invented_*` checks.
# 4. Most similar intents: … → expected confusions.
# 5. Synthetic vs real gap: … → external test on BANKING77.
