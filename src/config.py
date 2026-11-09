"""Central configuration. Every notebook and script imports paths and settings from here,
so the whole team uses the same split, seed, models and folders.

Override anything with environment variables (e.g. in Colab: os.environ["LLM_MODEL"] = "...").
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MAPPINGS = DATA / "mappings"
OUTPUTS = ROOT / "outputs"
FIGS = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
RUNS = OUTPUTS / "runs"          # one JSONL file per (system, eval set) run
LLM_CACHE = OUTPUTS / "llm_cache"  # every LLM call is cached here -> re-runs are free & reproducible
PROMPTS = ROOT / "prompts"

for _p in (RAW, PROCESSED, MAPPINGS, FIGS, TABLES, RUNS, LLM_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# Frozen artefacts produced by notebook 01 (do NOT regenerate after the team freezes them)
SPLITS_FILE = PROCESSED / "bitext_splits.parquet"
SPLIT_MANIFEST = PROCESSED / "split_manifest.json"
B77_MAPPING_FILE = MAPPINGS / "banking77_to_bitext.csv"
OOD_FILE = MAPPINGS / "out_of_domain_queries.csv"
AMBIGUOUS_FILE = MAPPINGS / "ambiguous_queries.csv"
EVAL_SETS_FILE = PROCESSED / "eval_sets.parquet"

# ---------------------------------------------------------------- data sources
BITEXT_HF_ID = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
# Optional: path to a local copy (parquet/csv) if Hugging Face is unreachable.
BITEXT_LOCAL = os.getenv("BITEXT_LOCAL", "")
BANKING77_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}

# ---------------------------------------------------------------- reproducibility
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_SPLIT_FOLDS = 20            # 14 folds -> train, 3 -> val, 3 -> test (StratifiedGroupKFold)
NEAR_DUP_THRESHOLD = float(os.getenv("NEAR_DUP_THRESHOLD", 0.90))  # char n-gram cosine for "same template"

# ---------------------------------------------------------------- models
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BERT_MODEL = os.getenv("BERT_MODEL", "bert-base-uncased")

# LLM: provider in {"openai", "anthropic", "openai_compatible", "mock"}.
# The model name is deliberately NOT hard-coded: pick the cheapest current small model
# from your provider's model list and keep it identical for every Task-2 system.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # only for openai_compatible (Groq, Ollama, vLLM ...)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))
# USD per 1M tokens, used only for the cost column. Fill in from your provider's pricing page.
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", 0.0))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", 0.0))

# ---------------------------------------------------------------- RAG
RAG_TOP_K = 3

# Figure style (validated categorical palette, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
