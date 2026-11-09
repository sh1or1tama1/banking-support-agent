# Intelligent Banking Customer Support Agent
**ADSP 32018 · NLP & Agentic AI · Final project** — a comparative study of intent classification and
grounded response generation for a (fictional) retail bank.

| Task | Baseline (champion) | Challenger(s) | Main metrics |
|---|---|---|---|
| 1 · Intent classification (26 intents) | TF-IDF + Logistic Regression | fine-tuned BERT · LLM few-shot | Accuracy, Macro-F1, per-flag robustness, BANKING77 transfer, latency |
| 2 · Response generation | Naive RAG (retrieve once → generate) | **Agentic RAG** (LangGraph: analyse → classifier tool → hybrid retrieve → grade → rewrite → generate → validate → revise / escalate) | behaviour accuracy, escalation F1, rubric (human + LLM judge), invented-specifics rate, cost, latency |

Direct LLM (no retrieval) is kept as a reference lower bound. Ablations switch off each agentic component.

## Repository map (who owns what)
```
data/mappings/       banking77_to_bitext.csv (C reviews) · out_of_domain_queries.csv · ambiguous_queries.csv (C extends)
data/processed/      FROZEN: bitext_splits.parquet, split_manifest.json, eval_sets.parquet, human_eval_subset.csv
prompts/             versioned prompts + CHANGELOG.md   (never edit a prompt that produced reported results)
src/data/            load.py · split.py (leakage-aware split) · eval_sets.py
src/eda/eda.py       every EDA table/figure the rubric asks for
src/classify/        tfidf_lr.py · bert.py · llm_fewshot.py · metrics.py            (owner C)
src/rag/             retriever.py · systems.py (Direct, Naive RAG) · agentic_rag.py   (owner B)
src/llm/             client.py (one client, disk cache, tokens/cost/latency) · prompts.py
src/eval/            harness.py (run any system, auto checks) · judge.py (LLM judge, human sheet, kappa)
notebooks/           01 split · 02 EDA · 03 TF-IDF · 04 eval set + Naive RAG · 05 Agentic RAG + ablations
                     06 BERT + LLM few-shot · 07 final evaluation            (py/ = same notebooks as text, for git diffs)
app/streamlit_app.py demo (owner C)
tests/               offline tests: split integrity, KB train-only, prompt rendering, agent branches
```

## Quick start (Google Colab, T4 GPU)
```python
!git clone https://github.com/<team>/banking-support-agent.git
%cd banking-support-agent
!pip -q install -r requirements.txt
```
Then open the notebooks in order. **01 is run once by A and the outputs are committed — nobody re-runs it.**

### LLM API (important)
Claude Pro / ChatGPT Plus subscriptions do **not** include API access. Create an API key (OpenAI or
Anthropic console), add a small prepaid credit with a spending limit, and store the key in Colab →
🔑 Secrets as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Then set:
```python
os.environ["LLM_PROVIDER"] = "openai"            # or "anthropic"
os.environ["LLM_MODEL"] = "<cheapest current small model>"   # SAME model for every Task-2 system
os.environ["PRICE_IN_PER_M"] = "<usd>"; os.environ["PRICE_OUT_PER_M"] = "<usd>"   # for the cost column
```
Every call is cached in `outputs/llm_cache/`, so re-running a notebook is free. Use `LIMIT = 20` for
development runs; run the full eval set only when prompts are frozen. `LLM_PROVIDER=mock` runs
everything offline with fake answers (for testing code only).

## Team rules that protect the grade
1. **Never look at / tune on the test split or `eval_sets.parquet`.** Tune on `val` and on dev runs only.
2. Same LLM, temperature 0, same policy prompt, same knowledge base, same eval set for every Task-2 system.
3. Prompt change → new file `*_v2.md` + a line in `prompts/CHANGELOG.md`; new run tag.
4. Commit run files (`outputs/runs/*.jsonl`) — they are the evidence behind every number in the report.
5. Disclose AI assistance (course policy): list tools + versions in the report's acknowledgements.

## Tests
```bash
python scripts/make_standin_bitext.py      # fake data with the Bitext schema (offline only)
LLM_PROVIDER=mock python -m pytest tests -q
```

## Data & licences
- Bitext Retail Banking (synthetic, 25,545 Q/A pairs, 26 intents) — CDLA-Sharing-1.0 —
  https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset
- BANKING77 (real customer queries, 77 intents; Casanueva et al., 2020) —
  https://github.com/PolyAI-LDN/task-specific-datasets · https://huggingface.co/datasets/PolyAI/banking77
- Background: Bitext, *From General-Purpose LLMs to Verticalized Enterprise Models* (blog).

## References (methods)
LangGraph documentation · Lewis et al. 2020 (RAG) · Yan et al. 2024 (Corrective RAG) · Asai et al. 2023
(Self-RAG) · Yao et al. 2023 (ReAct) · Shinn et al. 2023 (Reflexion) · Devlin et al. 2019 (BERT) ·
Reimers & Gurevych 2019 (Sentence-BERT) · Robertson & Zaragoza 2009 (BM25) · Cormack et al. 2009 (RRF) ·
Es et al. 2024 (RAGAS) · Zheng et al. 2023 (LLM-as-a-judge).
