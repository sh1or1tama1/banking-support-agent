# Phase 1 status (branch `dev/zfang75`)

Run by Ziming on 2026-11-22, locally on macOS (Python 3.11, `.venv`), not in Colab.
Everything below is on this branch for review. Nothing has been merged into `main`.
Coding assistant: Claude Code (Claude Opus 5.5); list it in the report's acknowledgements.

## Checklist (from Plan.pdf, Phase 1)

| Item | Status |
|---|---|
| Repo created, everyone invited | Partly. Szuyu is not a collaborator yet. Only Ryusei (admin) can invite. |
| LLM provider + API key | Provider chosen: OpenAI. Key and model still to be set up. |
| Notebook 01: split + leakage audit, commit split files | Done. Split frozen pending review. |
| Notebook 02: EDA + 5 findings | Done. Findings are in the last cell of `notebooks/02_eda.ipynb`. |
| Notebook 03: TF-IDF baseline | Done. |
| Review `banking77_to_bitext.csv` | Done. See "Decisions to review". |
| Extend `ambiguous_queries.csv` to ~30 | Done (30). |
| Notebook 04 §1-2: frozen eval set | Done: `data/processed/eval_sets.parquet`. |
| Notebook 04 §3: Direct LLM + Naive RAG runs | **Not done yet.** Needs the OpenAI key. |

## Key numbers

**Split** (seed 42, grouped 70/15/15): 17,882 train / 3,832 val / 3,831 test. 0 label conflicts; every intent is in every split.
Re-running notebook 01 reproduces the identical split.

**Leakage audit:**

| Split | Median max similarity to train | Test rows with a near-copy (≥ 0.9) in train |
|---|---|---|
| Grouped (ours) | 0.904 | 52.1% |
| Random | 0.927 | 59.9% |

Exact duplicates across splits: 0%. 99.8% of the remaining near-copies have the same intent, for example
"how do I activate a card?" (test) vs "I need to activate a card, how do I do it?" (train).
These are template paraphrases that the 0.90 within-intent grouping does not catch. This is a property of
Bitext (template-generated), so in-distribution scores will be near the ceiling. BANKING77 is the real test.

**Task 1, TF-IDF + LR** (best C = 100 on val):

| Eval set | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Bitext test | 3,831 | 0.999 | 0.999 |
| BANKING77 mapped (high + medium) | 520 | 0.410 | 0.516 |
| BANKING77 mapped (high only) | 240 | 0.554 | 0.630 |

**Frozen Task 2 eval set** (436 queries; human-rating subset: 100):

| Eval set | Expected action | Queries |
|---|---|---|
| bitext_core | answer | 104 |
| bitext_hard | answer | 50 |
| b77_real_inscope | answer | 100 |
| b77_unsupported | escalate | 122 |
| out_of_domain | escalate | 30 |
| ambiguous | clarify | 30 |

`bitext_hard` has 50, not 70: two of the seven `HARD_FLAGS` (N, K) never occur in Bitext retail banking.

**Retrieval sanity check** (intent of retrieved docs, `outputs/tables/retrieval_proxy_metrics.csv`):

| Retriever | bitext_core R@1 | bitext_hard R@1 | b77_real_inscope R@1 |
|---|---|---|---|
| Dense (Naive RAG) | 1.00 | 0.92 | 0.71 |
| BM25 | 0.91 | 0.86 | 0.30 |

## Decisions to review

1. **BANKING77 mapping** (`data/mappings/banking77_to_bitext.csv`). The file listed 49 of the 77 intents; the other 28
   were silently dropped from every test. All 77 are now listed:
   - 19 added as `UNSUPPORTED, high`: refunds, declined/failed/pending transfers, double charges, FX rates, card acceptance, card linking, card or contactless not working, and similar.
   - 8 added as `UNSUPPORTED, medium`: loosely related to a Bitext intent but not answerable from it. Examples: pending payments vs `check_recent_transactions`, `pin_blocked` vs passwords, `receiving_money` vs `make_transfer`, and `card_payment_not_recognised`. Bitext only covers ATM disputes.
   - 1 added as `make_transfer, low`: `transfer_into_account` is ambiguous between a transfer and a top-up. Low rows are excluded from evaluation.
   - Fixed the rationale of `get_physical_card`. Its real queries ask where to find the PIN of a new card, not how to order one.
   - No existing mapping was changed. The `b77_unsupported` set grew from 2 x 34 to 2 x 61 intents (68 to 122 queries).
2. **Ambiguous queries**: 22 new queries (author column `ziming (AI-assisted, Claude)`). Each fits at least two Bitext
   intents, and every `plausible_intents` entry was checked against the real intent list.
3. **Notebooks**: real repo URL filled in. Notebook 04 now reads optional `LIMIT` and `RUN_PREFIX` environment variables.

## Notes for Phase 2 (Agentic RAG)

Dev comparison on VAL and BANKING77 *train* (not test), 800 queries each. Intent of the top-3 retrieved docs:

| Mode | Val R@1 | BANKING77 train R@1 |
|---|---|---|
| Dense (Naive RAG) | 0.988 | 0.624 |
| BM25 | 0.900 | 0.322 |
| Hybrid (RRF) | 0.986 | 0.642 |
| Hybrid + TF-IDF intent filter (conf ≥ 0.6) | 0.999 | 0.520 |

With TF-IDF as the intent tool, the filter **hurts** on real queries. TF-IDF is only 39% accurate there, but it is
confident (≥ 0.6) on 60% of them. Use BERT as the tool, or raise the threshold, before freezing the agent.

## How to reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_data_and_split.ipynb
```

On macOS 14 with Apple Silicon, use `torch==2.14.1`. torch 2.15 crashes on the MPS backend.
