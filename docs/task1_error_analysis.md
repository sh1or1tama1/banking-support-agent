# Task 1 results and error analysis

*Phase 2, Szuyu's area; run by Ziming on 2026-11-23 (notebook 06, BERT trained on a Colab T4).
Drafted with AI assistance (Claude Code). Data: `outputs/tables/task1_all_models.csv`,
`task1_banking77_predictions.csv`, `task1_error_examples.csv`.*

## Results

**Table 1. Intent classification (26 intents).** The Bitext subset is 20 test queries per intent. BANKING77 mapped
uses the 520 test queries whose intent maps to Bitext with high or medium confidence.

| Model | Bitext test (3,831) | Bitext subset (520) | BANKING77 mapped (520) | Macro-F1 on BANKING77 | Latency per query | Cost |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.999 | 0.998 | 0.410 | 0.516 | 0.05 ms | free |
| BERT (fine-tuned) | 0.999 | 1.000 | 0.477 | 0.567 | 6-14 ms (CPU) | free |
| LLM few-shot (gpt-5.4-nano, 2 shots per intent) | not run (cost) | 0.944 | **0.692** | **0.853** | ~600 ms | $0.14 per 520 queries |

BERT's validation Macro-F1 rose from 0.9965 to 0.9997 over 4 epochs (`task1_bert_training_history.csv`).

**Reading the table.**
- **On Bitext, every model is at the ceiling.** This is expected, because 52% of test queries have a near-paraphrase in training.
- **On real queries, every model drops sharply.** BERT is only 7 points better than TF-IDF, which shows that fine-tuning on templated data does not buy much robustness to real wording.
- **The LLM is the most robust classifier on real queries without any training,** but it is 40 to 100 times slower than BERT and costs money per query.
- **Of the LLM's invalid labels on BANKING77 (20.8%), 107 of 108 are the category name `FEES` instead of the intent `check_fees`.** It is the only intent in that category, so the LLM understood the question. Read leniently, its accuracy would be 0.898. We report the strict number, because copying the label exactly is part of the task.

## Error analysis (BERT on BANKING77)

BERT makes 272 errors on the 520 mapped BANKING77 queries.

- **48% of the errors (131) come from four fee intents** mapped to `check_fees` with medium confidence:
  `exchange_charge`, `card_payment_fee_charged`, `transfer_fee_charged` and `cash_withdrawal_charge`.
  Real customers ask *why* a specific fee was charged. Bitext's `check_fees` asks *where to see* fees. BERT
  follows the other content words instead: transfer → `make_transfer`, card → `check_card_annual_fee`, cash → `dispute_ATM_withdrawal`.
- **Keyword anchoring** explains most other errors. "ATM" pulls queries to `recover_swallowed_card`, "transaction" pulls them to
  `dispute_ATM_withdrawal`, and "transfer" pulls them to `make_transfer`. Words that are frequent in real queries but absent from
  Bitext (*why, pending, passcode, charged*) give BERT no signal.
- **Confidence is informative only at the top.** At confidence ≥ 0.99, BERT is right 90% of the time (136 queries).
  At ≥ 0.6 it is right only 55% of the time. This confirms the 0.99 threshold for the agent's intent filter, which was
  chosen on BANKING77 *train* (`intent_tool_threshold_sweep.csv`).
- **The models fail differently.** The LLM is right on 138 of BERT's 272 errors and TF-IDF on 49, so they are not making the same mistakes.

**Table 2. 20 sampled errors**, 10 model problems and 10 mapping or label problems (`task1_error_examples.csv`).
*Model problem*: the query clearly expresses the gold intent and BERT's prediction cannot be defended.
*Mapping problem*: the BANKING77 label is noisy, or another Bitext intent fits as well as the mapped one.

| # | Query (BANKING77) | Gold | BERT (conf) | Type | Why |
|---|---|---|---|---|---|
| 1 | I accidentally made a transaction to the wrong account. | cancel_transfer | dispute_ATM_withdrawal (0.94) | model | Bitext has this template with "transfer"; "transaction" misleads BERT |
| 2 | I submitted a transaction to the incorrect account. | cancel_transfer | dispute_ATM_withdrawal (0.98) | model | Same shift, confidently wrong |
| 3 | When I made a transfer, there was a fee. Why? | check_fees | make_transfer (0.96) | model | Anchors on "transfer" |
| 4 | I did not get as much cash as I requested at the ATM. | dispute_ATM_withdrawal | recover_swallowed_card (0.60) | model | Anchors on "ATM" |
| 5 | The wrong amount of cash came out of the ATM today… | dispute_ATM_withdrawal | recover_swallowed_card (0.74) | model | Anchors on "ATM"; longer than any Bitext query |
| 6 | I'm in Spain and my stuff has been stolen… the old one frozen. | block_card | recover_swallowed_card (0.48) | model | Multi-part real query |
| 7 | I do not know my passcode. | get_password | block_card (0.40) | model | "passcode" is rare in Bitext |
| 8 | They charged me for paying with my card. | check_fees | cancel_card (0.80) | model | Not defensible |
| 9 | If I feel someone has my card information, can I get a new card? | block_card | recover_swallowed_card (0.76) | model | Not defensible |
| 10 | I need to reset my passcode. How do I do it? | get_password | activate_card (0.73) | model | TF-IDF is right here |
| 11 | Is there a fee to exchange foreign money? | check_fees | activate_card_international_usage (0.66) | mapping | Bitext does not cover FX fees |
| 12 | Will I always be charged a fee for using my card? | check_fees | check_card_annual_fee (0.77) | mapping | Annual fee is an equally good reading |
| 13 | Someone else may be using my card. There are transactions I don't recognize. | block_card | dispute_ATM_withdrawal (0.88) | mapping | Disputing is just as plausible |
| 14 | Why am I being charged when I withdraw cash? | check_fees | dispute_ATM_withdrawal (0.93) | mapping | Both intents partly fit |
| 15 | More fees, this time for withdrawing my own cash. Why are you robbing me? | check_fees | dispute_ATM_withdrawal (0.45) | mapping | A complaint about a charge |
| 16 | Think someone has took money out with my card. What shall I do? | dispute_ATM_withdrawal | recover_swallowed_card (0.38) | mapping | block_card fits as well |
| 17 | My ATM got stuck and I'm not sure what to do. | recover_swallowed_card | find_ATM (0.989) | mapping | Ambiguous: is the card stuck? |
| 18 | Oh no! I lost my card! Help! | block_card | cancel_card (0.83) | mapping | Bitext separates block and cancel; BANKING77 does not |
| 19 | WHAT CAN I DO AFTER THE CARD MISSING | activate_card | block_card (0.61) | mapping | Label noise; BERT's answer is better |
| 20 | Is there a way my new card can be renewed? | activate_card | recover_swallowed_card (0.94) | mapping | Label noise; renewing is not activating |

## Choice of intent tool for the Agentic RAG

The agent uses **BERT with a confidence threshold of 0.99**: retrieval is filtered by BERT's intent only above that
confidence. The choice was made on dev data, not on the eval set:

| Tool and threshold | VAL R@1 | BANKING77 train R@1 |
|---|---|---|
| No filter (hybrid retrieval) | 0.986 | 0.642 |
| BERT, 0.6 | 1.000 | 0.532 |
| BERT, 0.9 | 1.000 | 0.605 |
| **BERT, 0.99** | **1.000** | **0.641** |
| TF-IDF, 0.99 | 0.990 | 0.640 |

Lower thresholds cost up to 11 points on real queries, because confidently wrong predictions filter out the right
documents. The LLM few-shot classifier is more accurate on real queries, but the agent's analysis step already asks the
LLM for an intent. The `agentic_no_classifier` ablation measures exactly that alternative.
