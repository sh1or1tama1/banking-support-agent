# Draft report chapters: Data and Exploratory Data Analysis

*Draft for Ryusei to edit (owner of these chapters). All numbers come from notebooks 01-04 on branch `dev/zfang75`.
Figures are in `outputs/figures/` and tables in `outputs/tables/`. Drafted with AI assistance (Claude Code); disclose in the acknowledgements.*

---

## 2. Data

### 2.1 Sources

We use two public datasets with complementary roles (Table 2.1).

**Table 2.1. Datasets.**

| Dataset | Content | Licence | Size | Role in this project |
|---|---|---|---|---|
| Bitext Retail Banking (Hugging Face) | Synthetic customer queries with intent labels and reference answers | CDLA-Sharing-1.0 | 25,545 Q&A pairs, 26 intents, 9 categories | Training, validation and test for Task 1; knowledge base and main test for Task 2 |
| BANKING77 (PolyAI, Casanueva et al., 2020) | Real online-banking customer queries with intent labels | CC-BY-4.0 | 13,083 queries (10,003 train, 3,080 test), 77 intents | External test of generalisation to real wording; source of out-of-scope queries that should be escalated |

Bitext is template-generated. Each row has a customer instruction, an intent, a category, a long reference response
and a set of *language-variation flags* that record how the query was perturbed. Examples are typos (Z), colloquial
wording (Q), offensive language (W) and multi-part phrasing (C). The schema check found no missing or empty values.
No instruction maps to more than one intent, and there are no exact duplicate instructions or (instruction, response) pairs
(`eda_schema.csv`, `eda_quality.csv`).

BANKING77 has only a query and an intent. Its intents are more fine-grained than Bitext's and only partly overlap
with them, so we built a manual mapping (Section 2.3).

We also wrote two small query sets by hand. **Out-of-domain** has 30 queries: weather, investment and legal advice,
prompt injection, requests that need authentication, and similar. **Ambiguous** has 30 queries such as
"something is wrong with my card", each plausible for at least two Bitext intents.

### 2.2 Leakage-aware split

Bitext's templating makes a random split misleading. Many queries are paraphrases of the same template, so a random
split puts near-copies of a test query into training. We therefore:

1. normalised each query (lower case, placeholders and punctuation removed);
2. grouped near-duplicates *within each intent* by greedy leader clustering on character 3-5-gram TF-IDF
   (cosine ≥ 0.90), which produced 19,193 groups (mean size 1.33, largest 17);
3. split whole groups with stratified group K-fold (20 folds, seed 42) into 70 / 15 / 15%.

**Table 2.2. Split.**

| Split | Rows | Share |
|---|---|---|
| Train | 17,882 | 70.0% |
| Validation | 3,832 | 15.0% |
| Test | 3,831 | 15.0% |

Every intent appears in every split, and there are no label conflicts. The split is frozen
(`data/processed/bitext_splits.parquet`). Re-running the notebook reproduces it exactly
(`split_manifest.json`, test-ID hash `c9dc69de5256db74`).

**Leakage audit.** For every test query we computed its maximum similarity to any training query and compared our
split with a random split of the same proportions (Figure 2.1, `split_leakage_audit.png`).

**Table 2.3. Leakage audit.**

| Split strategy | Median max similarity to train | Test queries with a near-copy (≥ 0.9) in train |
|---|---|---|
| Grouped (ours) | 0.904 | 52.1% |
| Random | 0.927 | 59.9% |

Grouping removes exact and very close copies, but more than half of the test queries still have a close paraphrase in
training. 99.8% of those paraphrases share the test query's intent, for example "how do I activate a card?" (test)
vs "I need to activate a card, how do I do it?" (train). This is a property of template-generated data rather than a
flaw of the split. It means in-distribution test scores will be close to the ceiling, which is why we add the
BANKING77 external test.

### 2.3 Mapping BANKING77 to Bitext

We mapped each of the 77 BANKING77 intents either to a Bitext intent or to `UNSUPPORTED`. Each mapping has a confidence
level and a written rationale (`data/mappings/banking77_to_bitext.csv`).

**Table 2.4. BANKING77 mapping.**

| Outcome | Confidence | Intents | Used for |
|---|---|---|---|
| Mapped to a Bitext intent | high | 6 | Task 1 strict external test; Task 2 `b77_real_inscope` |
| Mapped to a Bitext intent | medium | 7 | Task 1 external test; Task 2 `b77_real_inscope` |
| Mapped, ambiguous | low | 3 | Not used |
| Unsupported | high / medium | 61 | Task 2 `b77_unsupported` (should be escalated) |

Only 13 of 77 real intents can be answered from Bitext. Most of the rest are about a specific transaction's status
(pending, declined or failed payments, refunds) or about products Bitext does not cover (top-ups, virtual cards,
currency exchange, identity verification). A responsible assistant should hand these to a human.

### 2.4 Task 2 evaluation set

All Task 2 systems are scored on one frozen table of 436 queries (`data/processed/eval_sets.parquet`). It combines six
subsets, each with an expected behaviour (Table 2.5). Bitext queries come only from the test split, and BANKING77
queries only from its test split.

**Table 2.5. Evaluation subsets.**

| Subset | Source | Queries | Expected action |
|---|---|---|---|
| bitext_core | Bitext test, 4 per intent | 104 | answer |
| bitext_hard | Bitext test, 10 per hard flag (C, Z, Q, W, E) | 50 | answer |
| b77_real_inscope | BANKING77 test, mapped intents (high/medium) | 100 | answer |
| b77_unsupported | BANKING77 test, 2 per unsupported intent | 122 | escalate |
| out_of_domain | hand-written | 30 | escalate |
| ambiguous | hand-written | 30 | clarify (escalate also accepted) |

A stratified subset of 100 queries is used for the blinded human rating (`human_eval_subset.csv`). Prompt tuning
uses a separate 48-query development set drawn from the validation split, BANKING77 *train* and different hand-written
queries (`scripts/make_dev_set.py`). The test split and the evaluation set are therefore never used for tuning.

---

## 3. Exploratory Data Analysis

### 3.1 Labels

The 26 intents are close to balanced (Figure 3.1, `eda_intent_distribution.png`). Most have 947-1,000 rows. The
exceptions are `get_password` (799) and `set_up_password` (901). The 9 categories are less even because CARD and LOAN
hold six intents each, while FEES holds one (Table 3.1). Because classes are balanced, accuracy and Macro-F1 will
be close; we report both.

**Table 3.1. Categories.**

| Category | Intents | Rows |
|---|---|---|
| CARD | 6 | 5,980 |
| LOAN | 6 | 5,954 |
| ACCOUNT | 3 | 2,994 |
| FIND | 2 | 1,998 |
| CONTACT | 2 | 1,997 |
| TRANSFER | 2 | 1,992 |
| ATM | 2 | 1,983 |
| PASSWORD | 2 | 1,700 |
| FEES | 1 | 947 |

**Language-variation flags** (Figure 3.2, `eda_flag_distribution.png`). Every query has the basic-syntax flag (B).
80.5% are coordinated or multi-part (C), 70.3% are questions (I), 51.2% contain typos (Z), 49.7% are colloquial (Q),
42.9% are polite (P) and 14.6% contain offensive words (W). Abbreviations (E, 10.7%) and morphological variation
(M, 7.0%) are rarer. These flags let us measure robustness separately for each kind of variation.

### 3.2 Length, vocabulary and n-grams

Queries are short and narrow; responses are long (Table 3.2, Figure 3.3, `eda_length_histograms.png`).

**Table 3.2. Lengths (words).**

| Field | Mean | Median | 90th pct. | Max |
|---|---|---|---|---|
| Query | 11.1 | 11 | 14 | 23 |
| Response | 148.6 | 141 | 220 | 431 |

In BERT word pieces the longest query is 27 tokens, so no query exceeds the 64-token limit we use for fine-tuning.
The query vocabulary is small: 3,432 word types over 283,089 tokens (type-token ratio 0.012), and 1,702 types occur
only once, mostly typos. The most frequent terms per intent are the task words themselves ("activate", "mobile",
"card" for `activate_card`; "lock", "block", "stolen", "lost" for `block_card`; `eda_terms_per_intent.csv`). The word
cloud and n-gram plots are in `eda_wordcloud.png` and `eda_top_ngrams.png`.

### 3.3 Linguistic features

On a 3,000-query sample, spaCy tags verbs (21%), pronouns (20%) and nouns (16%) most often (`eda_pos.csv`). This fits
short imperative requests such as "I want to cancel my card". Named entities are rare in queries. Most are card brands
or "ATM", often mis-tagged as ORG or PERSON. Responses contain many cardinal numbers (6,490), which come from numbered
steps rather than facts (`eda_entities_*.csv`).

### 3.4 Embedding map and intent similarity

Sentence embeddings (all-MiniLM-L6-v2) projected with UMAP form tight, well-separated clusters, one or more per intent
(Figure 3.4, `eda_embedding_map.png`). This is another sign of templating. The most similar intent pairs, by
cosine of mean embeddings, are the expected confusions (Table 3.3, Figure 3.5, `eda_intent_similarity_heatmap.png`).
They also explain why the agent should filter retrieval by intent only when the classifier is confident.

**Table 3.3. Most similar intent pairs.**

| Intent A | Intent B | Cosine |
|---|---|---|
| get_password | set_up_password | 0.92 |
| cancel_loan | cancel_mortgage | 0.87 |
| check_loan_payments | check_mortgage_payments | 0.85 |
| activate_card | activate_card_international_usage | 0.81 |
| apply_for_loan | apply_for_mortgage | 0.79 |

### 3.5 Synthetic-data checks

**Templating.** Every query and response is distinct as a whole, but only 25% of queries and 4.6% of responses have a
distinct opening 5-gram. 18.3% of responses start with "I'm here to assist you" (`eda_template_diversity.csv`).
Similarity to the reference answer is therefore a weak quality signal. We rely on a rubric scored by humans and an LLM
judge instead.

**Placeholders.** 55% of responses (14,064) contain `{{placeholders}}` instead of concrete facts, and no query does
(`eda_placeholders.csv`). The most common are Customer Support Phone Number (7,879), Company Website URL (6,063), Credit
Card (5,310) and Customer Support Working Hours (3,024). Our shared policy prompt asks every system to use the same
placeholder style instead of inventing phone numbers, URLs or fees, and an automatic check counts invented specifics.

**Anomalies.** The shortest and longest queries are valid (for example "block a Visa" and long multi-part transfer
requests). Three responses are refusals ("I'm unable to assist…"), and a few responses have unbalanced placeholder
braces. Both are negligible but are noted because they sit in the training split and therefore in the knowledge base.

### 3.6 Synthetic vs real queries

BANKING77 queries have a similar mean length to Bitext (11.8 vs 11.1 words) but a very different distribution
(Figure 3.6, `eda_synthetic_vs_real_length.png`). Real queries peak at about 8 words, while Bitext peaks at 11-13.
Real queries also have a long tail beyond 40 words, while no Bitext query exceeds 23.

The vocabularies differ even more. Only 10% of BANKING77 word types appear in Bitext queries, which covers 69% of
BANKING77 tokens. Frequent real words that never appear in Bitext include *why*, *pending*, *declined*, *charged*,
*refund*, *exchange*, *app* and *pin*. Real customers often ask *why* something happened to a specific transaction.
Bitext customers only ask *how to* perform a task.

### 3.7 Implications for modelling

1. **Expect a ceiling on Bitext and a drop on BANKING77.** The TF-IDF baseline reaches 0.999 accuracy on the Bitext
   test split but 0.410 on mapped BANKING77 queries (0.554 with high-confidence mappings only).
2. **Use a rubric, not reference similarity**, because responses are templated.
3. **Enforce placeholders** through one shared policy and measure invented specifics automatically.
4. **Treat escalation as a first-class outcome.** Most real banking requests in BANKING77 cannot be answered from the
   knowledge base, so the evaluation rewards escalating them.
5. **Expect confusions between related intents** (password, loan vs mortgage), and gate the agent's intent filter
   on classifier confidence.
