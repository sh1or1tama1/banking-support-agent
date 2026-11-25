"""Task 1: strict vs lenient scores for the LLM few-shot classifier on BANKING77.

The few-shot prompt lists intents as "CATEGORY: intent". On 108 of 520 BANKING77 queries the LLM
returned only the category "FEES". The FEES category contains a single intent (check_fees), so the
answer is unambiguous; strict scoring counts it as INVALID. This script reports both, using the
saved predictions (no LLM calls) and the same metric function as notebook 06.

    python scripts/task1_lenient_scores.py   ->  outputs/tables/task1_banking77_strict_vs_lenient.csv
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.classify import metrics as M

pred = pd.read_csv(C.TABLES / "task1_banking77_predictions.csv")
# Categories that hold exactly one intent can be mapped back without guessing.
bitext = pd.read_parquet(C.PROCESSED / "bitext_splits.parquet")
single = bitext.groupby("category")["intent"].unique()
single = {cat.upper(): ints[0] for cat, ints in single.items() if len(ints) == 1}


def category_only(raw: str):
    try:
        lab = json.loads(raw)["intent"].strip()
    except Exception:
        return None
    return single.get(lab.upper())


lenient = pred["llm"].where(pred["llm"] != "INVALID", pred["llm_raw"].map(category_only).fillna("INVALID"))
rows = []
for name, col in (("tfidf_lr", pred["tfidf"]), ("bert_finetuned", pred["bert"]),
                  ("llm_fewshot (strict)", pred["llm"]), ("llm_fewshot (lenient: FEES -> check_fees)", lenient)):
    s = M.summary(pred["gold_intent"], col, name, "banking77_mapped", float("nan"))
    s["invalid_label_rate"] = round((col == "INVALID").mean(), 4)
    rows.append(s)
out = pd.DataFrame(rows).drop(columns=["ms_per_query"], errors="ignore")
out.to_csv(C.TABLES / "task1_banking77_strict_vs_lenient.csv", index=False)
print("single-intent categories:", single)
print(out.to_string(index=False))
